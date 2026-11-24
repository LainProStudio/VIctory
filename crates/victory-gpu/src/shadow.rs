//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! in GPU-local memory (`D3D12_HEAP_TYPE_DEFAULT`) on first use, and the
//! ranges each submission reads (vertex, index and constant data, buffer
//! views) are brought up to date from title memory before it runs via pooled
//! `UPLOAD` staging copies: the 64 KiB pages the title wrote since they were
//! last copied ([`crate::watch`]: write watching, or read-only guards on
//! physical-page views), all of it where neither works. A title GPU address
//! inside the buffer is the same offset in its copy. Ranges shaders write
//! are read back through pooled `READBACK` staging and merged into title
//! memory after the submission ([`Shadow::gpu_writes`]).
//!
//! Mirrored buffer SRVs in bound heaps also refresh at submission, since
//! a shader may select their index dynamically. Directly indexed UAVs
//! still need read/write tracking independent of conventional table use.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

#[cfg(test)]
mod submission_tests;

/// One staged upload copy to record onto the submission's upload list.
pub(crate) struct UploadCopy {
    pub(crate) shadow: Arc<Shadow>,
    pub(crate) src: ID3D12Resource,
    pub(crate) src_offset: u64,
    pub(crate) dst_offset: u64,
    pub(crate) len: u64,
}

/// One CPU->upload-staging byte copy collected during `Shadow` staging so
/// multiple staging copies can run concurrently across worker threads.
pub(crate) struct StagedCopy {
    pub(crate) dst: usize,
    pub(crate) src: usize,
    pub(crate) len: usize,
}

/// A title buffer's host copy in `D3D12_HEAP_TYPE_DEFAULT` memory.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
}

#[cfg(feature = "diagnostics")]
fn log_copyback() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| crate::switch::on("VICTORY_LOG_COPYBACK"))
}

// Snapshot byte-compare scratch buffer (avoids 64 KiB heap alloc per `copy_back`).
thread_local! {
    static COPY_BACK_SCRATCH: std::cell::RefCell<Vec<u8>> =
        std::cell::RefCell::new(vec![0u8; PAGE]);
    #[cfg(test)]
    static BEFORE_COPY_BACK_REARM: std::cell::RefCell<Option<Box<dyn FnMut()>>> =
        const { std::cell::RefCell::new(None) };
}

/// Whether `[start, start + len)` lies inside a single committed writable
/// region (`PAGE_READWRITE` or `PAGE_EXECUTE_READWRITE`).
fn is_writable_commit(start: usize, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_READWRITE, VirtualQuery,
    };
    let Some(end) = start.checked_add(len) else {
        return false;
    };
    // SAFETY: plain data filled in by `VirtualQuery`.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: `VirtualQuery` inspects the address range without dereferencing `start`.
    let n = unsafe { VirtualQuery(start as *const _, &mut info, size_of_val(&info)) };
    n == size_of_val(&info)
        && info.State == MEM_COMMIT
        && matches!(info.Protect, PAGE_READWRITE | PAGE_EXECUTE_READWRITE)
        && (info.BaseAddress as usize) <= start
        && (info.BaseAddress as usize).saturating_add(info.RegionSize) >= end
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; mutable state is mutex-guarded.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

/// Coalesces overlapping or adjacent `[from, to)` byte ranges in place.
pub(crate) fn merge_ranges(ranges: &mut Vec<(usize, usize)>) {
    ranges.retain(|&(a, b)| a < b);
    if ranges.len() <= 1 {
        return;
    }
    ranges.sort_unstable();
    let mut out = 0;
    for i in 1..ranges.len() {
        let (a, b) = ranges[i];
        if a <= ranges[out].1 {
            ranges[out].1 = ranges[out].1.max(b);
        } else {
            out += 1;
            ranges[out] = (a, b);
        }
    }
    ranges.truncate(out + 1);
}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    pub(crate) fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (
            from - start,
            (address.saturating_add(len)).clamp(from, end) - start,
        )
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: records that a submitted
    /// list's shaders may write `[address, address + len)` so later refreshes
    /// in the same submission do not overwrite it.
    pub(crate) fn mark_pending(&self, address: usize, len: usize) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address.saturating_add(len)));
    }

    /// After the submission completes and readbacks are merged: clears all
    /// pending shader-write ranges.
    pub(crate) fn clear_pending(&self) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// After the submission ran: compares the readback snapshots taken before
    /// (`old`) and after (`after`) the submission for buffer offset range
    /// `[from, from + old.len())`, and copies only the GPU-modified byte runs
    /// back into title memory (unmapped pages left alone). Preserves any
    /// concurrent CPU writes to other bytes on the same page; pages without
    /// concurrent CPU writes are marked current after re-arming write-watch so
    /// our own write-back is not re-uploaded on the next submission. Returns
    /// the bytes written.
    pub(crate) fn copy_back_range(&self, from: usize, old: &[u8], after: &[u8]) -> usize {
        debug_assert_eq!(old.len(), after.len());
        let len = old.len().min(after.len());
        if len == 0 {
            return 0;
        }
        let start = self.start as usize;
        let shadow_len = self.len as usize;
        let to = from + len;
        let mut written = 0;
        const PIECE: usize = 64 << 10;
        COPY_BACK_SCRATCH.with(|buf| {
            let mut title_buf = buf.borrow_mut();
            if title_buf.len() < PIECE {
                title_buf.resize(PIECE, 0);
            }
            let mut dirty_runs: Vec<(usize, usize)> = Vec::new();
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                let rel = at - from;
                let host = &after[rel..rel + n];
                let prev = &old[rel..rel + n];
                let title = &mut title_buf[..n];
                if host != prev && crate::host::read_into((start + at) as *const u8, title) {
                    dirty_runs.clear();
                    let mut dirty_start: Option<usize> = None;
                    let mut cpu_diff = false;
                    let mut i = 0usize;
                    while i < n {
                        if i + 8 <= n && host[i..i + 8] == prev[i..i + 8] {
                            if let Some(s) = dirty_start.take() {
                                dirty_runs.push((s, i));
                            }
                            if host[i..i + 8] != title[i..i + 8] {
                                cpu_diff = true;
                            }
                            i += 8;
                            continue;
                        }
                        let end_j = (i + 8).min(n);
                        for j in i..end_j {
                            let h = host[j];
                            let t_byte = title[j];
                            if h != prev[j] {
                                if t_byte != h {
                                    title[j] = h;
                                    dirty_start.get_or_insert(j);
                                }
                            } else {
                                if let Some(s) = dirty_start.take() {
                                    dirty_runs.push((s, j));
                                }
                                if t_byte != h {
                                    cpu_diff = true;
                                }
                            }
                        }
                        i = end_j;
                    }
                    if let Some(s) = dirty_start {
                        dirty_runs.push((s, n));
                    }
                    if !dirty_runs.is_empty() {
                        let chunk_start = start + at;
                        crate::watch::writing(chunk_start, n);
                        let direct = dirty_runs.len() > 16 && is_writable_commit(chunk_start, n);
                        for &(s, e) in &dirty_runs {
                            let run_len = e - s;
                            if direct {
                                // SAFETY: `watch::writing` opened guarded blocks and
                                // `is_writable_commit` verified `[chunk_start, chunk_start + n)`
                                // is committed and writable.
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        title[s..].as_ptr(),
                                        (chunk_start + s) as *mut u8,
                                        run_len,
                                    );
                                }
                            } else {
                                crate::host::write_mapped(
                                    (chunk_start + s) as *mut u8,
                                    title[s..].as_ptr(),
                                    run_len,
                                );
                            }
                            written += run_len;
                        }
                    }
                    // Only mark the page synced if title memory and VRAM now match
                    // byte for byte; if the title concurrently wrote other bytes on
                    // this page, leave it dirty so the next refresh uploads them.
                    if !cpu_diff {
                        synced.push((start + at) / PAGE);
                    }
                }
                at += n;
            }
            // Re-arm write-watch only on contiguous runs of full pages synced above
            // so unrelated untouched subranges avoid multi-MB kernel queries.
            synced.retain(|&p| {
                let Some(pf) = (p * PAGE).checked_sub(start) else {
                    return false;
                };
                let pt = pf + PAGE;
                pt <= shadow_len && from <= pf && pt <= to
            });
            synced.sort_unstable();
            synced.dedup();
            if !synced.is_empty() && self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
                let base = start / PAGE;
                let mut idx = 0;
                while idx < synced.len() {
                    let run_start = synced[idx];
                    let mut run_end = run_start + 1;
                    while idx + 1 < synced.len() && synced[idx + 1] == run_end {
                        idx += 1;
                        run_end += 1;
                    }
                    #[cfg(test)]
                    BEFORE_COPY_BACK_REARM.with(|hook| {
                        if let Some(f) = hook.borrow_mut().as_mut() {
                            f();
                        }
                    });
                    if let Some((now, _)) =
                        crate::watch::writes(run_start * PAGE, run_end * PAGE, true)
                    {
                        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                        for page in run_start..run_end {
                            let title_page = &mut title_buf[..PAGE];
                            let pf = page * PAGE - start;
                            let host_page = &after[pf - from..pf - from + PAGE];
                            if crate::host::read_into((page * PAGE) as *const u8, title_page)
                                && title_page == host_page
                            {
                                copied[page - base] = now;
                            }
                        }
                    }
                    idx += 1;
                }
            }
        });
        #[cfg(feature = "diagnostics")]
        if written > 0 && log_copyback() {
            tracing::info!(
                shadow_start = format_args!("{:#x}", self.start),
                from,
                len,
                written,
                "shadow copy_back"
            );
        }
        written
    }

    /// Whether `[from, to)` on this shadow can safely consume a pre-fetched
    /// cross-shadow `watch::writes` answer (no overlapping busy shader writes
    /// and not already known unwatched).
    pub(crate) fn can_use_prefetched(&self, from: usize, to: usize) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to || self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return false;
        }
        !self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|&(a, e)| a < to && e > from)
    }

    /// Updates `[address, address + len)` using a pre-fetched cross-shadow
    /// write-watch answer when eligible, otherwise falls back to [`Shadow::update`].
    /// When `deferred` is `Some`, staging memory is allocated and queued in
    /// `deferred` without executing `copy_mapped` inline.
    pub(crate) fn update_with_watch_into(
        self: &Arc<Self>,
        address: usize,
        len: usize,
        prefetched: Option<(u64, &[(usize, u64)])>,
        alloc: &mut impl FnMut(usize) -> Option<(ID3D12Resource, u64, *mut u8)>,
        out: &mut Vec<UploadCopy>,
        deferred: Option<&mut Vec<StagedCopy>>,
    ) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), address.saturating_add(len).min(end));
        if from >= to {
            return 0;
        }
        if let Some((now, written)) = prefetched
            && self.can_use_prefetched(from, to)
        {
            self.watched.store(1, Ordering::Relaxed);
            let copied = self.copy_stale_into(from, to, now, written, alloc, out, deferred);
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::Checked, (to - from) as u64);
                add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
            }
            return copied;
        }
        self.update_into(from, to - from, alloc, out, deferred)
    }

    /// At submission: stages what changed of `[address, address + len)` since
    /// it was last copied (all of it if the memory is not write-watched) into
    /// upload memory allocated via `alloc`, pushing GPU copy descriptors onto
    /// `out`. Returns the bytes staged.
    #[cfg(test)]
    pub(crate) fn update(
        self: &Arc<Self>,
        address: usize,
        len: usize,
        alloc: &mut impl FnMut(usize) -> Option<(ID3D12Resource, u64, *mut u8)>,
        out: &mut Vec<UploadCopy>,
    ) -> usize {
        self.update_into(address, len, alloc, out, None)
    }

    fn update_into(
        self: &Arc<Self>,
        address: usize,
        len: usize,
        alloc: &mut impl FnMut(usize) -> Option<(ID3D12Resource, u64, *mut u8)>,
        out: &mut Vec<UploadCopy>,
        deferred: Option<&mut Vec<StagedCopy>>,
    ) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), address.saturating_add(len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: one `watch::writes` for the
            // bounding span, then `copy_stale` on each gap so busy pages are
            // left untouched and still keep their old `copied` watermark.
            busy.sort_unstable();
            let mut gaps: Vec<(usize, usize)> = Vec::new();
            let mut at = from;
            for (a, e) in busy {
                if a > at {
                    gaps.push((at, a));
                }
                at = at.max(e);
            }
            if at < to {
                gaps.push((at, to));
            }
            return if gaps.is_empty() {
                0
            } else {
                self.update_gaps_into(&gaps, alloc, out, deferred)
            };
        }
        self.update_free_into(from, to, alloc, out, deferred)
    }

    /// [`Shadow::update`] of disjoint non-busy `gaps` (`from < to`, in ascending
    /// order) using a single write-watch query across the bounding span.
    fn update_gaps_into(
        self: &Arc<Self>,
        gaps: &[(usize, usize)],
        alloc: &mut impl FnMut(usize) -> Option<(ID3D12Resource, u64, *mut u8)>,
        out: &mut Vec<UploadCopy>,
        mut deferred: Option<&mut Vec<StagedCopy>>,
    ) -> usize {
        #[cfg(feature = "diagnostics")]
        let span: usize = gaps.iter().map(|&(f, t)| t - f).sum();
        let (g_from, g_to) = (gaps[0].0, gaps[gaps.len() - 1].1);
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(g_from, g_to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let mut copied = 0usize;
                for &(f, t) in gaps {
                    copied += self.copy_stale_into(
                        f,
                        t,
                        now,
                        &written,
                        alloc,
                        out,
                        deferred.as_deref_mut(),
                    );
                }
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, span as u64);
                    add(C::Unchanged, span.saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        let mut staged = 0usize;
        for &(f, t) in gaps {
            staged += self.stage_into(f, t, alloc, out, deferred.as_deref_mut());
        }
        staged
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free_into(
        self: &Arc<Self>,
        from: usize,
        to: usize,
        alloc: &mut impl FnMut(usize) -> Option<(ID3D12Resource, u64, *mut u8)>,
        out: &mut Vec<UploadCopy>,
        deferred: Option<&mut Vec<StagedCopy>>,
    ) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale_into(from, to, now, &written, alloc, out, deferred);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.stage_into(from, to, alloc, out, deferred)
    }

    /// Stages the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    #[allow(clippy::too_many_arguments)]
    fn copy_stale_into(
        self: &Arc<Self>,
        from: usize,
        to: usize,
        now: u64,
        written: &[(usize, u64)],
        alloc: &mut impl FnMut(usize) -> Option<(ID3D12Resource, u64, *mut u8)>,
        out: &mut Vec<UploadCopy>,
        mut deferred: Option<&mut Vec<StagedCopy>>,
    ) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let start_idx = written.partition_point(|&(p, _)| p < from / PAGE);
        let mut writes = written[start_idx..].iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.stage_into(
                    (first * PAGE).max(from),
                    (page * PAGE).min(to),
                    alloc,
                    out,
                    deferred.as_deref_mut(),
                );
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.stage_into(
                (first * PAGE).max(from),
                ((last + 1) * PAGE).min(to),
                alloc,
                out,
                deferred,
            );
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into upload
    /// staging memory and records the GPU copy into this default-heap buffer.
    fn stage_into(
        self: &Arc<Self>,
        from: usize,
        to: usize,
        alloc: &mut impl FnMut(usize) -> Option<(ID3D12Resource, u64, *mut u8)>,
        out: &mut Vec<UploadCopy>,
        deferred: Option<&mut Vec<StagedCopy>>,
    ) -> usize {
        const STAGE_STRIPE: usize = 1 << 20;
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        let len = to.saturating_sub(from);
        if len == 0 {
            return 0;
        }
        let Some((src, src_offset, dst_ptr)) = alloc(len) else {
            return 0;
        };
        if let Some(vec) = deferred {
            let mut off = 0usize;
            while off < len {
                let chunk = (len - off).min(STAGE_STRIPE);
                vec.push(StagedCopy {
                    dst: dst_ptr as usize + off,
                    src: from + off,
                    len: chunk,
                });
                off += chunk;
            }
        } else {
            crate::host::copy_mapped(dst_ptr, from as *const u8, len, false);
        }
        out.push(UploadCopy {
            shadow: self.clone(),
            src,
            src_offset,
            dst_offset: (from - start) as u64,
            len: len as u64,
        });
        len
    }
}

struct Entry {
    end: u64,
    shadow: Arc<OnceLock<Option<Arc<Shadow>>>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The longest title buffer registered so far (it stays after the buffer is
/// released): a buffer starting further below an address than this cannot
/// hold it. Changed under `BUFFERS`' lock.
static MAX_SPAN: AtomicU64 = AtomicU64::new(0);

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    let mut map = buffers();
    MAX_SPAN.fetch_max(len, Ordering::Relaxed);
    map.insert(
        (start, resource),
        Entry {
            end: start.saturating_add(len),
            shadow: Arc::new(OnceLock::new()),
        },
    );
}

/// The title released it; any live host copy is retired behind the queue.
pub(crate) fn unregister(resource: usize, start: u64) {
    if let Some(entry) = buffers().remove(&(start, resource))
        && let Some(Some(shadow)) = entry.shadow.get()
    {
        crate::host::retire(shadow.clone());
    }
}

/// The innermost entry holding `[address, end)`: the one starting closest
/// below `address` that reaches `end`, among those starting at most `span`
/// (the longest entry's length) below `end`. Big buffers hold many smaller
/// ones (a heap's placed buffers), so the one holding an address past
/// those can be any number of entries back.
fn enclosing(map: &BTreeMap<(u64, usize), Entry>, span: u64, address: u64, end: u64) -> Option<(u64, usize)> {
    let lowest = end.saturating_sub(span);
    if lowest > address {
        return None;
    }
    map.range((lowest, 0)..=(address, usize::MAX))
        .rev()
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k)
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    if address < 0x10000 {
        return None;
    }
    let end = address.checked_add(len.max(1))?;
    let (key, len, slot) = {
        let map = buffers();
        let key = enclosing(&map, MAX_SPAN.load(Ordering::Relaxed), address, end);
        let Some(key) = key else {
            drop(map);
            return loose(address, end);
        };
        let entry = map.get(&key)?;
        (key, entry.end - key.0, entry.shadow.clone())
    };
    slot.get_or_init(|| {
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // GPU-local default heap so shaders read/write VRAM at full bandwidth.
        let buffer = host()?.default_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        Some(Arc::new(Shadow::new(buffer, gpu, key.0, len)))
    })
    .clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

/// The longest window in `LOOSE`. Changed under its lock.
static LOOSE_SPAN: AtomicU64 = AtomicU64::new(0);

/// Below this nothing is ever mapped (the null page and its guard area).
const LOWEST_MAPPED: u64 = 0x10000;

/// The window copy holding `[address, end)` among `windows` (by start and
/// length, none longer than `span`), the innermost if several do.
fn enclosing_window<V>(windows: &BTreeMap<(u64, u64), V>, span: u64, address: u64, end: u64) -> Option<&V> {
    let lowest = end.saturating_sub(span);
    if lowest > address {
        return None;
    }
    windows
        .range((lowest, 0)..=(address, u64::MAX))
        .rev()
        .find(|&(&(s, l), _)| s.saturating_add(l) >= end)
        .map(|(_, v)| v)
}

/// The window copy for `[address, end)` outside buffers: one already made
/// that holds it (a range spanning two windows has one of its own, which
/// later ranges in it share), else a new one. None for memory that is not
/// committed (a stale or null address): 4 MiB of zeros would not help.
fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    if address < LOWEST_MAPPED {
        return None;
    }
    let start = address & !(WINDOW - 1);
    let end = end.max(address.saturating_add(64 << 10));
    let len = end.checked_next_multiple_of(WINDOW)? - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = enclosing_window(&map, LOOSE_SPAN.load(Ordering::Relaxed), address, end) {
        return Some(shadow.clone());
    }
    if !crate::texture::committed(address as *const u8, 1) {
        return None;
    }
    let buffer = host()?.default_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, gpu, start, len));
    LOOSE_SPAN.fetch_max(len, Ordering::Relaxed);
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| {
            e.shadow
                .get()
                .and_then(|o| o.as_deref())
                .map(|s| (s.gpu, s.start, s.len))
        })
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D12::{D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    struct WatchedAlloc {
        ptr: *mut u8,
        len: usize,
        resource_id: usize,
    }

    impl WatchedAlloc {
        fn new(len: usize, resource_id: usize) -> Self {
            // SAFETY: fresh write-watched virtual allocation, freed in Drop.
            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            }
            .cast::<u8>();
            assert!(!ptr.is_null());
            register(resource_id, ptr as u64, len as u64);
            Self {
                ptr,
                len,
                resource_id,
            }
        }

        fn addr(&self) -> u64 {
            self.ptr as u64
        }

        fn bytes_mut(&mut self) -> &mut [u8] {
            // SAFETY: exclusive access to our committed allocation.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    impl Drop for WatchedAlloc {
        fn drop(&mut self) {
            unregister(self.resource_id, self.addr());
            // SAFETY: frees the allocation obtained from VirtualAlloc.
            unsafe { VirtualFree(self.ptr.cast(), 0, MEM_RELEASE) };
        }
    }

    fn title_pages(pages: usize) -> Option<(Vec<u8>, usize, Arc<Shadow>, Vec<u8>)> {
        let memory = vec![0u8; (pages + 1) * PAGE];
        let base = (memory.as_ptr() as usize).next_multiple_of(PAGE);
        let buffer = host()?.default_buffer((pages * PAGE) as u64)?;
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Arc::new(Shadow::new(buffer, gpu, base as u64, (pages * PAGE) as u64));
        let staging = vec![0u8; pages * PAGE];
        Some((memory, base, shadow, staging))
    }

    fn stage_into(
        shadow: &Arc<Shadow>,
        address: usize,
        len: usize,
        staging: &mut [u8],
    ) -> usize {
        let base = shadow.start as usize;
        let mut uploads = Vec::new();
        let staged = shadow.update(
            address,
            len,
            &mut |n| {
                let (b, p) = host()?.upload_buffer(n as u64)?;
                Some((b, 0, p))
            },
            &mut uploads,
        );
        for u in &uploads {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: upload buffer mapped for `u.len` bytes; `dst_offset..dst_offset+len` in `staging`.
            unsafe {
                if u.src.Map(0, None, Some(&mut ptr)).is_ok() {
                    std::ptr::copy_nonoverlapping(
                        ptr.cast::<u8>().add(u.src_offset as usize),
                        staging[u.dst_offset as usize..].as_mut_ptr(),
                        u.len as usize,
                    );
                    u.src.Unmap(0, None);
                }
            }
        }
        let _ = base;
        staged
    }

    /// A write of one byte: the title's into its memory, or a shader's
    /// into a readback buffer.
    fn poke(at: usize, value: u8) {
        // SAFETY: the tests pass addresses inside the memory.
        unsafe { (at as *mut u8).write_volatile(value) }
    }

    fn peek(at: usize) -> u8 {
        // SAFETY: as in `poke`.
        unsafe { (at as *const u8).read_volatile() }
    }

    #[test]
    fn merge_ranges_coalesces_overlapping_and_adjacent() {
        let mut empty = Vec::new();
        merge_ranges(&mut empty);
        assert!(empty.is_empty());

        let mut ranges = vec![
            (64, 64),
            (200, 100),
            (128, 256),
            (0, 64),
            (64, 96),
            (96, 128),
            (512, 600),
            (550, 700),
        ];
        merge_ranges(&mut ranges);
        assert_eq!(ranges, vec![(0, 256), (512, 700)]);
    }

    #[test]
    fn default_vram_shadow_uploads_only_dirty_pages() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let mut mem = WatchedAlloc::new(2 * PAGE, 0x5000_0001);
        mem.bytes_mut().fill(0);
        mem.bytes_mut()[0..4].copy_from_slice(&0x1122_3344u32.to_le_bytes());
        mem.bytes_mut()[PAGE..PAGE + 4].copy_from_slice(&0x5566_7788u32.to_le_bytes());

        let shadow = find(mem.addr(), (2 * PAGE) as u64).expect("default shadow");

        // Verify the shadow buffer lives in GPU-local DEFAULT VRAM.
        let mut heap = D3D12_HEAP_PROPERTIES::default();
        // SAFETY: live D3D12 resource.
        unsafe { shadow.buffer.GetHeapProperties(Some(&mut heap), None) }.expect("heap props");
        assert_eq!(heap.Type, D3D12_HEAP_TYPE_DEFAULT);

        let (rb, rb_ptr) = host
            .readback_buffer((2 * PAGE) as u64)
            .expect("readback buffer");

        // Submission 1: initial upload copies both pages into DEFAULT VRAM.
        let list = host.new_list().expect("new list");
        shadow.refresh(&list, mem.addr(), (2 * PAGE) as u64);
        // SAFETY: list is open; buffers stay alive through execute.
        unsafe {
            list.list
                .CopyBufferRegion(&rb, 0, &shadow.buffer, 0, (2 * PAGE) as u64);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: plain device-removed check.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        // SAFETY: fence completed; readback buffer is mapped for 2 * PAGE bytes.
        let got = unsafe { std::slice::from_raw_parts(rb_ptr, 2 * PAGE) };
        assert_eq!(&got[0..4], &0x1122_3344u32.to_le_bytes());
        assert_eq!(&got[PAGE..PAGE + 4], &0x5566_7788u32.to_le_bytes());

        // Unchanged check: zero dirty pages -> zero bytes staged.
        assert_eq!(shadow.watched.load(Ordering::Relaxed), 1);
        let mut uploads = Vec::new();
        let staged = shadow.update(
            mem.addr() as usize,
            2 * PAGE,
            &mut |n| {
                let (b, p) = host.upload_buffer(n as u64)?;
                Some((b, 0, p))
            },
            &mut uploads,
        );
        assert_eq!(staged, 0);
        assert!(uploads.is_empty());

        // Dirty only page 1: only page 1 is staged at dst_offset == PAGE.
        mem.bytes_mut()[PAGE..PAGE + 4].copy_from_slice(&0x99AA_BBCCu32.to_le_bytes());
        let mut uploads = Vec::new();
        let staged = shadow.update(
            mem.addr() as usize,
            2 * PAGE,
            &mut |n| {
                let (b, p) = host.upload_buffer(n as u64)?;
                Some((b, 0, p))
            },
            &mut uploads,
        );
        assert_eq!(staged, PAGE);
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].dst_offset, PAGE as u64);
        assert_eq!(uploads[0].len, PAGE as u64);

        // Submission 2: write page 1 again and verify GPU sees new page 1 + original page 0.
        mem.bytes_mut()[PAGE..PAGE + 4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        list.reset();
        shadow.refresh(&list, mem.addr(), (2 * PAGE) as u64);
        // SAFETY: list is open; buffers stay alive through execute.
        unsafe {
            list.list
                .CopyBufferRegion(&rb, 0, &shadow.buffer, 0, (2 * PAGE) as u64);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: plain device-removed check.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        // SAFETY: fence completed; readback buffer is mapped for 2 * PAGE bytes.
        let got = unsafe { std::slice::from_raw_parts(rb_ptr, 2 * PAGE) };
        assert_eq!(&got[0..4], &0x1122_3344u32.to_le_bytes());
        assert_eq!(&got[PAGE..PAGE + 4], &0xDEAD_BEEFu32.to_le_bytes());
    }

    #[test]
    fn gpu_writes_readback_merges_and_preserves_concurrent_cpu_writes() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let mut mem = WatchedAlloc::new(PAGE, 0x5000_0002);
        mem.bytes_mut().fill(0);
        let initial: [u32; 5] = [u32::MAX, 1, 0, 0, 0x1234_5678];
        for (i, word) in initial.iter().enumerate() {
            mem.bytes_mut()[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
        }

        let shadow = find(mem.addr(), PAGE as u64).expect("default shadow");
        let list = host.new_list().expect("new list");
        shadow.refresh(&list, mem.addr(), PAGE as u64);
        shadow.gpu_writes(&list, mem.addr(), 64);
        let bounded = crate::indirect::bound(
            &list,
            (&shadow.buffer, shadow.offset(mem.addr())),
            None,
            (0, 16, 1),
        )
        .expect("indirect bound");
        // `indirect::bound` writes clamped arguments into `bounded`; copy them into
        // `shadow.buffer` on the GPU so readback merges the GPU write into title memory.
        // SAFETY: open list and live resources in their tracked states (`BUFFER_READ` / `INDIRECT_ARGUMENT`).
        unsafe {
            use windows::Win32::Graphics::Direct3D12::{
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATES,
            };
            let read_state = D3D12_RESOURCE_STATES(crate::host::BUFFER_READ as i32);
            list.list.ResourceBarrier(&[
                crate::host::transition(&shadow.buffer, read_state, D3D12_RESOURCE_STATE_COPY_DEST),
                crate::host::transition(
                    &bounded,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                ),
            ]);
            list.list
                .CopyBufferRegion(&shadow.buffer, shadow.offset(mem.addr()), &bounded, 0, 16);
            list.list.ResourceBarrier(&[
                crate::host::transition(&shadow.buffer, D3D12_RESOURCE_STATE_COPY_DEST, read_state),
                crate::host::transition(
                    &bounded,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                ),
            ]);
        }
        crate::indirect::done(&list, &bounded);
        list.close();
        host.execute(&[&list]);
        // SAFETY: plain device-removed check.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        let clamped = u32::from_le_bytes(mem.bytes_mut()[0..4].try_into().unwrap());
        let sentinel = u32::from_le_bytes(mem.bytes_mut()[16..20].try_into().unwrap());
        assert_eq!(clamped, 1 << 24);
        assert_eq!(sentinel, 0x1234_5678);

        // Flush write-watch state so the page starts clean.
        let mut uploads = Vec::new();
        shadow.update(
            mem.addr() as usize,
            PAGE,
            &mut |n| {
                let (b, p) = host.upload_buffer(n as u64)?;
                Some((b, 0, p))
            },
            &mut uploads,
        );
        uploads.clear();
        assert_eq!(
            shadow.update(
                mem.addr() as usize,
                PAGE,
                &mut |n| {
                    let (b, p) = host.upload_buffer(n as u64)?;
                    Some((b, 0, p))
                },
                &mut uploads,
            ),
            0
        );

        // Case 1: GPU-only write back on a whole page marks the page synced so our
        // own write-back is not re-uploaded on the next submission.
        let old = mem.bytes_mut().to_vec();
        let mut after = old.clone();
        after[0..4].copy_from_slice(&0x1357_2468u32.to_le_bytes());
        assert_eq!(shadow.copy_back_range(0, &old, &after), 4);
        assert_eq!(
            u32::from_le_bytes(mem.bytes_mut()[0..4].try_into().unwrap()),
            0x1357_2468
        );
        uploads.clear();
        assert_eq!(
            shadow.update(
                mem.addr() as usize,
                PAGE,
                &mut |n| {
                    let (b, p) = host.upload_buffer(n as u64)?;
                    Some((b, 0, p))
                },
                &mut uploads,
            ),
            0
        );

        // Case 2: GPU write on [0..4) plus concurrent CPU write on [32..36) of the
        // same page merges GPU bytes, preserves CPU bytes, and leaves the page dirty
        // so the next upload pushes the merged page back to DEFAULT VRAM.
        let old = mem.bytes_mut().to_vec();
        let mut after = old.clone();
        after[0..4].copy_from_slice(&0xCAFE_BABEu32.to_le_bytes());
        mem.bytes_mut()[32..36].copy_from_slice(&0xFACE_FEEDu32.to_le_bytes());

        assert_eq!(shadow.copy_back_range(0, &old, &after), 4);
        assert_eq!(
            u32::from_le_bytes(mem.bytes_mut()[0..4].try_into().unwrap()),
            0xCAFE_BABE
        );
        assert_eq!(
            u32::from_le_bytes(mem.bytes_mut()[32..36].try_into().unwrap()),
            0xFACE_FEED
        );

        uploads.clear();
        let staged = shadow.update(
            mem.addr() as usize,
            PAGE,
            &mut |n| {
                let (b, p) = host.upload_buffer(n as u64)?;
                Some((b, 0, p))
            },
            &mut uploads,
        );
        assert_eq!(staged, PAGE);
        assert_eq!(uploads.len(), 1);
    }

    const MIB: u64 = 1 << 20;

    fn entry(start: u64, len: u64) -> Entry {
        Entry {
            end: start + len,
            shadow: Arc::new(OnceLock::new()),
        }
    }

    /// A 256 MiB parent with `children` 64 KiB buffers placed in its first
    /// part, as a heap's placed buffers.
    fn heap(base: u64, children: u64) -> BTreeMap<(u64, usize), Entry> {
        let mut map = BTreeMap::new();
        map.insert((base, 1), entry(base, 256 * MIB));
        for i in 0..children {
            map.insert(
                (base + i * 0x10000, 100 + i as usize),
                entry(base + i * 0x10000, 0x10000),
            );
        }
        map
    }

    #[test]
    fn a_parent_is_found_past_many_children() {
        let base = 0x10_0000_0000;
        let map = heap(base, 200);
        let span = 256 * MIB;
        // Past all 200 children (and the 64 entries looked at before).
        let at = base + 200 * 0x10000 + 0x100;
        assert_eq!(enclosing(&map, span, at, at + 16), Some((base, 1)));
        let first_64 = map
            .range(..=(at, usize::MAX))
            .rev()
            .take(64)
            .find(|(_, e)| e.end >= at + 16);
        assert!(first_64.is_none(), "the parent is more than 64 entries back");
        // Near the parent's end.
        let at = base + 256 * MIB - 16;
        assert_eq!(enclosing(&map, span, at, at + 16), Some((base, 1)));
        // Inside a child: the child, the innermost.
        let at = base + 150 * 0x10000 + 8;
        assert_eq!(
            enclosing(&map, span, at, at + 8),
            Some((base + 150 * 0x10000, 250))
        );
        // Across two children: only the parent holds it.
        let at = base + 151 * 0x10000 - 8;
        assert_eq!(enclosing(&map, span, at, at + 16), Some((base, 1)));
        // Past the parent, and before it.
        let at = base + 256 * MIB;
        assert_eq!(enclosing(&map, span, at, at + 1), None);
        assert_eq!(enclosing(&map, span, base - 16, base + 16), None);
        // Longer than any buffer: none, without looking.
        assert_eq!(enclosing(&map, span, base, base + span + 1), None);
        assert_eq!(enclosing(&map, 0, base, base + 1), None);
    }

    #[test]
    fn registered_buffers_widen_the_search() {
        // High above anything a test maps; resource numbers no live
        // resource has.
        let base = 0x7f00_0000_0000;
        let len = 512 * MIB;
        register(1, base, len);
        for i in 0..100u64 {
            register(2 + i as usize, base + i * 0x1000, 0x1000);
        }
        assert!(MAX_SPAN.load(Ordering::Relaxed) >= len);
        let at = base + len - 64;
        let found = enclosing(&buffers(), MAX_SPAN.load(Ordering::Relaxed), at, at + 64);
        for i in 0..100u64 {
            unregister(2 + i as usize, base + i * 0x1000);
        }
        unregister(1, base);
        assert_eq!(found, Some((base, 1)));
        assert!(
            buffers()
                .range((base, 0)..=(base + len, usize::MAX))
                .next()
                .is_none()
        );
    }

    #[test]
    fn loose_windows_are_reused_for_ranges_inside_them() {
        let w = WINDOW;
        let base = 0x20_0000_0000;
        let mut windows = BTreeMap::new();
        windows.insert((base, w), "first");
        // A range across the next two windows got one of its own.
        windows.insert((base + w, 2 * w), "spanning");
        let span = 2 * w;
        let find =
            |address: u64, len: u64| enclosing_window(&windows, span, address, address + len).copied();
        assert_eq!(find(base, 1), Some("first"));
        assert_eq!(find(base + w - 16, 16), Some("first"));
        // Inside the spanning window, also in its second half.
        assert_eq!(find(base + w + 8, 64), Some("spanning"));
        assert_eq!(find(base + 2 * w + 8, 64), Some("spanning"));
        assert_eq!(find(base + 3 * w - 8, 8), Some("spanning"));
        // Across the first window's end: none holds it whole.
        assert_eq!(find(base + w - 8, 16), None);
        // Past every window, before them, longer than any.
        assert_eq!(find(base + 3 * w, 1), None);
        assert_eq!(find(base - 8, 4), None);
        assert_eq!(find(base, 3 * w), None);
    }

    #[test]
    fn loose_memory_must_be_committed() {
        // The null page and its guard area.
        assert!(loose(0, 16).is_none());
        assert!(loose(0x1000, 0x1010).is_none());
        assert!(loose(LOWEST_MAPPED - 1, LOWEST_MAPPED + 1).is_none());
        // Reserved, not committed.
        // SAFETY: a fresh reservation, released below.
        let p = unsafe { VirtualAlloc(std::ptr::null(), WINDOW as usize, MEM_RESERVE, PAGE_READWRITE) }
            as u64;
        assert_ne!(p, 0);
        assert!(loose(p + 64, p + 128).is_none());
        // SAFETY: our reservation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        // Too close to the top to round up to a window.
        assert!(loose(u64::MAX - 8, u64::MAX).is_none());
    }

    #[test]
    fn copy_back_only_writes_gpu_modified_bytes() {
        let Some((_memory, base, s, mut staging)) = title_pages(1) else {
            return;
        };
        for off in 0..PAGE {
            poke(base + off, 0x11);
        }
        stage_into(&s, base, PAGE, &mut staging);
        let old = staging.clone();
        let mut after = old.clone();
        after[16..20].fill(0xaa);
        after[1024..1028].fill(0xbb);
        poke(base + 500, 0x77);
        let written = s.copy_back_range(0, &old, &after);
        assert_eq!(written, 8);
        assert_eq!(peek(base + 15), 0x11);
        assert_eq!(peek(base + 16), 0xaa);
        assert_eq!(peek(base + 20), 0x11);
        assert_eq!(peek(base + 500), 0x77);
        assert_eq!(peek(base + 1023), 0x11);
        assert_eq!(peek(base + 1024), 0xbb);
        assert_eq!(peek(base + 1028), 0x11);
    }

    #[test]
    fn update_with_multiple_busy_gaps_preserves_busy_ranges() {
        let Some((_memory, base, s, mut staging)) = title_pages(3) else {
            return;
        };
        let (p0, p1, p2) = (base, base + PAGE, base + 2 * PAGE);
        poke(p0 + 1, 0x10);
        poke(p1 + 1, 0x20);
        poke(p2 + 1, 0x30);
        stage_into(&s, base, 3 * PAGE, &mut staging);
        s.mark_pending(p1, PAGE);
        poke(p0 + 1, 0x11);
        poke(p1 + 1, 0x22);
        poke(p2 + 1, 0x33);
        stage_into(&s, base, 3 * PAGE, &mut staging);
        assert_eq!(staging[1], 0x11);
        assert_eq!(staging[PAGE + 1], 0x20, "busy middle page not overwritten");
        assert_eq!(staging[2 * PAGE + 1], 0x33);
        let old = staging[PAGE..2 * PAGE].to_vec();
        let mut after = old.clone();
        after[2] = 0xaa;
        assert_eq!(s.copy_back_range(PAGE, &old, &after), 1);
        s.clear_pending();
        assert_eq!(peek(p1 + 2), 0xaa);
        assert_eq!(peek(p1 + 1), 0x22);
    }

    #[test]
    fn copy_back_only_watches_synced_pages() {
        let Some((_memory, base, s, mut staging)) = title_pages(3) else {
            return;
        };
        for p in 0..3 {
            poke(base + p * PAGE + 4, 0x11 + p as u8);
        }
        stage_into(&s, base, 3 * PAGE, &mut staging);
        let old = staging.clone();
        let mut after = old.clone();
        after[PAGE + 8] = 0xef;
        poke(base + 12, 0x77);
        poke(base + 2 * PAGE + 12, 0x99);
        assert_eq!(s.copy_back_range(0, &old, &after), 1);
        assert_eq!(peek(base + PAGE + 8), 0xef);
        stage_into(&s, base, 3 * PAGE, &mut staging);
        assert_eq!(staging[12], 0x77);
        assert_eq!(staging[2 * PAGE + 12], 0x99);
    }

    #[test]
    fn copy_back_strided_writes_preserve_interleaved_cpu_bytes() {
        let Some((_memory, base, s, mut staging)) = title_pages(1) else {
            return;
        };
        for off in 0..PAGE {
            poke(base + off, 0x00);
        }
        stage_into(&s, base, PAGE, &mut staging);
        let old = staging.clone();
        let mut after = old.clone();

        // 1,000 interleaved 4-byte GPU and 4-byte CPU writes inside one 64 KiB
        // page: triggers the `dirty_runs.len() > 16` direct strided copy-back.
        for k in 0..1000usize {
            let gpu_off = k * 8;
            let cpu_off = gpu_off + 4;
            for b in 0..4 {
                after[gpu_off + b] = 0x80 | (((k + b) as u8) & 0x7f);
                poke(base + cpu_off + b, 0x40 | ((k as u8) & 0x3f));
            }
        }

        let written = s.copy_back_range(0, &old, &after);
        assert_eq!(written, 1000 * 4);
        for k in 0..1000usize {
            let gpu_off = k * 8;
            let cpu_off = gpu_off + 4;
            for b in 0..4 {
                assert_eq!(peek(base + gpu_off + b), 0x80 | (((k + b) as u8) & 0x7f));
                assert_eq!(peek(base + cpu_off + b), 0x40 | ((k as u8) & 0x3f));
            }
        }
        assert_eq!(peek(base + 1000 * 8), 0x00);
    }

    #[test]
    fn copy_back_keeps_page_stale_on_concurrent_title_write() {
        let Some(host) = host() else {
            return;
        };
        let Some(buffer) = host.default_buffer(PAGE as u64) else {
            return;
        };
        // SAFETY: a fresh write-watched page, freed at the end of the test.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0);
        crate::watch::forget_journal(base, PAGE);
        // SAFETY: `buffer` is a live host buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let s = Arc::new(Shadow::new(buffer, gpu, base as u64, PAGE as u64));
        let mut staging = vec![0u8; PAGE];

        poke(base + 16, 0x11);
        stage_into(&s, base, PAGE, &mut staging);
        let before_copy_back = s.copied.lock().unwrap()[0];
        assert_ne!(before_copy_back, 0);

        let old = staging.clone();
        let mut after = old.clone();
        after[32] = 0xaa;

        // Simulate a title CPU write landing after copy-back scanned the page
        // but before `watch::writes` clears the kernel write-watch bit.
        BEFORE_COPY_BACK_REARM.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move || {
                poke(base + 64, 0x77);
            }));
        });
        let written = s.copy_back_range(0, &old, &after);
        BEFORE_COPY_BACK_REARM.with(|hook| {
            *hook.borrow_mut() = None;
        });

        assert_eq!(written, 1);
        assert_eq!(peek(base + 32), 0xaa);
        assert_eq!(peek(base + 64), 0x77);
        assert_eq!(
            s.copied.lock().unwrap()[0],
            before_copy_back,
            "page watermark stays stale when title memory changed before re-arm"
        );

        // Next update must still pick up the concurrent title write.
        let staged = stage_into(&s, base, PAGE, &mut staging);
        assert_eq!(staged, PAGE);
        assert_eq!(staging[64], 0x77);

        crate::watch::forget_journal(base, PAGE);
        // SAFETY: our write-watched allocation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }
}

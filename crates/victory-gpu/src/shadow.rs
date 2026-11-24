//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! Mirrored buffer SRVs in bound heaps also refresh at submission, since
//! a shader may select their index dynamically. Directly indexed UAVs
//! still need read/write tracking independent of conventional table use.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

#[cfg(test)]
mod submission_tests;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
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
    /// Sparse disjoint baseline segments kept before the submission ran: what
    /// the GPU wrote differs without cloning unused interior heap gaps.
    before: Mutex<Option<Snapshot>>,
}

struct Snapshot {
    /// Packed baseline bytes for each disjoint `(from, to)` segment.
    bytes: Vec<u8>,
    /// Disjoint `(from, to, buf_offset)` intervals in shadow-relative offsets.
    segments: Vec<(usize, usize, usize)>,
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

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Bytes of idle snapshot memory kept besides the largest buffer.
const POOL_BYTES: usize = 256 << 20;

/// Returns a snapshot buffer to the pool: the largest is always kept (it is
/// the one whose reuse saves millions of page faults), the others while
/// they add up to `POOL_BYTES` (and at most 8 in all).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    let mut kept = 0usize;
    let mut n = 0usize;
    pool.retain(|v| {
        n += 1;
        if n == 1 {
            return true;
        }
        kept += v.capacity();
        n <= 8 && kept <= POOL_BYTES
    });
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

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

    #[cfg(test)]
    pub(crate) fn read_u32(&self, offset: usize) -> u32 {
        assert!(offset + 4 <= self.len as usize);
        // SAFETY: `offset..offset + 4` lies within the mapped host buffer.
        unsafe { self.data.add(offset).cast::<u32>().read_unaligned() }
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// No-op compatibility hook for tests; [`Shadow::writing_batch`] captures
    /// only disjoint written ranges directly.
    #[cfg(test)]
    pub(crate) fn reserve_snapshot(&self, _address: usize, _len: usize) {}

    /// Declares multiple shader-written `[address, address + len)` ranges of
    /// this shadow in one batch, snapping only the disjoint uncovered parts
    /// once and avoiding copying large interior gaps.
    pub(crate) fn writing_batch(&self, ranges: &[(usize, usize)]) {
        if ranges.is_empty() {
            return;
        }
        let mut fresh = Vec::new();
        {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            let mut declared: Vec<(usize, usize)> =
                pending.iter().map(|&(a, e)| self.inside(a, e - a)).collect();
            for &(address, len) in ranges {
                if len == 0 {
                    continue;
                }
                let (from, to) = self.inside(address, len);
                if from >= to {
                    continue;
                }
                pending.push((address, address + len));
                let parts = outside(from, to, declared.clone());
                for &(a, e) in &parts {
                    declared.push((a, e));
                    fresh.push((a, e));
                }
            }
        }
        let total_fresh: usize = fresh.iter().map(|&(a, e)| e - a).sum();
        if total_fresh == 0 {
            return;
        }
        // SAFETY: every `(a, e)` lies within `[0, self.len)` of the mapping.
        let host =
            |a: usize, e: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), e - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let snap = before.get_or_insert_with(|| Snapshot {
            bytes: pooled(total_fresh),
            segments: Vec::with_capacity(fresh.len()),
        });
        snap.bytes.reserve_exact(total_fresh);
        for (a, e) in fresh {
            let buf_off = snap.bytes.len();
            snap.bytes.extend_from_slice(host(a, e));
            snap.segments.push((a, e, buf_off));
        }
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its disjoint baseline bytes are kept to tell later
    /// what the GPU wrote.
    pub(crate) fn writing(&self, address: usize, len: usize) {
        self.writing_batch(&[(address, len)]);
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    #[cfg(test)]
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_ranges(&[(address, len)])
    }

    /// Coalesced [`Shadow::copy_back`] across multiple `(address, len)` ranges
    /// belonging to the same shadow in one submission, scanning overlapping
    /// bytes at most once and re-arming write-watch only on synced pages.
    pub(crate) fn copy_back_ranges(&self, ranges: &[(usize, usize)]) -> usize {
        let mut clamped: Vec<(usize, usize)> = ranges
            .iter()
            .map(|&(address, len)| self.inside(address, len))
            .filter(|&(from, to)| from < to)
            .collect();
        clamped.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(clamped.len());
        for (from, to) in clamped {
            match merged.last_mut() {
                Some((_, end)) if from <= *end => *end = (*end).max(to),
                _ => merged.push((from, to)),
            }
        }

        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let start = self.start as usize;
        let shadow_len = self.len as usize;
        if let Some(snap) = &mut *snapshot
            && !merged.is_empty()
        {
            const PIECE: usize = 64 << 10;
            snap.segments.sort_unstable_by_key(|&(f, _, _)| f);
            COPY_BACK_SCRATCH.with(|buf| {
                let mut title_buf = buf.borrow_mut();
                if title_buf.len() < PIECE {
                    title_buf.resize(PIECE, 0);
                }
                let mut dirty_runs: Vec<(usize, usize)> = Vec::new();
                let mut synced = Vec::new();
                for &(from, to) in &merged {
                    let mut seg_i = snap.segments.partition_point(|&(_, se, _)| se <= from);
                    while seg_i < snap.segments.len() && snap.segments[seg_i].0 < to {
                        let (seg_from, seg_to, buf_off) = snap.segments[seg_i];
                        let (f, t) = (from.max(seg_from), to.min(seg_to));
                        let mut at = f;
                        while at < t {
                            let n = (PIECE - (start + at) % PIECE).min(t - at);
                            // SAFETY: inside the mapping; only the submitting thread writes it.
                            let host =
                                unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                            let off = buf_off + (at - seg_from);
                            let old = &mut snap.bytes[off..off + n];
                            let title = &mut title_buf[..n];
                            if host != old
                                && crate::host::read_into((start + at) as *const u8, title)
                            {
                                dirty_runs.clear();
                                let mut dirty_start: Option<usize> = None;
                                let mut i = 0usize;
                                while i < n {
                                    if i + 8 <= n && host[i..i + 8] == old[i..i + 8] {
                                        if let Some(s) = dirty_start.take() {
                                            dirty_runs.push((s, i));
                                        }
                                        if host[i..i + 8] != title[i..i + 8] {
                                            host[i..i + 8].copy_from_slice(&title[i..i + 8]);
                                            old[i..i + 8].copy_from_slice(&title[i..i + 8]);
                                        }
                                        i += 8;
                                        continue;
                                    }
                                    let end_j = (i + 8).min(n);
                                    for j in i..end_j {
                                        let h = host[j];
                                        let t_byte = title[j];
                                        if h != old[j] && t_byte != h {
                                            title[j] = h;
                                            dirty_start.get_or_insert(j);
                                            old[j] = h;
                                        } else {
                                            if let Some(s) = dirty_start.take() {
                                                dirty_runs.push((s, j));
                                            }
                                            if h != t_byte {
                                                host[j] = t_byte;
                                            }
                                            if old[j] != t_byte {
                                                old[j] = t_byte;
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
                                    let direct = dirty_runs.len() > 16
                                        && is_writable_commit(chunk_start, n);
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
                                synced.push((start + at) / PAGE);
                            }
                            at += n;
                        }
                        seg_i += 1;
                    }
                }
                // Re-arm write-watch only on contiguous runs of full pages synced above
                // so unrelated untouched subranges avoid multi-MB kernel queries.
                let covers_full_page = |p: usize| -> bool {
                    let Some(pf) = (p * PAGE).checked_sub(start) else {
                        return false;
                    };
                    let pt = pf + PAGE;
                    pt <= shadow_len && merged.iter().any(|&(f, t)| f <= pf && pt <= t)
                };
                synced.retain(|&p| covers_full_page(p));
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
                                // SAFETY: `covers_full_page` guarantees `page * PAGE..page * PAGE + PAGE`
                                // lies within `[start, start + shadow_len)`.
                                let host_page = unsafe {
                                    std::slice::from_raw_parts(
                                        self.data.add(page * PAGE - start),
                                        PAGE,
                                    )
                                };
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
        }
        #[cfg(feature = "diagnostics")]
        if written > 0 && log_copyback() {
            let (first_addr, first_len) = ranges.first().copied().unwrap_or((0, 0));
            tracing::info!(
                shadow_start = format_args!("{:#x}", self.start),
                address = format_args!("{first_addr:#x}"),
                len = first_len,
                ranges = ranges.len(),
                written,
                "shadow copy_back"
            );
        }
        // Every submitted step that declared these ranges is done with them,
        // and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        for &(address, len) in ranges {
            pending.retain(|&r| r != (address, address + len));
        }
        if pending.is_empty()
            && let Some(snap) = snapshot.take()
        {
            give_back(snap.bytes);
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
    pub(crate) fn update_with_watch(
        &self,
        address: usize,
        len: usize,
        prefetched: Option<(u64, &[(usize, u64)])>,
    ) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        if let Some((now, written)) = prefetched
            && self.can_use_prefetched(from, to)
        {
            self.watched.store(1, Ordering::Relaxed);
            let copied = self.copy_stale(from, to, now, written);
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::Checked, (to - from) as u64);
                add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
            }
            return copied;
        }
        self.update(from, to - from)
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
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
                self.update_gaps(&gaps)
            };
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of disjoint non-busy `gaps` (`from < to`, in ascending
    /// order) using a single write-watch query across the bounding span.
    fn update_gaps(&self, gaps: &[(usize, usize)]) -> usize {
        let span: usize = gaps.iter().map(|&(f, t)| t - f).sum();
        let (g_from, g_to) = (gaps[0].0, gaps[gaps.len() - 1].1);
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(g_from, g_to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let mut copied = 0usize;
                for &(f, t) in gaps {
                    copied += self.copy_stale(f, t, now, &written);
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
        for &(f, t) in gaps {
            self.copy(f, t);
        }
        span
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
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
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
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
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

/// The parts of `[from, to)` outside every range in `ranges`, in order.
fn outside(from: usize, to: usize, mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let (mut at, mut parts) = (from, Vec::new());
    for (a, e) in ranges {
        if at < to && a > at {
            parts.push((at, a.min(to)));
        }
        at = at.max(e);
    }
    if at < to {
        parts.push((at, to));
    }
    parts
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
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
            shadow: None,
        },
    );
}

/// The title released it; any live host copy is retired behind the queue.
pub(crate) fn unregister(resource: usize, start: u64) {
    if let Some(entry) = buffers().remove(&(start, resource))
        && let Some(shadow) = entry.shadow
    {
        crate::host::retire(shadow);
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
    let mut map = buffers();
    let key = enclosing(&map, MAX_SPAN.load(Ordering::Relaxed), address, end);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
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
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
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
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    LOOSE_SPAN.fetch_max(len, Ordering::Relaxed);
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
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

    fn title_pages(pages: usize) -> Option<(Vec<u8>, usize, Shadow)> {
        let memory = vec![0u8; (pages + 1) * PAGE];
        let base = (memory.as_ptr() as usize).next_multiple_of(PAGE);
        let (buffer, data) = host()?.shared_buffer((pages * PAGE) as u64)?;
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Shadow::new(buffer, data, gpu, base as u64, (pages * PAGE) as u64);
        Some((memory, base, shadow))
    }

    /// Three pages of title memory on page bounds (plain heap memory, so
    /// not write-watched: refreshes copy whole ranges) and their host
    /// copy; `None` without a host GPU. Returns the memory's owner, the
    /// address of the first page and the copy.
    fn title() -> Option<(Vec<u8>, usize, Shadow)> {
        title_pages(3)
    }

    /// A write of one byte: the title's into its memory, or a shader's
    /// into the copy.
    fn poke(at: usize, value: u8) {
        // SAFETY: the tests pass addresses inside the memory or the copy.
        unsafe { (at as *mut u8).write_volatile(value) }
    }

    fn peek(at: usize) -> u8 {
        // SAFETY: as in `poke`.
        unsafe { (at as *const u8).read_volatile() }
    }

    /// A range declared after ranges on both sides of it, once the snapshot
    /// already spans it: what its refresh brought into the copy is not a
    /// shader's write, and does not go back over what the title wrote since.
    #[test]
    fn a_range_between_earlier_ones_keeps_the_titles_newer_bytes() {
        let Some((_memory, base, s)) = title() else {
            return; // no host GPU here
        };
        let (a, b, c) = (base, base + PAGE, base + 2 * PAGE);
        let copy = |at: usize| s.data as usize + (at - base);
        // An earlier submission read the middle page; the title has written
        // it since.
        s.update(b, PAGE);
        poke(b + 5, 1);
        poke(b + 6, 3);
        // This submission's shaders write the outer pages, then the middle
        // one: each range refreshed, then declared.
        for at in [a, c, b] {
            s.update(at, PAGE);
            s.writing(at, PAGE);
        }
        assert_eq!(peek(copy(b + 5)), 1, "the refresh");
        // While the GPU runs: a shader writes two bytes, the title one.
        poke(copy(a + 7), 9);
        poke(copy(b + 9), 7);
        poke(b + 5, 2);
        for at in [a, c, b] {
            s.copy_back(at, PAGE);
        }
        assert_eq!(peek(b + 5), 2, "the title's newer byte");
        assert_eq!(peek(b + 6), 3, "the title's byte from before");
        // What the shader wrote does go back, in either kind of range.
        assert_eq!((peek(a + 7), peek(b + 9)), (9, 7));
        assert!(s.before.lock().unwrap().is_none(), "the snapshot is done");
    }

    /// Two declared ranges that overlap: the first one's copy-back gives
    /// the copy the title's bytes, which the second must not take for a
    /// shader's and put back over what the title wrote in between.
    #[test]
    fn overlapping_ranges_keep_the_titles_newer_bytes() {
        let Some((_memory, base, s)) = title() else {
            return; // no host GPU here
        };
        let copy = |at: usize| s.data as usize + (at - base);
        s.update(base, 2 * PAGE);
        s.writing(base, PAGE);
        s.writing(base, 2 * PAGE);
        // While the GPU runs: a shader's byte and the title's, in one page.
        poke(copy(base + 3), 9);
        poke(base + 8, 1);
        s.copy_back(base, PAGE);
        assert_eq!((peek(base + 3), peek(base + 8)), (9, 1));
        // The title writes its byte again before the second range's turn.
        poke(base + 8, 2);
        s.copy_back(base, 2 * PAGE);
        assert_eq!(peek(base + 8), 2, "the title's newer byte");
        assert_eq!(peek(base + 3), 9, "the shader's byte");
    }

    const MIB: u64 = 1 << 20;

    fn entry(start: u64, len: u64) -> Entry {
        Entry {
            end: start + len,
            shadow: None,
        }
    }

    /// A 256 MiB parent with `children` 64 KiB buffers placed in its first
    /// part, as a heap's placed buffers.
    fn heap(base: u64, children: u64) -> BTreeMap<(u64, usize), Entry> {
        let mut map = BTreeMap::new();
        map.insert((base, 1), entry(base, 256 * MIB));
        for i in 0..children {
            map.insert((base + i * 0x10000, 100 + i as usize), entry(base + i * 0x10000, 0x10000));
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
        let first_64 = map.range(..=(at, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= at + 16);
        assert!(first_64.is_none(), "the parent is more than 64 entries back");
        // Near the parent's end.
        let at = base + 256 * MIB - 16;
        assert_eq!(enclosing(&map, span, at, at + 16), Some((base, 1)));
        // Inside a child: the child, the innermost.
        let at = base + 150 * 0x10000 + 8;
        assert_eq!(enclosing(&map, span, at, at + 8), Some((base + 150 * 0x10000, 250)));
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
        assert!(buffers().range((base, 0)..=(base + len, usize::MAX)).next().is_none());
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
        let find = |address: u64, len: u64| enclosing_window(&windows, span, address, address + len).copied();
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
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        // The null page and its guard area.
        assert!(loose(0, 16).is_none());
        assert!(loose(0x1000, 0x1010).is_none());
        assert!(loose(LOWEST_MAPPED - 1, LOWEST_MAPPED + 1).is_none());
        // Reserved, not committed.
        // SAFETY: a fresh reservation, released below.
        let p = unsafe { VirtualAlloc(std::ptr::null(), WINDOW as usize, MEM_RESERVE, PAGE_READWRITE) } as u64;
        assert_ne!(p, 0);
        assert!(loose(p + 64, p + 128).is_none());
        // SAFETY: our reservation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        // Too close to the top to round up to a window.
        assert!(loose(u64::MAX - 8, u64::MAX).is_none());
    }

    #[test]
    fn copy_back_only_writes_gpu_modified_bytes() {
        let Some((_memory, base, s)) = title() else {
            return; // no host GPU here
        };
        let copy = |at: usize| s.data as usize + (at - base);
        for off in 0..PAGE {
            poke(base + off, 0x11);
        }
        s.update(base, PAGE);
        s.writing(base, PAGE);
        // Simulate GPU writing only two disjoint 4-byte runs inside the page.
        for off in 16..20 {
            poke(copy(base + off), 0xaa);
        }
        for off in 1024..1028 {
            poke(copy(base + off), 0xbb);
        }
        // Protect unrelated prefix/middle/suffix bytes outside the GPU runs as
        // an indirect check via title mutation right before copy_back, and
        // assert exact byte count returned (`written == 8`).
        poke(base + 500, 0x77);
        let written = s.copy_back(base, PAGE);
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
    fn reserve_snapshot_before_reverse_writing_keeps_bytes_intact() {
        let Some((_memory, base, s)) = title() else {
            return;
        };
        let (a, b, c) = (base, base + PAGE, base + 2 * PAGE);
        let copy = |at: usize| s.data as usize + (at - base);
        poke(b + 12, 0x33);
        poke(a + 4, 0x44);
        poke(c + 8, 0x55);
        s.update(base, 3 * PAGE);
        s.reserve_snapshot(base, 3 * PAGE);
        // Reverse order with an interior gap: snapshot must stay consistent.
        s.writing(c, PAGE);
        s.writing(a, PAGE);
        poke(copy(c + 20), 0x99);
        poke(copy(a + 24), 0x88);
        poke(b + 12, 0x77);
        assert_eq!(s.copy_back(c, PAGE), 1);
        assert_eq!(s.copy_back(a, PAGE), 1);
        assert_eq!(peek(c + 20), 0x99);
        assert_eq!(peek(a + 24), 0x88);
        assert_eq!(peek(b + 12), 0x77);
        assert!(s.before.lock().unwrap().is_none());
    }

    #[test]
    fn update_with_multiple_busy_gaps_preserves_busy_ranges() {
        let Some((_memory, base, s)) = title() else {
            return;
        };
        let (p0, p1, p2) = (base, base + PAGE, base + 2 * PAGE);
        let copy = |at: usize| s.data as usize + (at - base);
        poke(p0 + 1, 0x10);
        poke(p1 + 1, 0x20);
        poke(p2 + 1, 0x30);
        s.update(base, 3 * PAGE);
        // Hold middle page `p1` busy on the GPU while outer pages change.
        s.writing(p1, PAGE);
        poke(p0 + 1, 0x11);
        poke(p1 + 1, 0x22);
        poke(p2 + 1, 0x33);
        poke(copy(p1 + 2), 0xaa);
        s.update(base, 3 * PAGE);
        assert_eq!(peek(copy(p0 + 1)), 0x11);
        assert_eq!(peek(copy(p1 + 1)), 0x20, "busy middle page not overwritten");
        assert_eq!(peek(copy(p2 + 1)), 0x33);
        assert_eq!(s.copy_back(p1, PAGE), 1);
        assert_eq!(peek(p1 + 2), 0xaa);
        assert_eq!(peek(p1 + 1), 0x22);
    }

    #[test]
    fn copy_back_ranges_coalesces_overlaps_and_preserves_concurrent_cpu_writes() {
        let Some((_memory, base, s)) = title() else {
            return;
        };
        let copy = |at: usize| s.data as usize + (at - base);
        for off in 0..2 * PAGE {
            poke(base + off, 0x10);
        }
        s.update(base, 2 * PAGE);
        s.writing(base, PAGE + 1024);
        s.writing(base + 512, 2 * PAGE - 512);
        poke(copy(base + 20), 0xde);
        poke(copy(base + 600), 0xad);
        poke(copy(base + PAGE + 64), 0xbe);
        poke(base + 700, 0x55);

        let written = s.copy_back_ranges(&[(base, PAGE + 1024), (base + 512, 2 * PAGE - 512)]);
        assert_eq!(written, 3);
        assert_eq!(peek(base + 20), 0xde);
        assert_eq!(peek(base + 600), 0xad);
        assert_eq!(peek(base + PAGE + 64), 0xbe);
        assert_eq!(peek(base + 700), 0x55);
        assert!(s.pending.lock().unwrap().is_empty());
        assert!(s.before.lock().unwrap().is_none());
    }

    #[test]
    fn copy_back_only_watches_synced_pages() {
        let Some((_memory, base, s)) = title() else {
            return;
        };
        let copy = |at: usize| s.data as usize + (at - base);
        for p in 0..3 {
            poke(base + p * PAGE + 4, 0x11 + p as u8);
        }
        s.update(base, 3 * PAGE);
        s.writing(base, 3 * PAGE);
        // GPU modifies only page 1 while CPU modifies page 0 and page 2 concurrently.
        poke(copy(base + PAGE + 8), 0xef);
        poke(base + 12, 0x77);
        poke(base + 2 * PAGE + 12, 0x99);
        assert_eq!(s.copy_back(base, 3 * PAGE), 1);
        assert_eq!(peek(base + PAGE + 8), 0xef);
        // Updating the full shadow must still pick up CPU writes on page 0 and page 2.
        s.update(base, 3 * PAGE);
        assert_eq!(peek(copy(base + 12)), 0x77);
        assert_eq!(peek(copy(base + 2 * PAGE + 12)), 0x99);
    }

    #[test]
    fn sparse_snapshot_skips_interior_gaps_and_preserves_gap_writes() {
        let Some((_memory, base, s)) = title_pages(10) else {
            return;
        };
        let copy = |at: usize| s.data as usize + (at - base);
        let last_off = 9 * PAGE;
        poke(base + 10, 0x11);
        poke(base + 4 * PAGE + 20, 0x22);
        poke(base + last_off + 30, 0x33);
        s.update(base, 10 * PAGE);

        s.writing_batch(&[(base, PAGE), (base + last_off, PAGE)]);
        {
            let before = s.before.lock().unwrap();
            let snap = before.as_ref().expect("snapshot created");
            assert_eq!(snap.bytes.len(), 2 * PAGE);
        }

        // GPU writes both endpoints; CPU concurrently writes inside the interior gap.
        poke(copy(base + 10), 0xaa);
        poke(copy(base + last_off + 30), 0xbb);
        poke(base + 4 * PAGE + 20, 0x44);

        let written = s.copy_back_ranges(&[(base, PAGE), (base + last_off, PAGE)]);
        assert_eq!(written, 2);
        assert_eq!(peek(base + 10), 0xaa);
        assert_eq!(peek(base + last_off + 30), 0xbb);
        assert_eq!(peek(base + 4 * PAGE + 20), 0x44);
        assert!(s.before.lock().unwrap().is_none());
    }

    #[test]
    fn copy_back_strided_writes_preserve_interleaved_cpu_bytes() {
        let Some((_memory, base, s)) = title() else {
            return;
        };
        let copy = |at: usize| s.data as usize + (at - base);
        for off in 0..PAGE {
            poke(base + off, 0x00);
        }
        s.update(base, PAGE);
        s.writing(base, PAGE);

        // 1,000 interleaved 4-byte GPU and 4-byte CPU writes inside one 64 KiB
        // page: triggers the `dirty_runs.len() > 16` direct strided copy-back.
        for k in 0..1000usize {
            let gpu_off = k * 8;
            let cpu_off = gpu_off + 4;
            for b in 0..4 {
                poke(copy(base + gpu_off + b), 0x80 | (((k + b) as u8) & 0x7f));
                poke(base + cpu_off + b, 0x40 | ((k as u8) & 0x3f));
            }
        }

        let written = s.copy_back(base, PAGE);
        assert_eq!(written, 1000 * 4);
        for k in 0..1000usize {
            let gpu_off = k * 8;
            let cpu_off = gpu_off + 4;
            for b in 0..4 {
                assert_eq!(peek(base + gpu_off + b), 0x80 | (((k + b) as u8) & 0x7f));
                assert_eq!(peek(base + cpu_off + b), 0x40 | ((k as u8) & 0x3f));
                assert_eq!(peek(copy(base + cpu_off + b)), 0x40 | ((k as u8) & 0x3f));
            }
        }
        assert_eq!(peek(base + 1000 * 8), 0x00);
        assert!(s.before.lock().unwrap().is_none());
    }

    #[test]
    fn copy_back_keeps_page_stale_on_concurrent_title_write() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        let Some((buffer, data)) = host().and_then(|h| h.shared_buffer(PAGE as u64)) else {
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
        let s = Shadow::new(buffer, data, gpu, base as u64, PAGE as u64);
        let copy = |at: usize| s.data as usize + (at - base);

        poke(base + 16, 0x11);
        s.update(base, PAGE);
        let before_copy_back = s.copied.lock().unwrap()[0];
        assert_ne!(before_copy_back, 0);

        s.writing(base, PAGE);
        poke(copy(base + 32), 0xaa);

        // Simulate a title CPU write landing after copy-back scanned the page
        // but before `watch::writes` clears the kernel write-watch bit.
        BEFORE_COPY_BACK_REARM.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move || {
                poke(base + 64, 0x77);
            }));
        });
        let written = s.copy_back(base, PAGE);
        BEFORE_COPY_BACK_REARM.with(|hook| {
            *hook.borrow_mut() = None;
        });

        assert_eq!(written, 1);
        assert_eq!(peek(base + 32), 0xaa);
        assert_eq!(peek(base + 64), 0x77);
        assert_eq!(peek(copy(base + 64)), 0);
        assert_eq!(
            s.copied.lock().unwrap()[0],
            before_copy_back,
            "page watermark stays stale when title memory changed before re-arm"
        );

        // Next update must still pick up the concurrent title write.
        s.update(base, PAGE);
        assert_eq!(peek(copy(base + 64)), 0x77);

        crate::watch::forget_journal(base, PAGE);
        // SAFETY: our write-watched allocation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }
}

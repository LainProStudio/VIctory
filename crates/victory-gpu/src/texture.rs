//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging footprint layout as the host texture's
/// subresources, and what was last detiled (written into the host's pooled
/// submission upload ring only when dirty).
pub(crate) struct Staging {
    total: u64,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// Original packed 16-bit format (86 or 115) when the host texture was
    /// widened to R8G8B8A8_UNORM, so detiled 2-byte texels expand into the
    /// 4-byte staging buffer.
    widened_from: Option<u32>,
    /// Lock-free mirror of `Last::asked` so clean watched textures skip locking
    /// `last` altogether when using pre-fetched batch write-watch results.
    last_asked: std::sync::atomic::AtomicU64,
    last: Mutex<Last>,
    #[cfg(test)]
    gpu_uploads: std::sync::atomic::AtomicU64,
}

/// What a texture's staging state holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
}

/// Scratch `(read, linear)` buffer pairs kept for reuse across worker threads
/// (already touched, so refreshing does not fault their pages in again).
static SCRATCH_POOL: Mutex<Vec<(Vec<u8>, Vec<u8>)>> = Mutex::new(Vec::new());

const MAX_SCRATCH: usize = 64 << 20;
const MAX_POOLED_SCRATCH: usize = 8;
const MAX_POOLED_SCRATCH_BYTES: usize = 256 << 20;

fn shrink_scratch(buf: &mut Vec<u8>) {
    if buf.capacity() > MAX_SCRATCH {
        buf.clear();
        buf.shrink_to(MAX_SCRATCH);
    }
}

fn take_scratch() -> (Vec<u8>, Vec<u8>) {
    SCRATCH_POOL
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .pop()
        .unwrap_or_default()
}

fn return_scratch(mut read: Vec<u8>, mut linear: Vec<u8>) {
    shrink_scratch(&mut read);
    shrink_scratch(&mut linear);
    let mut pool = SCRATCH_POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push((read, linear));
    pool.sort_by_key(|(r, l)| std::cmp::Reverse(r.capacity() + l.capacity()));
    let mut kept = 0usize;
    let mut n = 0usize;
    pool.retain(|(r, l)| {
        n += 1;
        kept += r.capacity() + l.capacity();
        n <= MAX_POOLED_SCRATCH && (n == 1 || kept <= MAX_POOLED_SCRATCH_BYTES)
    });
}

/// Submission-lifetime upload buffer ring shared across all dirty textures in a
/// frame, resetting right after the synchronous submission finishes on the GPU.
pub(crate) struct UploadPool {
    chunks: Vec<(ID3D12Resource, usize, u64)>,
    current: usize,
    used: u64,
    oversized: Vec<ID3D12Resource>,
}

// SAFETY: only touched under the host submission lock while sequential submissions prepare.
unsafe impl Send for UploadPool {}

impl UploadPool {
    const CHUNK_BYTES: u64 = 32 << 20;
    const MAX_POOLED_BYTES: u64 = 256 << 20;

    pub(crate) const fn new() -> Self {
        Self {
            chunks: Vec::new(),
            current: 0,
            used: 0,
            oversized: Vec::new(),
        }
    }

    pub(crate) fn alloc(
        &mut self,
        host: &crate::host::Host,
        bytes: u64,
    ) -> Option<(ID3D12Resource, *mut u8, u64)> {
        const ALIGN: u64 = 512;
        let bytes = bytes.max(1);
        if bytes > Self::CHUNK_BYTES {
            let (buffer, data) = host.upload_buffer(bytes)?;
            crate::host::name(&buffer, || format!("texture upload oversized {bytes:#x}"));
            self.oversized.push(buffer.clone());
            return Some((buffer, data, 0));
        }
        let aligned = self.used.next_multiple_of(ALIGN);
        if self.current < self.chunks.len()
            && aligned.saturating_add(bytes) > self.chunks[self.current].2
        {
            self.current += 1;
            self.used = 0;
        } else if self.current < self.chunks.len() {
            self.used = aligned;
        }
        if self.current >= self.chunks.len() {
            let (buffer, data) = host.upload_buffer(Self::CHUNK_BYTES)?;
            crate::host::name(&buffer, || "texture upload chunk".into());
            self.chunks.push((buffer, data as usize, Self::CHUNK_BYTES));
            self.used = 0;
        }
        let (ref buffer, ptr, _) = self.chunks[self.current];
        let offset = self.used;
        self.used += bytes;
        // SAFETY: `offset..offset + bytes` falls inside the chunk's `CHUNK_BYTES` mapping.
        let dst = unsafe { (ptr as *mut u8).add(offset as usize) };
        Some((buffer.clone(), dst, offset))
    }

    pub(crate) fn reset(&mut self) {
        self.oversized.clear();
        let mut kept = 0u64;
        self.chunks.retain(|(_, _, cap)| {
            if kept + *cap <= Self::MAX_POOLED_BYTES {
                kept += *cap;
                true
            } else {
                false
            }
        });
        self.current = 0;
        self.used = 0;
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> (usize, usize) {
        (self.chunks.len(), self.oversized.len())
    }
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    set.get_or_insert_with(Default::default)
        .insert(resource as usize);
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
}

/// Whether `resource` was previously marked as host-written (copy/resolve target).
pub(crate) fn is_host_written(resource: usize) -> bool {
    HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&resource))
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Registers a title-written texture with `list` (once per list) so that at
/// submission time its staging buffer is checked for CPU modifications and,
/// only when its contents changed (including first upload), copied into its
/// host texture on the list's prelude command list.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if is_host_written(resource as usize) || !list.first_use(resource as usize) {
        return;
    }
    let Some(copy) = crate::resource::host_copy(resource) else {
        return;
    };
    let Some(staging) = staging_for(resource, texture) else {
        return;
    };
    // Once per submission however many of its lists sample the texture;
    // `copy` keeps the host texture alive even if the title releases it
    // before submission, and `CopyTextureRegion` is only recorded onto the
    // prelude when `staging.refresh()` reports dirty contents.
    list.defer_texture(staging, copy);
}

pub(crate) fn hash_textures() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_HASH_TEXTURES"))
}

impl Staging {
    /// Title memory span `(start, end)` backing this texture.
    pub(crate) fn memory_range(&self) -> (usize, usize) {
        let start = self.memory as usize;
        (start, start.saturating_add(self.layout.size as usize))
    }

    /// Detiles the texture memory into `pool` only when it changed since the last
    /// time (and is mapped at all). Returns `Some((upload_buffer, base_offset))`
    /// when a copy to the host texture is required.
    #[cfg(test)]
    pub(crate) fn refresh(
        &self,
        host: &crate::host::Host,
        pool: &mut UploadPool,
    ) -> Option<(ID3D12Resource, u64)> {
        let watch_now = match self.check_watch_clean(None) {
            Some(None) => return None,
            Some(Some(now)) => Some(now),
            None => None,
        };
        let (mut read, mut linear) = take_scratch();
        let mut packed = Vec::new();
        let pool_mutex = Mutex::new(pool);
        let res = self.refresh_prepared(
            host,
            &pool_mutex,
            watch_now,
            &mut read,
            &mut linear,
            &mut packed,
        );
        return_scratch(read, linear);
        res
    }

    /// Checks write-watch status on the calling (preparing) thread so `Sweep`
    /// applies. Returns:
    /// - `Some(None)` if watched and clean (skip refresh completely);
    /// - `Some(Some(now))` if watched and dirty (needs read + detile at epoch `now`);
    /// - `None` if not watched or `hash_textures()` is enabled (needs read + blake3 hash).
    pub(crate) fn check_watch_clean(
        &self,
        prefetched: Option<(u64, &[(usize, u64)])>,
    ) -> Option<Option<u64>> {
        use std::sync::atomic::Ordering;
        if hash_textures() {
            return None;
        }
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let watched_answer = if let Some((now, all_written)) = prefetched {
            let first = start / crate::watch::PAGE;
            let last_page = (start + size).div_ceil(crate::watch::PAGE);
            let lo = all_written.partition_point(|&(p, _)| p < first);
            let hi = all_written.partition_point(|&(p, _)| p < last_page);
            Some((now, std::borrow::Cow::Borrowed(&all_written[lo..hi])))
        } else {
            crate::watch::writes(start, start + size, true)
                .map(|(now, v)| (now, std::borrow::Cow::Owned(v)))
        };
        let (now, written) = watched_answer?;
        // Lock-free clean fast path: if our last detiled watch epoch already
        // covers every write in this texture's pages, skip the mutex.
        let asked = self.last_asked.load(Ordering::Acquire);
        if asked != 0 && written.iter().all(|&(_, w)| w <= asked) {
            return Some(None);
        }
        let last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
            self.last_asked.store(last.asked, Ordering::Release);
            return Some(None);
        }
        Some(Some(now))
    }

    /// Refreshes a single dirty or hashed texture into `pool` using caller-owned
    /// scratch buffers (safe to run concurrently across distinct textures).
    pub(crate) fn refresh_prepared(
        &self,
        host: &crate::host::Host,
        pool: &Mutex<&mut UploadPool>,
        watch_now: Option<u64>,
        read: &mut Vec<u8>,
        linear: &mut Vec<u8>,
        packed: &mut Vec<u8>,
    ) -> Option<(ID3D12Resource, u64)> {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (bytes_read, uploaded) =
            self.refresh_prepared_inner(host, pool, watch_now, read, linear, packed);
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(bytes_read as u64, Relaxed);
        uploaded
    }

    fn refresh_prepared_inner(
        &self,
        host: &crate::host::Host,
        pool: &Mutex<&mut UploadPool>,
        watch_now: Option<u64>,
        read: &mut Vec<u8>,
        linear: &mut Vec<u8>,
        packed: &mut Vec<u8>,
    ) -> (usize, Option<(ID3D12Resource, u64)>) {
        use std::sync::atomic::Ordering;
        let size = self.layout.size as usize;
        if let Some(now) = watch_now {
            let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(extra) = size.checked_sub(read.len())
                && read.try_reserve_exact(extra).is_err()
            {
                once!(bytes = size, "no room to read a texture's memory; not uploaded");
                return (0, None);
            }
            let Some((upload_buf, upload_ptr, base_offset)) =
                pool.lock().unwrap_or_else(|e| e.into_inner()).alloc(host, self.total)
            else {
                return (0, None);
            };
            last.asked = now;
            self.last_asked.store(now, Ordering::Release);
            read.resize(size, 0);
            crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
            self.detile(read, linear, packed, upload_ptr);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return (size, Some((upload_buf, base_offset)));
        }
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some(extra) = size.checked_sub(read.len())
            && read.try_reserve_exact(extra).is_err()
        {
            once!(bytes = size, "no room to read a texture's memory; not uploaded");
            return (0, None);
        }
        read.resize(size, 0);
        if !crate::host::read_into(self.memory, read) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return (0, None);
        }
        let hash = blake3::hash(read);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return (size, None);
        }
        let Some((upload_buf, upload_ptr, base_offset)) =
            pool.lock().unwrap_or_else(|e| e.into_inner()).alloc(host, self.total)
        else {
            return (size, None);
        };
        last.hash = Some(hash);
        self.detile(read, linear, packed, upload_ptr);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, Some((upload_buf, base_offset)))
    }

    /// Records `CopyTextureRegion` for every subresource from `upload` at
    /// `base_offset` into `texture` (which must already be in `COPY_DEST`).
    pub(crate) fn record_copies(
        &self,
        list: &ID3D12GraphicsCommandList,
        texture: &ID3D12Resource,
        upload: &ID3D12Resource,
        base_offset: u64,
    ) {
        for (i, footprint) in self.footprints.iter().enumerate() {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: i as u32,
                },
            };
            let mut placed = *footprint;
            placed.Offset += base_offset;
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(upload) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: placed,
                },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
            #[cfg(test)]
            self.gpu_uploads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    fn gpu_uploads(&self) -> u64 {
        self.gpu_uploads.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Detiles `tiled` (the texture's memory) into `dst` (at `footprint.Offset`
    /// inside an upload pool slice). The buffer is a mapped upload heap
    /// (write-combined), where scattered writes are slow, so each subresource is
    /// detiled into `linear` and copied in one sequential pass.
    fn detile(&self, tiled: &[u8], linear: &mut Vec<u8>, packed: &mut Vec<u8>, dst: *mut u8) {
        #[cfg(feature = "diagnostics")]
        if tracing::enabled!(tracing::Level::DEBUG) && tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let lines = rows as usize * f.Depth as usize;
            let len = if self.widened_from.is_some() && lines > 0 {
                // The last subresource allocation stops at the last row's used bytes.
                pitch * (lines - 1) + f.Width as usize * 4
            } else {
                pitch * lines
            };
            if linear.len() < len {
                linear.resize(len, 0);
            }
            let scratch = &mut linear[..len];
            if let Err(e) = detile_subresource(
                &self.layout,
                tiled,
                mip,
                slice,
                scratch,
                pitch,
                rows as usize,
                f.Depth as usize,
                self.widened_from,
                packed,
            ) {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
            // SAFETY: `dst + footprint.Offset..dst + footprint.Offset + len`
            // lies inside the upload pool slice allocated with `self.total` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    scratch.as_ptr(),
                    dst.add(footprint.Offset as usize),
                    len,
                )
            };
        }
    }
}

struct SendUpload(usize, ID3D12Resource, u64);
// SAFETY: `ID3D12Resource` is free-threaded; moved once back to the preparing thread.
unsafe impl Send for SendUpload {}

/// Reads, hashes (when unwatched), and detiles all `items` in parallel using a
/// bounded worker pool (`available_parallelism.min(8)`). Each item must refer
/// to a distinct `Staging` instance.
pub(crate) fn refresh_batch(
    host: &crate::host::Host,
    pool: &mut UploadPool,
    items: Vec<(usize, Arc<Staging>, Option<u64>)>,
) -> Vec<(usize, ID3D12Resource, u64)> {
    if items.is_empty() {
        return Vec::new();
    }
    if items.len() == 1 {
        let (idx, staging, watch_now) = &items[0];
        let (mut read, mut linear) = take_scratch();
        let mut packed = Vec::new();
        let pool_mutex = Mutex::new(pool);
        let res = staging
            .refresh_prepared(
                host,
                &pool_mutex,
                *watch_now,
                &mut read,
                &mut linear,
                &mut packed,
            )
            .map(|(buf, off)| (*idx, buf, off))
            .into_iter()
            .collect();
        return_scratch(read, linear);
        return res;
    }
    let workers = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(8)
        .min(items.len())
        .max(1);
    let pool_mutex = Mutex::new(pool);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<Vec<SendUpload>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                s.spawn(|| {
                    let (mut read, mut linear) = take_scratch();
                    let mut packed = Vec::new();
                    let mut local = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= items.len() {
                            break;
                        }
                        let (idx, ref staging, watch_now) = items[i];
                        if let Some((buf, off)) = staging.refresh_prepared(
                            host,
                            &pool_mutex,
                            watch_now,
                            &mut read,
                            &mut linear,
                            &mut packed,
                        ) {
                            local.push(SendUpload(idx, buf, off));
                        }
                    }
                    return_scratch(read, linear);
                    local
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_default())
            .collect()
    });
    let mut out: Vec<(usize, ID3D12Resource, u64)> = results
        .drain(..)
        .flatten()
        .map(|SendUpload(idx, buf, off)| (idx, buf, off))
        .collect();
    out.sort_unstable_by_key(|&(idx, _, _)| idx);
    out
}

#[allow(clippy::too_many_arguments)]
fn detile_subresource(
    layout: &victory_formats::texture::Layout,
    tiled: &[u8],
    mip: u32,
    slice: u32,
    linear: &mut [u8],
    row_pitch: usize,
    rows: usize,
    depth: usize,
    widened_from: Option<u32>,
    packed: &mut Vec<u8>,
) -> Result<(), victory_formats::texture::Error> {
    let Some(packed_format) = widened_from else {
        return layout.detile(tiled, mip, slice, linear, row_pitch, row_pitch * rows);
    };
    let width = layout
        .mips
        .get(mip as usize)
        .map_or(0, |m| m.width as usize);
    let packed_pitch = width * 2;
    let packed_plane = packed_pitch * rows;
    packed.resize(packed_plane * depth, 0);
    layout.detile(tiled, mip, slice, packed, packed_pitch, packed_plane)?;
    crate::resource::expand_packed16_rows(
        packed_format,
        width,
        rows * depth,
        packed,
        packed_pitch,
        linear,
        row_pitch,
    );
    Ok(())
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// The staging state of a title texture, made on first use without holding
/// `STAGING` across layout computation or footprint queries.
pub(crate) fn staging_for(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = map.as_ref().and_then(|m| m.get(&(resource as usize))) {
            return Some(s.clone());
        }
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    let s = create_staging(host, resource, texture, layout, memory)?;
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    let entry = map.entry(resource as usize).or_insert_with(|| {
        tracing::debug!(?desc, "texture contents queued for staging upload");
        s
    });
    Some(entry.clone())
}

fn create_staging(
    host: &crate::host::Host,
    resource: This,
    texture: &ID3D12Resource,
    layout: victory_formats::texture::Layout,
    memory: *const u8,
) -> Option<Arc<Staging>> {
    // SAFETY: a live resource.
    let mut d = unsafe { texture.GetDesc() };
    let widened_from = (d.Format.0 as u32 == crate::resource::RGBA8_UNORM)
        .then(|| crate::resource::texture_memory(resource))
        .flatten()
        .map(|(_, desc)| crate::resource::pc_format(desc.format))
        .filter(|title| crate::resource::PACKED_16.contains(title));
    // Use the title's concrete format when the host texture is typeless so
    // `CopyTextureRegion` footprints never carry a `TYPELESS` format.
    if widened_from.is_none()
        && let Some(concrete) = crate::resource::view_format(resource)
    {
        d.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(concrete as i32);
    }
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    Some(Arc::new(Staging {
        total,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        widened_from,
        last_asked: std::sync::atomic::AtomicU64::new(0),
        last: Mutex::default(),
        #[cfg(test)]
        gpu_uploads: std::sync::atomic::AtomicU64::new(0),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::Host;
    use crate::resource::HostTexture;
    use std::sync::atomic::{AtomicU32, Ordering};
    use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

    const SHADER_RESOURCE: u32 = (D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0
        | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0) as u32;

    fn make_rgba_texture(host: &Host, width: u32, height: u32) -> (Arc<HostTexture>, Texture) {
        let layout_tex = Texture::new(&TextureDesc {
            dimension: Dimension::Tex2d,
            format: DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            width,
            height,
            depth_or_array: 1,
            mips: 1,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        })
        .expect("linear RGBA layout");
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: plain default-heap 2D texture creation on our test device.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut resource,
                )
                .expect("create host texture");
        }
        let copy = Arc::new(HostTexture {
            texture: resource.unwrap(),
            state: AtomicU32::new(D3D12_RESOURCE_STATE_COMMON.0 as u32),
            twin: None,
        });
        (copy, layout_tex)
    }

    /// Reads a 2D RGBA8 texture back from `host` into a tightly packed row buffer.
    fn read_back_rgba(host: &Host, copy: &Arc<HostTexture>, width: u32, height: u32) -> Vec<u8> {
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut rows = 0u32;
        let mut total = 0u64;
        // SAFETY: querying footprint of a live texture.
        let desc = unsafe { copy.texture.GetDesc() };
        // SAFETY: single-subresource footprint query.
        unsafe {
            host.device.GetCopyableFootprints(
                &desc,
                0,
                1,
                0,
                Some(&mut footprint),
                Some(&mut rows),
                None,
                Some(&mut total),
            );
        }
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain readback buffer creation.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .expect("create readback buffer");
        }
        let readback = readback.unwrap();
        let list = host.new_list().expect("readback list");
        let before = copy.state.load(Ordering::Relaxed);
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer for the copy call.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer for the copy call.
            pResource: unsafe { std::mem::transmute_copy(&copy.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: recording a valid readback copy on an open list.
        unsafe {
            if before != D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32 {
                list.list.ResourceBarrier(&[crate::host::transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATES(before as i32),
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
            }
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            if before != D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32 {
                list.list.ResourceBarrier(&[crate::host::transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATES(before as i32),
                )]);
            }
        }
        list.close();
        host.execute(&[&list]);

        let mut ptr = std::ptr::null_mut();
        // SAFETY: GPU finished writing `readback`; map for CPU read.
        unsafe {
            readback.Map(0, None, Some(&mut ptr)).expect("map readback");
            let pitch = footprint.Footprint.RowPitch as usize;
            let row_bytes = width as usize * 4;
            let src = std::slice::from_raw_parts(ptr.cast::<u8>(), total as usize);
            let mut out = Vec::with_capacity(row_bytes * height as usize);
            for y in 0..height as usize {
                let start = footprint.Offset as usize + y * pitch;
                out.extend_from_slice(&src[start..start + row_bytes]);
            }
            readback.Unmap(0, None);
            out
        }
    }

    fn fill_linear_rgba(layout: &victory_formats::texture::Layout, mem: &mut [u8], pixel: [u8; 4]) {
        let mut linear = vec![0u8; layout.desc.width as usize * layout.desc.height as usize * 4];
        for chunk in linear.as_chunks_mut::<4>().0 {
            *chunk = pixel;
        }
        let pitch = layout.desc.width as usize * 4;
        layout
            .tile(mem, 0, 0, &linear, pitch, linear.len())
            .expect("tile pixels");
    }

    fn all_pixels_eq(bytes: &[u8], expected: [u8; 4]) -> bool {
        let (chunks, rem) = bytes.as_chunks::<4>();
        rem.is_empty() && !chunks.is_empty() && chunks.iter().all(|&px| px == expected)
    }

    /// Exercises the real D3D12 WARP prelude upload path end to end:
    /// (a) initial use uploads title memory to GPU,
    /// (b) subsequent frames with unchanged title memory record zero GPU copies,
    /// (c) multi-list submissions after a modification upload only once,
    /// (d) modifying title memory triggers a fresh GPU upload with new pixels.
    #[test]
    fn warp_skips_clean_texture_uploads_and_reuploads_on_change() {
        let Some(host) = Host::new_warp() else {
            return; // WARP software adapter unavailable here
        };
        let (width, height) = (4, 4);
        let (copy, tex) = make_rgba_texture(&host, width, height);
        let layout = tex.planes[0].layout.clone();
        let mut title_mem = vec![0u8; layout.size as usize];
        let v1 = [0x11, 0x22, 0x33, 0xff];
        fill_linear_rgba(&layout, &mut title_mem, v1);

        let staging = create_staging(
            &host,
            std::ptr::null(),
            &copy.texture,
            layout.clone(),
            title_mem.as_ptr(),
        )
        .expect("create staging");
        assert_eq!(staging.gpu_uploads(), 0);

        let make_sampling_list = || {
            let list = host.new_list().expect("new list");
            list.defer_texture(staging.clone(), copy.clone());
            list.expect_texture_state(&copy, SHADER_RESOURCE);
            list.close();
            list
        };

        // (a) Submission 1: first use must upload onto the prelude and land on GPU.
        let list1 = make_sampling_list();
        host.execute(&[&list1]);
        assert_eq!(staging.gpu_uploads(), 1, "initial use must record 1 copy");
        assert_eq!(copy.state.load(Ordering::Relaxed), SHADER_RESOURCE);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v1
        ));

        // (b) Submission 2: unchanged title memory across two lists records ZERO copies.
        let list2a = make_sampling_list();
        let list2b = make_sampling_list();
        host.execute(&[&list2a, &list2b]);
        assert_eq!(
            staging.gpu_uploads(),
            1,
            "unchanged texture must record zero CopyTextureRegion uploads"
        );
        // Re-submitting the same closed list with unchanged memory also records zero copies.
        host.execute(&[&list2a]);
        assert_eq!(staging.gpu_uploads(), 1);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v1
        ));

        // (c) Submission 3: CPU modifies title memory; two lists sample it -> exactly 1 upload.
        let v2 = [0xaa, 0xbb, 0xcc, 0xff];
        fill_linear_rgba(&layout, &mut title_mem, v2);
        let list3a = make_sampling_list();
        let list3b = make_sampling_list();
        host.execute(&[&list3a, &list3b]);
        assert_eq!(
            staging.gpu_uploads(),
            2,
            "modified texture across two lists must upload exactly once"
        );
        assert_eq!(copy.state.load(Ordering::Relaxed), SHADER_RESOURCE);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v2
        ));

        // (d) Submission 4: unchanged again -> zero additional uploads.
        let list4 = make_sampling_list();
        host.execute(&[&list4]);
        assert_eq!(staging.gpu_uploads(), 2);
    }

    /// Verifies the write-watched (`MEM_WRITE_WATCH`) dirty-tracking path on
    /// real D3D12 WARP: first use uploads, clean frames skip, and a CPU page
    /// write triggers a single fresh upload.
    #[test]
    fn warp_write_watched_texture_uploads_only_when_written() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        let Some(host) = Host::new_warp() else {
            return;
        };
        let (width, height) = (4, 4);
        let (copy, tex) = make_rgba_texture(&host, width, height);
        let layout = tex.planes[0].layout.clone();
        let alloc_size = (layout.size as usize).max(crate::watch::PAGE);
        // SAFETY: allocating a private write-watched page buffer freed before returning.
        let raw = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                alloc_size,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(!raw.is_null());

        let v1 = [0x33, 0x44, 0x55, 0xff];
        // SAFETY: `raw` is committed read-write memory of `alloc_size` bytes.
        unsafe {
            let mem = std::slice::from_raw_parts_mut(raw, layout.size as usize);
            fill_linear_rgba(&layout, mem, v1);
        }

        let staging = create_staging(&host, std::ptr::null(), &copy.texture, layout.clone(), raw)
            .expect("create staging");

        let run_once = || {
            let list = host.new_list().expect("new list");
            list.defer_texture(staging.clone(), copy.clone());
            list.expect_texture_state(&copy, SHADER_RESOURCE);
            list.close();
            host.execute(&[&list]);
        };

        // Initial submission uploads.
        run_once();
        assert_eq!(staging.gpu_uploads(), 1);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v1
        ));

        // Unchanged write-watched memory skips GPU upload.
        run_once();
        assert_eq!(staging.gpu_uploads(), 1);

        // CPU write dirties the watched page -> next submission uploads.
        let v2 = [0xde, 0xad, 0xbe, 0xff];
        // SAFETY: `raw` is still committed read-write memory.
        unsafe {
            let mem = std::slice::from_raw_parts_mut(raw, layout.size as usize);
            fill_linear_rgba(&layout, mem, v2);
        }
        run_once();
        assert_eq!(staging.gpu_uploads(), 2);
        assert!(all_pixels_eq(
            &read_back_rgba(&host, &copy, width, height),
            v2
        ));

        // Clean again -> zero upload.
        run_once();
        assert_eq!(staging.gpu_uploads(), 2);

        // SAFETY: freeing our own allocation.
        unsafe { VirtualFree(raw.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn host_written_texture_never_queues_cpu_upload() {
        let Some(host) = Host::new_warp() else {
            return;
        };
        let (copy, _) = make_rgba_texture(&host, 4, 4);
        let fake_resource = 0x1234_5678usize as This;
        host_written(fake_resource);
        let list = host.new_list().expect("new list");
        upload(&list, fake_resource, &copy.texture);
        assert!(
            !list.has_deferred_work(),
            "host-written textures must not queue CPU-to-GPU uploads"
        );
        forget(fake_resource as usize);
    }

    #[test]
    fn detile_widened_packed16_expands_into_rgba8_staging() {
        for format in crate::resource::PACKED_16 {
            let desc = TextureDesc {
                dimension: Dimension::Tex2d,
                width: 17,
                height: 9,
                depth_or_array: 1,
                mips: 2,
                format,
                samples: 1,
                swizzle: Swizzle::SW_4KB_S,
            };
            let tex = Texture::new(&desc).unwrap();
            let layout = &tex.planes[0].layout;
            let mut tiled = vec![0u8; layout.size as usize];
            let mut expected_mips = Vec::new();

            for mip in 0..2u32 {
                let m = layout.mips[mip as usize];
                let row_bytes = m.width as usize * 2;
                let mut src_mip = vec![0u8; row_bytes * m.height as usize];
                for y in 0..m.height as usize {
                    for x in 0..m.width as usize {
                        let px = ((mip as u16 + 1) * 0x1111)
                            .wrapping_add((y as u16) * 0x0103 + x as u16);
                        let at = y * row_bytes + x * 2;
                        src_mip[at..at + 2].copy_from_slice(&px.to_le_bytes());
                    }
                }
                layout
                    .tile(&mut tiled, mip, 0, &src_mip, row_bytes, src_mip.len())
                    .unwrap();
                expected_mips.push(src_mip);
            }

            let mut packed = Vec::new();
            for (mip, src_mip) in expected_mips.iter().enumerate() {
                let m = layout.mips[mip];
                let width = m.width as usize;
                let rows = m.height as usize;
                let row_pitch = (width * 4).next_multiple_of(256);
                let exact_len = row_pitch * (rows - 1) + width * 4;
                let mut staging = vec![0x5au8; exact_len];
                detile_subresource(
                    layout,
                    &tiled,
                    mip as u32,
                    0,
                    &mut staging,
                    row_pitch,
                    rows,
                    1,
                    Some(format),
                    &mut packed,
                )
                .unwrap();
                for y in 0..rows {
                    let mut expected = vec![0u8; width * 4];
                    crate::resource::expand_packed16(
                        format,
                        &src_mip[y * width * 2..][..width * 2],
                        &mut expected,
                    );
                    assert_eq!(
                        &staging[y * row_pitch..][..width * 4],
                        &expected[..],
                        "format {format} mip {mip} row {y}"
                    );
                }
            }
        }
    }

    #[test]
    fn upload_pool_allocates_zero_on_clean_and_reuses_chunk() {
        let Some(host) = Host::new_warp() else {
            return;
        };
        let (width, height) = (4, 4);
        let (copy, tex) = make_rgba_texture(&host, width, height);
        let layout = tex.planes[0].layout.clone();
        let mut title_mem = vec![0u8; layout.size as usize];
        fill_linear_rgba(&layout, &mut title_mem, [0x22, 0x33, 0x44, 0xff]);
        let staging = create_staging(
            &host,
            std::ptr::null(),
            &copy.texture,
            layout.clone(),
            title_mem.as_ptr(),
        )
        .expect("create staging");

        let mut pool = UploadPool::new();
        assert_eq!(pool.stats(), (0, 0));

        // First refresh is dirty -> allocates 1 pool chunk.
        assert!(staging.refresh(&host, &mut pool).is_some());
        assert_eq!(pool.stats(), (1, 0));
        pool.reset();

        // Clean refresh -> 0 pool allocations, chunk count unchanged.
        assert!(staging.refresh(&host, &mut pool).is_none());
        assert_eq!(pool.stats(), (1, 0));

        // Modify memory -> reuses existing chunk (still 1 chunk).
        fill_linear_rgba(&layout, &mut title_mem, [0x55, 0x66, 0x77, 0xff]);
        assert!(staging.refresh(&host, &mut pool).is_some());
        assert_eq!(pool.stats(), (1, 0));
    }
}

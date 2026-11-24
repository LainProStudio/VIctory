//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, EXCEPTION_ACCESS_VIOLATION, HANDLE,
    INVALID_HANDLE_VALUE, NTSTATUS,
};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
    EXCEPTION_POINTERS,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_WRITE, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_DECOMMIT,
    MEM_EXTENDED_PARAMETER, MEM_FREE, MEM_MAPPED, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE,
    MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION,
    MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, SEC_COMMIT, SEC_RESERVE, UnmapViewOfFile,
    UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQuery, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{K32GetPerformanceInfo, PERFORMANCE_INFORMATION};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentThreadId};
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);
/// Bytes currently live in [`XMemAllocDefault`] heap blocks (all and GPU-optimal).
static HEAP_USED: AtomicUsize = AtomicUsize::new(0);
static HEAP_GPU_OPTIMAL_USED: AtomicUsize = AtomicUsize::new(0);
/// Non-placeholder [`XMemVirtualAlloc`] reservations: base -> (length, gpu_optimal).
/// Queried for committed pages when reporting working-set statistics so host
/// allocations (GPU mirrors, shader compiler) are excluded and `VirtualFree`
/// decommits/releases are observed automatically.
static VIRTUAL_REGIONS: LazyLock<Mutex<BTreeMap<usize, (usize, bool)>>> =
    LazyLock::new(Default::default);

fn saturating_sub(counter: &AtomicUsize, delta: usize) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
        Some(cur.saturating_sub(delta))
    });
}

fn record_virtual_region(at: usize, requested: usize, gpu_optimal: bool) {
    let mut mbi = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: pseudo-handle and writable MEMORY_BASIC_INFORMATION.
    let queried = unsafe {
        VirtualQueryEx(
            GetCurrentProcess(),
            at as *const c_void,
            &mut mbi,
            size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    };
    let len = if queried != 0 && mbi.RegionSize > 0 {
        mbi.RegionSize.max(requested)
    } else {
        requested
    };
    let end = at.saturating_add(len);
    let mut regions = VIRTUAL_REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    // Drop any stale entry overlapping `[at, end)` (e.g. reused after VirtualFree).
    let stale: Vec<usize> = regions
        .iter()
        .filter(|&(&base, &(rlen, _))| base < end && base.saturating_add(rlen) > at)
        .map(|(&base, _)| base)
        .collect();
    for base in stale {
        regions.remove(&base);
    }
    regions.insert(at, (len, gpu_optimal));
}

fn virtual_used() -> (usize, usize) {
    let mut regions = VIRTUAL_REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    let mut total = 0usize;
    let mut gpu_optimal = 0usize;
    let mut stale = Vec::new();
    // SAFETY: pseudo-handle needs no closing.
    let process = unsafe { GetCurrentProcess() };
    for (&base, &(len, is_gpu)) in regions.iter() {
        let end = base.saturating_add(len);
        let mut cur = base;
        let mut alive = false;
        while cur < end {
            let mut mbi = MEMORY_BASIC_INFORMATION::default();
            // SAFETY: valid process handle and writable struct.
            let ok = unsafe {
                VirtualQueryEx(
                    process,
                    cur as *const c_void,
                    &mut mbi,
                    size_of::<MEMORY_BASIC_INFORMATION>(),
                )
            };
            if ok == 0 || mbi.RegionSize == 0 {
                break;
            }
            let region_end = (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize);
            let next = region_end.min(end);
            if next <= cur {
                break;
            }
            if mbi.State != MEM_FREE && mbi.AllocationBase as usize == base {
                alive = true;
                if mbi.State == MEM_COMMIT {
                    let bytes = next - cur;
                    total = total.saturating_add(bytes);
                    if is_gpu {
                        gpu_optimal = gpu_optimal.saturating_add(bytes);
                    }
                }
            }
            cur = next;
        }
        if !alive {
            stale.push(base);
        }
    }
    for base in stale {
        regions.remove(&base);
    }
    (total, gpu_optimal)
}

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;
/// Coalesces adjacent split placeholders back into one (not in windows-sys' memory module).
const MEM_COALESCE_PLACEHOLDERS: u32 = 0x0000_0001;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (Grand Theft
/// Auto VI's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (Grand Theft Auto VI's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let at = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if constrained {
                &mut parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    if !at.is_null() && !placeholder {
        let addr = at as usize;
        let already_tracked = !constrained && {
            let regions = VIRTUAL_REGIONS.lock().unwrap_or_else(|e| e.into_inner());
            regions
                .range(..=addr)
                .next_back()
                .is_some_and(|(&rbase, &(rlen, _))| addr < rbase.saturating_add(rlen))
        };
        if !already_tracked {
            record_virtual_region(addr, size, flags & GRAPHICS != 0);
        }
    }
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (Grand Theft Auto VI's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type (4 bits at 29; non-zero = graphics/GPU),
// then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_TYPE_SHIFT: u64 = 29;
const ALLOC_TYPE_MASK: u64 = 0xf;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment (bit 0 packs `gpu_optimal`,
/// since alignment is always a power of two >= 16), just below the pointer
/// handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn is_alloc_gpu_optimal(attributes: u64) -> bool {
    ((attributes >> ALLOC_TYPE_SHIFT) & ALLOC_TYPE_MASK) != 0
}

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (Grand Theft Auto VI does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (Grand Theft Auto VI uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        return block.cast();
    }
    let gpu_optimal = is_alloc_gpu_optimal(attributes);
    HEAP_USED.fetch_add(layout.size(), Ordering::Relaxed);
    if gpu_optimal {
        HEAP_GPU_OPTIMAL_USED.fetch_add(layout.size(), Ordering::Relaxed);
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes. Alignment is a power of two >= 16 so bit 0 is free.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align() | usize::from(gpu_optimal)]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            VIRTUAL_REGIONS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&(address as usize));
            VirtualFree(address, 0, MEM_RELEASE);
            return;
        }
        let user = address.cast::<u8>();
        let [size, packed_align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        let gpu_optimal = (packed_align & 1) != 0;
        let align = packed_align & !1;
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
        saturating_sub(&HEAP_USED, size);
        if gpu_optimal {
            saturating_sub(&HEAP_GPU_OPTIMAL_USED, size);
        }
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;

struct Section {
    handle: HANDLE,
    gpu_optimal: bool,
    /// Persistent linear view of the whole section (placed above `2^40` when
    /// possible so console heaps keep the low 1 TiB). Eliminates per-request
    /// `MapViewOfFile`/`UnmapViewOfFile` TLB shootdowns during streaming.
    view: MEMORY_MAPPED_VIEW_ADDRESS,
    pages: usize,
    /// One bit per 64 KiB page: set once the page is committed (on Wine all
    /// bits start set because sections are created `SEC_COMMIT`).
    committed: Box<[AtomicU64]>,
}

// SAFETY: a section handle, its persistent view, and its atomic bitmap may be
// accessed from any thread (callers synchronize disjoint byte ranges).
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Section {
    fn new(handle: HANDLE, pages: usize, gpu_optimal: bool) -> Option<Self> {
        let size = pages.checked_mul(PAGE)?;
        let view = map_persistent_view(handle, size);
        if view.Value.is_null() {
            return None;
        }
        let words = pages.div_ceil(64);
        let init = if on_wine() { u64::MAX } else { 0 };
        let committed = (0..words).map(|_| AtomicU64::new(init)).collect();
        Some(Self {
            handle,
            gpu_optimal,
            view,
            pages,
            committed,
        })
    }

    fn page_ptr(&self, index: usize) -> *mut u8 {
        debug_assert!(index < self.pages);
        self.view.Value.cast::<u8>().wrapping_add(index * PAGE)
    }

    fn is_committed(&self, page: usize) -> bool {
        (self.committed[page / 64].load(Ordering::Acquire) & (1u64 << (page % 64))) != 0
    }

    fn word_mask(start_bit: usize, end_bit: usize) -> u64 {
        debug_assert!(start_bit < end_bit && end_bit <= 64);
        let width = end_bit - start_bit;
        let ones = if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        ones << start_bit
    }

    fn all_committed(&self, start: usize, count: usize) -> bool {
        let end = start + count;
        let first_word = start / 64;
        let last_word = (end - 1) / 64;
        for w in first_word..=last_word {
            let lo = if w == first_word { start % 64 } else { 0 };
            let hi = if w == last_word {
                ((end - 1) % 64) + 1
            } else {
                64
            };
            let mask = Self::word_mask(lo, hi);
            if self.committed[w].load(Ordering::Acquire) & mask != mask {
                return false;
            }
        }
        true
    }

    fn mark_committed(&self, start: usize, count: usize) {
        let end = start + count;
        let first_word = start / 64;
        let last_word = (end - 1) / 64;
        for w in first_word..=last_word {
            let lo = if w == first_word { start % 64 } else { 0 };
            let hi = if w == last_word {
                ((end - 1) % 64) + 1
            } else {
                64
            };
            let mask = Self::word_mask(lo, hi);
            self.committed[w].fetch_or(mask, Ordering::Release);
        }
    }

    /// Commits any uncommitted pages in `[start, start + count)` via the
    /// persistent view once per page, publishing completion with Release order.
    fn ensure_committed(&self, start: usize, count: usize) -> bool {
        if count == 0 || on_wine() {
            return true;
        }
        debug_assert!(start.checked_add(count).is_some_and(|e| e <= self.pages));
        if self.all_committed(start, count) {
            return true;
        }
        let end = start + count;
        let mut i = start;
        while i < end {
            if self.is_committed(i) {
                i += 1;
                continue;
            }
            let run_start = i;
            i += 1;
            while i < end && !self.is_committed(i) {
                i += 1;
            }
            let run_len = i - run_start;
            if !commit(self.page_ptr(run_start) as usize, run_len * PAGE) {
                return false;
            }
            self.mark_committed(run_start, run_len);
        }
        true
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        // SAFETY: our persistent view and section handle; title views hold
        // their own kernel references to the section until unmapped.
        unsafe {
            if !self.view.Value.is_null() {
                UnmapViewOfFile(self.view);
            }
            CloseHandle(self.handle);
        }
    }
}

/// Maps a section linearly in host virtual address space, preferring addresses
/// above `2^40` so title reservations below `2^40` are never crowded out.
fn map_persistent_view(handle: HANDLE, size: usize) -> MEMORY_MAPPED_VIEW_ADDRESS {
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: (1usize << 40) as *mut c_void,
        HighestEndingAddress: std::ptr::null_mut(),
        Alignment: 0,
    };
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // SAFETY: `handle` is our newly created section and `parameter` points at
    // `requirements`, which outlives the call.
    let view = unsafe {
        MapViewOfFile3(
            handle,
            GetCurrentProcess(),
            std::ptr::null(),
            0,
            size,
            0,
            PAGE_READWRITE,
            &mut parameter,
            1,
        )
    };
    if !view.Value.is_null() {
        return view;
    }
    // Fallback if the host rejects the extended address requirement.
    // SAFETY: `handle` is our newly created section.
    unsafe { MapViewOfFile(handle, FILE_MAP_WRITE, 0, 0, size) }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

#[derive(Default)]
struct Physical {
    next_page: usize,
    pages: HashMap<usize, Slice>,
    gpu_optimal_pages: usize,
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice)>,
    /// Reverse index from `Arc::as_ptr(&section) as usize` to mapped
    /// `(view_base, start_page_index, page_count)` tuples.
    section_views: HashMap<usize, Vec<(usize, usize, usize)>>,
}

static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(|| {
    Mutex::new(Physical {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        ..Default::default()
    })
});

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    let Some(size) = n.checked_mul(PAGE).filter(|&s| s > 0) else {
        return 0;
    };
    // A pagefile-backed section of `size` bytes, its pages committed when
    // first mapped or written (`commit`): Grand Theft Auto VI asks for 6.1
    // GiB at once, which a busy host often cannot commit (error 1450).
    // SAFETY: plain section creation.
    let handle = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            // Under Wine a committed section is a sparse file, free until
            // touched, and committing a reserved one page by page costs a
            // wineserver request each that scans every range committed so
            // far: commit it whole there.
            PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
            (size as u64 >> 32) as u32,
            size as u32,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        let error = std::io::Error::last_os_error();
        tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    }
    let gpu_optimal = flags & GRAPHICS != 0;
    let Some(section) = Section::new(handle, n, gpu_optimal) else {
        let error = std::io::Error::last_os_error();
        tracing::warn!(n, %error, "XMemAllocatePhysicalPages: persistent view failed");
        // SAFETY: `handle` was just created above and not yet owned by a `Section`.
        unsafe { CloseHandle(handle) };
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    };
    tracing::debug!(
        n,
        flags = format_args!("{flags:#x}"),
        view = format_args!("{:p}", section.view.Value),
        "XMemAllocatePhysicalPages"
    );
    let section = Arc::new(section);
    let mut physical = physical();
    let first = physical.next_page;
    physical.next_page += n * FRAMES;
    if gpu_optimal {
        physical.gpu_optimal_pages = physical.gpu_optimal_pages.saturating_add(n);
    }
    for index in 0..n {
        let slice = Slice {
            section: section.clone(),
            index,
        };
        let number = first + index * FRAMES;
        physical.pages.insert(number, slice);
        // SAFETY: the title passes room for `n` page numbers.
        unsafe { pages.add(index).write(number) };
    }
    1
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: Grand Theft Auto VI keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    let mut released: Vec<Arc<Section>> = Vec::new();
    {
        let mut physical = physical();
        for i in 0..n {
            // SAFETY: the title passes `n` page numbers.
            if let Some(slice) = physical.pages.remove(&unsafe { *pages.add(i) }) {
                if slice.section.gpu_optimal {
                    physical.gpu_optimal_pages = physical.gpu_optimal_pages.saturating_sub(1);
                }
                if !released
                    .last()
                    .is_some_and(|s| Arc::ptr_eq(s, &slice.section))
                {
                    released.push(slice.section);
                }
            }
        }
    }
    // Unmap persistent views and close handles outside the `PHYSICAL` lock.
    drop(released);
    1
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table, which the
/// title's own mapping calls share with every DirectStorage worker. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let physical = physical();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = physical.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// A contiguous range of pages in one `Section` together with the exact byte
/// span `[within, within + len)` written inside it.
struct SectionWrite {
    section: Arc<Section>,
    start_page: usize,
    within: usize,
    len: usize,
}

fn collect_section_write_ranges(
    physical: &Physical,
    section: &Arc<Section>,
    start_page: usize,
    within: usize,
    len: usize,
    out: &mut Vec<(usize, usize)>,
) {
    if len == 0 {
        return;
    }
    let key = Arc::as_ptr(section) as usize;
    let Some(views) = physical.section_views.get(&key) else {
        return;
    };
    let write_start = start_page * PAGE + within;
    let write_end = write_start + len;
    for &(view_base, v_start, v_count) in views {
        let view_start = v_start * PAGE;
        let view_end = (v_start + v_count) * PAGE;
        let overlap_start = write_start.max(view_start);
        let overlap_end = write_end.min(view_end);
        if overlap_start < overlap_end {
            out.push((
                view_base + (overlap_start - view_start),
                overlap_end - overlap_start,
            ));
        }
    }
}

fn coalesce_ranges(ranges: &mut Vec<(usize, usize)>) {
    if ranges.len() <= 1 {
        return;
    }
    ranges.sort_unstable_by_key(|&(start, len)| (start, len));
    let mut write = 0;
    for i in 1..ranges.len() {
        let (cur_start, cur_len) = ranges[write];
        let cur_end = cur_start.saturating_add(cur_len);
        let (next_start, next_len) = ranges[i];
        let next_end = next_start.saturating_add(next_len);
        if next_start <= cur_end {
            ranges[write].1 = cur_end.max(next_end) - cur_start;
        } else {
            write += 1;
            ranges[write] = (next_start, next_len);
        }
    }
    ranges.truncate(write + 1);
}

/// Collects mapped virtual `(va_start, len)` ranges across `runs` for the exact
/// byte span `[within, within + total_len)` under `PHYSICAL`, releasing the
/// lock before returning.
fn mapped_ranges_for(
    runs: &[(Slice, usize)],
    mut within: usize,
    total_len: usize,
) -> Vec<(usize, usize)> {
    if total_len == 0 {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    {
        let physical = physical();
        let mut done = 0;
        for (slice, n) in runs {
            let k = (n * PAGE - within).min(total_len - done);
            collect_section_write_ranges(
                &physical,
                &slice.section,
                slice.index,
                within,
                k,
                &mut ranges,
            );
            within = 0;
            done += k;
        }
    }
    coalesce_ranges(&mut ranges);
    ranges
}

fn mapped_ranges_for_write(write: &SectionWrite) -> Vec<(usize, usize)> {
    if write.len == 0 {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    {
        let physical = physical();
        collect_section_write_ranges(
            &physical,
            &write.section,
            write.start_page,
            write.within,
            write.len,
            &mut ranges,
        );
    }
    coalesce_ranges(&mut ranges);
    ranges
}

fn notify_mapped_writes(runs: &[(Slice, usize)], within: usize, total_len: usize) {
    for (va, len) in mapped_ranges_for(runs, within, total_len) {
        victory_gpu::kernel_will_write(va, len);
    }
}

fn notify_mapped_page_window(write: &SectionWrite) {
    for (va, len) in mapped_ranges_for_write(write) {
        victory_gpu::kernel_will_write(va, len);
    }
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// Writes directly through each section's persistent view (committing each
/// page once on first touch) with no transient map/unmap TLB shootdowns.
/// False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        if !slice.section.ensure_committed(slice.index, *n) {
            return false;
        }
        let k = (n * PAGE - within).min(data.len() - done);
        // SAFETY: `[slice.index, slice.index + *n)` are committed pages in
        // `slice.section`'s persistent view and `within + k <= *n * PAGE`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().add(done),
                slice.section.page_ptr(slice.index).add(within),
                k,
            );
        }
        (within, done) = (0, done + k);
    }
    notify_mapped_writes(&runs, offset % PAGE, data.len());
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// through the section's persistent view: DirectStorage decodes directly
/// into it without transient views or extra copies.
pub(crate) struct PageWindow {
    write: SectionWrite,
    ptr: *mut u8,
}

// SAFETY: `ptr` points into `write.section`'s persistent read-write view; each
// caller writes only its own request's byte range.
unsafe impl Send for PageWindow {}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: `write.section` keeps the persistent view mapped and
        // `ensure_committed` committed all backing pages before constructing `PageWindow`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.write.len) }
    }
}

impl Drop for PageWindow {
    fn drop(&mut self) {
        notify_mapped_page_window(&self.write);
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    if !slice.section.ensure_committed(slice.index, *n) {
        return None;
    }
    let within = offset % PAGE;
    // SAFETY: `slice.index + *n <= slice.section.pages` and `within + len <= *n * PAGE`.
    let ptr = unsafe { slice.section.page_ptr(slice.index).add(within) };
    let write = SectionWrite {
        section: slice.section.clone(),
        start_page: slice.index,
        within,
        len,
    };
    notify_mapped_page_window(&write);
    Some(PageWindow { write, ptr })
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    if n == 0 {
        return address;
    }
    let base = address as usize;
    let mut physical = physical();
    let _remap = RemapGuard::enter();
    unmap(&mut physical, base, n * PAGE);
    if pages.is_null() {
        return address;
    }
    // SAFETY: the title passes `n` page numbers.
    let pages = unsafe { std::slice::from_raw_parts(pages, n) };
    let mut i = 0;
    while i < n {
        let Some(first) = physical.pages.get(&pages[i]).cloned() else {
            tracing::warn!(page = pages[i], "XMemMapPhysicalPages: unknown page");
            return std::ptr::null_mut();
        };
        // One view per run of consecutive slices of one section.
        let run = 1 + pages[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(page, k)| {
                physical.pages.get(page).is_some_and(|s| {
                    Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + k
                })
            })
            .count();
        if !map_or_split(&mut physical, base + i * PAGE, run * PAGE, first) {
            return std::ptr::null_mut();
        }
        i += run;
    }
    address
}

/// One page at each address; null `pages`, or a page number 0, unmaps.
/// Coalesces consecutive addresses that unmap or map contiguous slices of
/// one section so bulk scatter updates avoid per-page view churn.
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    if n == 0 {
        return 1;
    }
    // SAFETY: the title passes `n` addresses and, unless null, `n` page numbers.
    let addresses = unsafe { std::slice::from_raw_parts(addresses, n) };
    let pages = if pages.is_null() {
        None
    } else {
        // SAFETY: checked non-null; the title passes `n` page numbers.
        Some(unsafe { std::slice::from_raw_parts(pages, n) })
    };
    let page_at = |idx: usize| pages.map_or(0, |p| p[idx]);

    let mut physical = physical();
    let _remap = RemapGuard::enter();
    let mut i = 0;
    while i < n {
        let base = addresses[i] as usize;
        if base == 0 {
            return 0;
        }
        let page = page_at(i);
        if page == 0 {
            let mut run = 1;
            while i + run < n
                && addresses[i + run] as usize == base.wrapping_add(run * PAGE)
                && page_at(i + run) == 0
            {
                run += 1;
            }
            unmap(&mut physical, base, run * PAGE);
            i += run;
            continue;
        }
        let Some(first) = physical.pages.get(&page).cloned() else {
            unmap(&mut physical, base, PAGE);
            tracing::warn!(page, "XMemMapPhysicalPagesScatter: unknown page");
            return 0;
        };
        let mut run = 1;
        while i + run < n && addresses[i + run] as usize == base.wrapping_add(run * PAGE) {
            let next = page_at(i + run);
            if next == 0 {
                break;
            }
            let contiguous = physical.pages.get(&next).is_some_and(|s| {
                Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + run
            });
            if !contiguous {
                break;
            }
            run += 1;
        }
        unmap(&mut physical, base, run * PAGE);
        if !map_or_split(&mut physical, base, run * PAGE, first) {
            return 0;
        }
        i += run;
    }
    1
}

/// Looks up the mapped view covering `at` and returns its starting slice plus
/// how many contiguous bytes (up to `max_bytes`, rounded to whole pages) remain
/// within that view.
fn lookup_view_run(physical: &Physical, at: usize, max_bytes: usize) -> Option<(Slice, usize)> {
    let (view_base, (view_size, view_slice)) = physical
        .views
        .range(..=at)
        .next_back()
        .filter(|(base, (size, _))| at < *base + size)?;
    let offset_pages = (at - *view_base) / PAGE;
    let avail_pages = (*view_base + *view_size - at) / PAGE;
    let need_pages = max_bytes.div_ceil(PAGE);
    let run_pages = avail_pages.min(need_pages);
    if run_pages == 0 {
        return None;
    }
    let slice = Slice {
        section: view_slice.section.clone(),
        index: view_slice.index + offset_pages,
    };
    Some((slice, run_pages * PAGE))
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. Used by the GPU's
/// page-mapping copies (`victory_gpu`, queue `CopyPageMappingsBatchX`).
/// Coalesces consecutive source pages from the same section into single views;
/// returns false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    if len == 0 {
        return true;
    }
    let mut physical = physical();
    let _remap = RemapGuard::enter();
    unmap(&mut physical, to, len);
    if from == 0 {
        return true;
    }
    let mut k = 0;
    while k < len {
        let Some((first_slice, mut run_bytes)) = lookup_view_run(&physical, from + k, len - k)
        else {
            return false;
        };
        // Also coalesce adjacent source views that continue the same section run.
        while k + run_bytes < len {
            let Some((next_slice, next_bytes)) =
                lookup_view_run(&physical, from + k + run_bytes, len - k - run_bytes)
            else {
                break;
            };
            if Arc::ptr_eq(&first_slice.section, &next_slice.section)
                && next_slice.index == first_slice.index + run_bytes / PAGE
            {
                run_bytes += next_bytes;
            } else {
                break;
            }
        }
        if !map_or_split(&mut physical, to + k, run_bytes, first_slice) {
            return false;
        }
        k += run_bytes;
    }
    true
}

/// Maps `[base, base + len)` in one view when possible; if a multi-page
/// coalesced run crosses separate placeholder reservations, falls back to
/// mapping one page at a time.
fn map_or_split(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    if len <= PAGE {
        return map(physical, base, len, slice, true);
    }
    if map(physical, base, len, slice.clone(), false) {
        return true;
    }
    let pages = len / PAGE;
    for k in 0..pages {
        let page_slice = Slice {
            section: slice.section.clone(),
            index: slice.index + k,
        };
        if !map(physical, base + k * PAGE, PAGE, page_slice, true) {
            return false;
        }
    }
    true
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in one placeholder.
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice, warn_on_fail: bool) -> bool {
    if !slice.section.ensure_committed(slice.index, len / PAGE) {
        return false;
    }
    // SAFETY: coalesces adjacent placeholders if split, splits the placeholder
    // around the range (failing harmlessly when the range is the whole
    // placeholder), then maps the section there; all act only on placeholders.
    let view = unsafe {
        if len > PAGE {
            VirtualFree(
                base as *mut c_void,
                len,
                MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS,
            );
        }
        VirtualFree(
            base as *mut c_void,
            len,
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        );
        MapViewOfFile3(
            slice.section.handle,
            GetCurrentProcess(),
            base as *const c_void,
            (slice.index * PAGE) as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if view.Value as usize != base {
        if warn_on_fail {
            tracing::warn!(
                address = format_args!("{base:#x}"),
                len,
                error = %std::io::Error::last_os_error(),
                "mapping physical pages failed (not a placeholder?)"
            );
        }
        return false;
    }
    // New contents, read-write: the GPU side's copies no longer guard it.
    victory_gpu::title_memory_remapped(base, len);
    let section_key = Arc::as_ptr(&slice.section) as usize;
    physical
        .section_views
        .entry(section_key)
        .or_default()
        .push((base, slice.index, len / PAGE));
    physical.views.insert(base, (len, slice));
    true
}

/// How much one `VirtualAlloc` commits. The host charges the whole call
/// before it answers and refuses one larger than the page file's free commit
/// (error 1455), so a huge contiguous range fails as a unit; in pieces the
/// same request commits as far as the host allows.
const COMMIT_PIECE: usize = 64 * MIB;

/// The host's commit limit and how much of it is taken, in bytes
/// (`PERFORMANCE_INFORMATION` counts pages).
fn commit_budget() -> (usize, usize) {
    // SAFETY: plain counters, zeroed and sized for the host.
    let mut info: PERFORMANCE_INFORMATION = unsafe { std::mem::zeroed() };
    info.cb = size_of::<PERFORMANCE_INFORMATION>() as u32;
    // SAFETY: our own process and a structure sized by `cb`.
    unsafe { K32GetPerformanceInfo(&raw mut info, info.cb) };
    (
        info.CommitLimit.saturating_mul(info.PageSize),
        info.CommitTotal.saturating_mul(info.PageSize),
    )
}

/// Commits `[base, base + len)` piece by piece and returns the bytes
/// committed before the host refused.
fn commit_pieces(base: usize, len: usize) -> usize {
    let mut done = 0;
    while done < len {
        let piece = COMMIT_PIECE.min(len - done);
        // SAFETY: `[base + done, base + done + piece)` is reserved section, in `len`.
        let at = unsafe {
            VirtualAlloc(
                (base + done) as *const c_void,
                piece,
                MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        if at.is_null() {
            let error = std::io::Error::last_os_error();
            let (limit, used) = commit_budget();
            tracing::warn!(
                done,
                len,
                commit_limit = limit,
                commit_used = used,
                %error,
                "commit stopped at the host's commit limit"
            );
            if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
                let mut info = PERFORMANCE_INFORMATION::default();
                info.cb = size_of::<PERFORMANCE_INFORMATION>() as u32;
                // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
                let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
                victory_loader::log_crash(&commit_limit_message(
                    piece,
                    &error,
                    queried.then_some(&info),
                ));
            }
            break;
        }
        done += piece;
    }
    done
}

/// Commits the section pages a view shows (once per page, whichever view):
/// all of them or nothing, because a map has no count to report a short one.
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    let committed = commit_pieces(base, len);
    if committed == len {
        return true;
    }
    if committed > 0 {
        // SAFETY: the committed head of a range whose `ensure_committed` failed;
        // its charge must not outlive the attempt.
        unsafe { VirtualFree(base as *mut c_void, committed, MEM_DECOMMIT) };
    }
    false
}

fn commit_limit_message(
    len: usize,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "VIctory: Windows commit limit reached while committing {len} bytes of physical pages: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Non-zero while `PHYSICAL` holds an active view split or whole-view
/// replacement (`UnmapViewOfFile2(..., MEM_PRESERVE_PLACEHOLDER)` ->
/// `MapViewOfFile3`), during which pages are temporarily placeholders;
/// concurrent threads hitting them wait for `PHYSICAL` and retry once mapped.
static SPLITTING: AtomicUsize = AtomicUsize::new(0);
static SPLITTER_TID: AtomicU32 = AtomicU32::new(0);

/// RAII guard protecting the placeholder window while holding `PHYSICAL`.
struct RemapGuard {
    prev_tid: u32,
}

impl RemapGuard {
    fn enter() -> Self {
        install_split_handler();
        // SAFETY: plain query of the calling thread id.
        let tid = unsafe { GetCurrentThreadId() };
        let prev_tid = SPLITTER_TID.swap(tid, Ordering::AcqRel);
        SPLITTING.fetch_add(1, Ordering::SeqCst);
        Self { prev_tid }
    }
}

impl Drop for RemapGuard {
    fn drop(&mut self) {
        SPLITTER_TID.store(self.prev_tid, Ordering::Release);
        SPLITTING.fetch_sub(1, Ordering::SeqCst);
    }
}

unsafe extern "system" fn on_split_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: the system passes valid exception pointers.
    let r = unsafe { &*(*info).ExceptionRecord };
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let kind = r.ExceptionInformation[0];
    if kind != 0 && kind != 1 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if SPLITTING.load(Ordering::Acquire) != 0 {
        // SAFETY: plain query of the calling thread id.
        if unsafe { GetCurrentThreadId() } == SPLITTER_TID.load(Ordering::Acquire) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // Wait until the active split finishes remapping the untouched ranges.
        drop(physical());
    }
    let at = r.ExceptionInformation[1];
    // SAFETY: plain data filled in by the query.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut mbi, size_of_val(&mbi)) };
    if n != 0 && mbi.State == MEM_COMMIT && mbi.Type == MEM_MAPPED {
        let ok = match kind {
            0 => {
                mbi.Protect
                    & (PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE)
                    != 0
            }
            1 => mbi.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0,
            _ => false,
        };
        if ok {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    EXCEPTION_CONTINUE_SEARCH
}

fn install_split_handler() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: registered once ahead of the crash reporter; inspects only
    // access violations during view splits after serializing on `PHYSICAL`.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_split_fault));
    });
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = base.saturating_add(len);
    // Views do not overlap: the one holding `base`, then those starting
    // inside.
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(physical.views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    if hit.is_empty() {
        return;
    }
    let _remap = RemapGuard::enter();
    for at in hit {
        let Some((size, slice)) = physical.views.remove(&at) else {
            continue;
        };
        let section_key = Arc::as_ptr(&slice.section) as usize;
        if let Some(entries) = physical.section_views.get_mut(&section_key) {
            if let Some(pos) = entries.iter().position(|&(v_base, v_start, v_count)| {
                v_base == at && v_start == slice.index && v_count == size / PAGE
            }) {
                entries.swap_remove(pos);
            }
            if entries.is_empty() {
                physical.section_views.remove(&section_key);
            }
        }
        victory_gpu::title_memory_remapped(at, size);
        // SAFETY: a view we mapped; it becomes a placeholder again.
        unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
        if at < base {
            map(physical, at, base - at, slice.clone(), true);
        }
        if at + size > end {
            let index = slice.index + (end - at) / PAGE;
            map(
                physical,
                end,
                at + size - end,
                Slice { index, ..slice },
                true,
            );
        }
        let hole_start = base.max(at);
        let hole_end = end.min(at + size);
        victory_gpu::title_memory_remapped(hole_start, hole_end.saturating_sub(hole_start));
    }
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and its tracked allocations (`XMemAllocDefault`,
/// committed `XMemVirtualAlloc` regions, and allocated physical pages),
/// excluding host-side allocations such as D3D12 mirrors and shader compiler
/// memory. No tools memory: a retail console.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let (virt_total, virt_gpu) = virtual_used();
    let (phys_pages, phys_gpu_pages) = {
        let physical = physical();
        (physical.pages.len(), physical.gpu_optimal_pages)
    };
    let heap_total = HEAP_USED.load(Ordering::Relaxed);
    let heap_gpu = HEAP_GPU_OPTIMAL_USED.load(Ordering::Relaxed);
    let title_used = virt_total
        .saturating_add(phys_pages.saturating_mul(PAGE))
        .saturating_add(heap_total);
    let gpu_optimal_used = virt_gpu
        .saturating_add(phys_gpu_pages.saturating_mul(PAGE))
        .saturating_add(heap_gpu);
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used,
        physical_pages: u32::try_from(phys_pages).unwrap_or(u32::MAX),
        gpu_optimal_limit: gpu_optimal,
        gpu_optimal_used,
        ..Default::default()
    };
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    // Before the title can ask for its 6.5 GB: name the ceiling the host
    // will hold it to, so a short physical-page count or commit failure is
    // explained up front instead of surfacing as an opaque failure later.
    let (limit, used) = commit_budget();
    tracing::info!(
        commit_limit = limit,
        commit_used = used,
        available = limit.saturating_sub(used),
        "xmem: host commit budget"
    );
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(watched, 0);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_limit > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn working_set_tracks_title_allocations_and_excludes_host_memory() {
        let stats = || {
            let mut ws = WorkingSet::default();
            assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut ws), 1);
            ws
        };

        // Host allocations (mirrors, shader compiler) must not inflate title_used.
        let before_host = stats();
        let mut host_buf = vec![0u8; 32 * MIB];
        for page in host_buf.chunks_mut(4096) {
            page[0] = 1;
        }
        std::hint::black_box(&host_buf);
        let after_host = stats();
        assert!(
            after_host.title_used < before_host.title_used + 16 * MIB,
            "host memory should not be counted in title_used: before={} after={}",
            before_host.title_used,
            after_host.title_used
        );
        drop(host_buf);

        // Uncommitted virtual reservations count 0; committing adds exact bytes;
        // VirtualFree(MEM_RELEASE) subtracts them automatically.
        let before_virt = stats();
        let virt = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 * MIB,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!virt.is_null());
        let after_reserve = stats();
        assert!(
            after_reserve.title_used < before_virt.title_used + 8 * MIB,
            "uncommitted reserve should not be counted"
        );
        let committed = XMemVirtualAlloc(virt, 16 * MIB, MEM_COMMIT, GRAPHICS, PAGE_READWRITE);
        assert_eq!(committed, virt);
        let after_commit = stats();
        assert!(after_commit.title_used >= before_virt.title_used + 12 * MIB);
        assert!(after_commit.gpu_optimal_used >= before_virt.gpu_optimal_used + 12 * MIB);
        // SAFETY: our reservation from above.
        assert_ne!(unsafe { VirtualFree(virt, 0, MEM_RELEASE) }, 0);
        let after_release = stats();
        assert!(after_release.title_used + 10 * MIB <= after_commit.title_used);
        assert!(after_release.gpu_optimal_used + 10 * MIB <= after_commit.gpu_optimal_used);

        // XMemAllocDefault tracks standard vs GPU-optimal allocations and frees.
        let before_heap = stats();
        let std_blk = XMemAllocDefault(16 * MIB, 0);
        let gpu_blk = XMemAllocDefault(16 * MIB, 1 << ALLOC_TYPE_SHIFT);
        assert!(!std_blk.is_null() && !gpu_blk.is_null());
        let after_heap = stats();
        assert!(after_heap.title_used >= before_heap.title_used + 28 * MIB);
        assert!(after_heap.gpu_optimal_used >= before_heap.gpu_optimal_used + 12 * MIB);
        assert!(after_heap.gpu_optimal_used < before_heap.gpu_optimal_used + 24 * MIB);
        XMemFreeDefault(std_blk, 0);
        XMemFreeDefault(gpu_blk, 0);
        let after_heap_free = stats();
        assert!(after_heap_free.title_used + 24 * MIB <= after_heap.title_used);
        assert!(after_heap_free.gpu_optimal_used + 10 * MIB <= after_heap.gpu_optimal_used);

        // Physical pages track total count, title_used, and gpu_optimal_used.
        let before_phys = stats();
        let mut pages = [0usize; 64];
        let mut count = pages.len();
        assert_eq!(
            XMemAllocatePhysicalPages(GRAPHICS, &mut count, pages.as_mut_ptr()),
            1
        );
        assert_eq!(count, 64);
        let after_phys = stats();
        assert!(after_phys.physical_pages >= before_phys.physical_pages + 50);
        assert!(after_phys.gpu_optimal_used >= before_phys.gpu_optimal_used + 3 * MIB);
        assert_eq!(XMemFreePhysicalPages(count, pages.as_ptr()), 1);
        let after_phys_free = stats();
        assert!(after_phys_free.physical_pages + 50 <= after_phys.physical_pages);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, &error, Some(&info));
        assert!(report.contains("4194304 bytes"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, &error, None);
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }
    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through its section's persistent view.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = physical().pages.get(&page).cloned().expect("known page");
        assert!(slice.section.ensure_committed(slice.index, 1));
        // SAFETY: `slice.index` is a committed page in `slice.section`'s persistent view.
        unsafe { std::slice::from_raw_parts(slice.section.page_ptr(slice.index), PAGE) }.to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn concurrent_access_survives_view_split() {
        use std::sync::atomic::AtomicBool;
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        let p0 = a as usize;
        let p1 = p0 + PAGE;
        let p2 = p0 + 2 * PAGE;

        for round in 0..50u32 {
            assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
            // SAFETY: all 3 pages are mapped read-write.
            unsafe {
                (p0 as *mut u32).write_volatile(0x1111_2222);
                (p2 as *mut u32).write_volatile(0x3333_4444);
            }
            let started = Arc::new(AtomicBool::new(false));
            let stop = Arc::new(AtomicBool::new(false));
            let started_w = started.clone();
            let stop_w = stop.clone();
            let worker = std::thread::spawn(move || {
                let mut iters = 0u64;
                while !stop_w.load(Ordering::Relaxed) {
                    // SAFETY: outer pages stay mapped throughout middle-page split.
                    unsafe {
                        let v0 = (p0 as *const u32).read_volatile();
                        assert_eq!(v0, 0x1111_2222);
                        ((p0 + 4) as *mut u32).write_volatile(round ^ iters as u32);
                        let v2 = (p2 as *const u32).read_volatile();
                        assert_eq!(v2, 0x3333_4444);
                        ((p2 + 4) as *mut u32).write_volatile(round ^ iters as u32);
                    }
                    if iters == 0 {
                        started_w.store(true, Ordering::Release);
                    }
                    iters += 1;
                }
                iters
            });
            while !started.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            // Forces a real split of the 3-page coalesced view into `[p0]` and `[p2]`.
            assert_eq!(
                XMemMapPhysicalPages(p1 as *mut c_void, 1, std::ptr::null()),
                p1 as *mut c_void
            );
            stop.store(true, Ordering::Release);
            let iters = worker.join().expect("worker must survive view split");
            assert!(iters > 0);
            assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        }
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(a, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn persistent_section_view_commits_once_and_cleans_up_on_drop() {
        // 70 pages spans two 64-bit committed-bitmap words.
        let pages = allocate(70);
        let section = physical()
            .pages
            .get(&pages[0])
            .expect("allocated page")
            .section
            .clone();
        let view_addr = section.view.Value;
        assert!(!view_addr.is_null());
        if !on_wine() {
            assert!(
                !section.is_committed(0) && !section.is_committed(63) && !section.is_committed(64),
                "pages start uncommitted on Windows"
            );
        }

        // Write across the 64-page bitmap word boundary (pages 62..66).
        let payload: Vec<u8> = (0..3 * PAGE + 123).map(|i| (i * 13 % 251) as u8).collect();
        assert!(write_physical(&pages[62..66], 17, &payload));
        assert!(section.all_committed(62, 4));

        // Write via `page_window` on pages 0..3.
        {
            let mut win = page_window(&pages[0..3], 9, 2 * PAGE).expect("one run");
            win.bytes().fill(0x3c);
        }
        assert!(section.all_committed(0, 3));
        if !on_wine() {
            assert!(
                !section.is_committed(10),
                "untouched page stays uncommitted"
            );
        }

        // Map pages 62..66 into a title placeholder AFTER committing via the
        // persistent view: the new view must immediately report MEM_COMMIT and
        // expose the written bytes.
        let reserve = XMemVirtualAlloc(
            std::ptr::null_mut(),
            6 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!reserve.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(
            XMemMapPhysicalPages(reserve, 4, pages[62..66].as_ptr()),
            reserve
        );
        assert_eq!(query(reserve).host.State, MEM_COMMIT);
        // SAFETY: 4 pages are mapped read-write.
        let mapped =
            unsafe { std::slice::from_raw_parts(reserve.cast::<u8>().add(17), payload.len()) };
        assert_eq!(mapped, payload);

        // Map untouched pages 10..12 BEFORE touching the persistent view:
        // mapping commits them in the bitmap and writes show through `page_bytes`.
        assert_eq!(
            XMemMapPhysicalPages(at(reserve, 4), 2, pages[10..12].as_ptr()),
            at(reserve, 4)
        );
        assert_eq!(query(at(reserve, 4)).host.State, MEM_COMMIT);
        assert!(section.all_committed(10, 2));
        // SAFETY: mapped read-write.
        unsafe { at(reserve, 4).cast::<u8>().add(5).write(0x7e) };
        assert_eq!(page_bytes(pages[10])[5], 0x7e);

        // Unmap title views, free all pages, and drop our local `Arc<Section>`:
        // the persistent view must be unmapped on section drop.
        assert_eq!(XMemMapPhysicalPages(reserve, 6, std::ptr::null()), reserve);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(reserve, 0, MEM_RELEASE) }, 0);
        assert_eq!(XMemFreePhysicalPages(pages.len(), pages.as_ptr()), 1);
        drop(section);
        assert_eq!(
            query(view_addr).host.State,
            windows_sys::Win32::System::Memory::MEM_FREE
        );
    }

    #[test]
    fn scatter_and_alias_coalesce_contiguous_runs_and_handle_zero_len() {
        let (a_pages, b_pages) = (allocate(5), allocate(2));
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                6 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);

        // Scatter 4 contiguous pages from `a_pages`, 1 from `b_pages`, and 1 unmap (0).
        let addrs = [at(a, 0), at(a, 1), at(a, 2), at(a, 3), at(a, 4), at(a, 5)];
        let map_pages = [
            a_pages[0], a_pages[1], a_pages[2], a_pages[3], b_pages[0], 0,
        ];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addrs.as_ptr(), addrs.len(), map_pages.as_ptr()),
            1
        );
        {
            let phys = physical();
            assert_eq!(
                phys.views.get(&(a as usize)).map(|(len, _)| *len),
                Some(4 * PAGE),
                "contiguous scatter run coalesced into one view"
            );
            assert_eq!(
                phys.views.get(&(at(a, 4) as usize)).map(|(len, _)| *len),
                Some(PAGE)
            );
            assert!(!phys.views.contains_key(&(at(a, 5) as usize)));
        }

        // Write a distinct byte per page and alias all 5 pages into `b`.
        for i in 0..5 {
            // SAFETY: mapped read-write.
            unsafe { at(a, i).cast::<u8>().write((0x10 + i) as u8) };
        }
        assert!(map_alias(b as u64, a as u64, (5 * PAGE) as u64));
        {
            let phys = physical();
            assert_eq!(
                phys.views.get(&(b as usize)).map(|(len, _)| *len),
                Some(4 * PAGE),
                "map_alias coalesced contiguous run into one view"
            );
            assert_eq!(
                phys.views.get(&(at(b, 4) as usize)).map(|(len, _)| *len),
                Some(PAGE)
            );
        }
        for i in 0..5 {
            // SAFETY: aliased read-write.
            assert_eq!(unsafe { at(b, i).cast::<u8>().read() }, (0x10 + i) as u8);
        }

        // Zero-length operations must be no-ops and never panic.
        assert_eq!(XMemMapPhysicalPages(a, 0, std::ptr::null()), a);
        assert_eq!(
            XMemMapPhysicalPagesScatter(addrs.as_ptr(), 0, std::ptr::null()),
            1
        );
        assert!(map_alias(b as u64, 0, 0));
        assert!(map_alias(b as u64, a as u64, 0));
        // Existing mappings are untouched.
        // SAFETY: `b` is still mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 0x10);

        // Two adjacent source views continuing the same section coalesce in `map_alias`.
        assert!(map_alias(b as u64, 0, (5 * PAGE) as u64));
        assert_eq!(XMemMapPhysicalPages(a, 6, std::ptr::null()), a);
        assert_eq!(XMemMapPhysicalPages(a, 2, a_pages[0..2].as_ptr()), a);
        assert_eq!(
            XMemMapPhysicalPages(at(a, 2), 2, a_pages[2..4].as_ptr()),
            at(a, 2)
        );
        assert!(map_alias(b as u64, a as u64, (4 * PAGE) as u64));
        assert_eq!(
            physical().views.get(&(b as usize)).map(|(len, _)| *len),
            Some(4 * PAGE),
            "map_alias coalesced across adjacent views of the same section"
        );

        // Coalesced scatter unmap clears the whole range back to placeholders.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addrs.as_ptr(), 4, std::ptr::null()),
            1
        );
        for i in 0..4 {
            assert_eq!(query(at(a, i)).host.State, MEM_RESERVE);
        }
        assert!(map_alias(b as u64, 0, (4 * PAGE) as u64));

        // SAFETY: our own reservations.
        unsafe {
            assert_ne!(VirtualFree(a, 0, MEM_RELEASE), 0);
            assert_ne!(VirtualFree(b, 0, MEM_RELEASE), 0);
        }
        assert_eq!(XMemFreePhysicalPages(a_pages.len(), a_pages.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(b_pages.len(), b_pages.as_ptr()), 1);
    }

    #[test]
    fn allocated_physical_pages_are_committed() {
        let asked = 4;
        let mut n = asked;
        let mut pages = [0usize; 4];
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert!(n > 0 && n <= asked, "got {n} pages of {asked}");
        let reserve = XMemVirtualAlloc(
            std::ptr::null_mut(),
            n * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!reserve.is_null());
        assert_eq!(XMemMapPhysicalPages(reserve, n, pages.as_ptr()), reserve);
        // The last byte of the last page: committed, not merely reserved.
        let last = reserve.cast::<u8>().wrapping_add(n * PAGE - 1);
        // SAFETY: `n * PAGE` bytes are mapped read-write.
        unsafe {
            last.write(0xa5);
            assert_eq!(last.read(), 0xa5);
        }
        assert_eq!(XMemMapPhysicalPages(reserve, n, std::ptr::null()), reserve);
        assert_eq!(XMemFreePhysicalPages(n, pages.as_ptr()), 1);
        // SAFETY: our own reservation, released by us.
        assert_ne!(unsafe { VirtualFree(reserve, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn commits_across_a_piece_boundary() {
        // One `VirtualAlloc` per piece, so a range that is not a whole number
        // of pieces needs its last, short one committed too; and pages already
        // committed must not be charged twice when the range is committed again.
        let pages = COMMIT_PIECE / PAGE + 2;
        let size = pages * PAGE;
        // SAFETY: plain section creation.
        let handle = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                (size as u64 >> 32) as u32,
                size as u32,
                std::ptr::null(),
            )
        };
        assert!(!handle.is_null());
        let section = Section::new(handle, pages, false).expect("persistent view");
        let base = section.page_ptr(0) as usize;
        assert_eq!(commit_pieces(base, size), size);
        assert_eq!(
            commit_pieces(base, size),
            size,
            "committed pages commit again"
        );
        assert!(section.ensure_committed(0, pages));
        assert!(section.all_committed(0, pages));
    }

    #[test]
    fn section_views_tracks_maps_aliases_splits_and_frees() {
        let pages = allocate(4);
        let section_key = Arc::as_ptr(&physical().pages[&pages[0]].section) as usize;
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                4 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE) as usize;

        assert_eq!(XMemMapPhysicalPages(a, 4, pages.as_ptr()), a);
        {
            let phys = physical();
            assert_eq!(
                phys.section_views.get(&section_key).map(Vec::as_slice),
                Some([(a as usize, 0, 4)].as_slice())
            );
        }

        // Alias pages 1..3 at `b`.
        assert!(map_alias(b as u64, at(a, 1) as u64, (2 * PAGE) as u64));
        {
            let phys = physical();
            let mut entries = phys.section_views[&section_key].clone();
            entries.sort_unstable();
            assert_eq!(entries, vec![(a as usize, 0, 4), (b as usize, 1, 2)]);
        }

        // Partial unmap of page 1 at `a` splits `[a, a+4P)` into `[a, 0, 1)` and `[a+2P, 2, 2)`.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1) as *mut c_void, 1, std::ptr::null()),
            at(a, 1) as *mut c_void
        );
        {
            let phys = physical();
            let mut entries = phys.section_views[&section_key].clone();
            entries.sort_unstable();
            assert_eq!(
                entries,
                vec![(a as usize, 0, 1), (at(a, 2), 2, 2), (b as usize, 1, 2)]
            );
        }

        // Freeing the physical page numbers leaves mapped views alive until unmapped.
        assert_eq!(XMemFreePhysicalPages(pages.len(), pages.as_ptr()), 1);
        assert!(physical().section_views.contains_key(&section_key));

        assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
        assert!(map_alias(b as u64, 0, (2 * PAGE) as u64));
        assert!(!physical().section_views.contains_key(&section_key));

        // SAFETY: our own reservations.
        unsafe {
            assert_ne!(VirtualFree(a, 0, MEM_RELEASE), 0);
            assert_ne!(VirtualFree(b, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn mapped_write_ranges_match_exact_overlapping_bytes() {
        let (a_pages, b_pages) = (allocate(4), allocate(2));
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                4 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE) as usize;

        // Map only `a_pages[1]` at `a + PAGE` and `b_pages[0]` at `a + 2*PAGE`,
        // leaving `a_pages[0]` and `a_pages[2]` unmapped in `a`.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1) as *mut c_void, 1, a_pages[1..2].as_ptr()),
            at(a, 1) as *mut c_void
        );
        assert_eq!(
            XMemMapPhysicalPages(at(a, 2) as *mut c_void, 1, b_pages[0..1].as_ptr()),
            at(a, 2) as *mut c_void
        );
        // Also alias `a_pages[1]` at `b`.
        assert!(map_alias(b as u64, at(a, 1) as u64, PAGE as u64));

        // Write across `[a_pages[0], a_pages[1], b_pages[0]]` starting 100 bytes
        // into `a_pages[0]` for `2 * PAGE + 25` bytes:
        // - `a_pages[0]` is unmapped -> none
        // - `a_pages[1]` is wholly written -> `[at(a, 1), PAGE)` and `[b, PAGE)`
        // - `b_pages[0]` has `[0, 125)` written -> `[at(a, 2), 125)`
        // Notice `at(a, 1)..at(a, 1)+PAGE` and `at(a, 2)..at(a, 2)+125` are contiguous and coalesce!
        let write_pages = [a_pages[0], a_pages[1], b_pages[0]];
        let total = 2 * PAGE + 25;
        let run_list = runs(&write_pages, 100, total).unwrap();
        let ranges = mapped_ranges_for(&run_list, 100, total);
        assert_eq!(ranges, vec![(at(a, 1), PAGE + 125), (b as usize, PAGE)]);

        // `page_window` on `a_pages[1..3]` with offset 200, len 500 only covers `a_pages[1]`.
        let mut window = page_window(&a_pages[1..3], 200, 500).expect("one run");
        assert_eq!(
            mapped_ranges_for_write(&window.write),
            vec![(at(a, 1) + 200, 500), (b as usize + 200, 500)]
        );
        // Map `a_pages[2]` while `window` is alive; re-querying (as `Drop` does) sees it too
        // if the window extends into page 2.
        let mut spanning = page_window(&a_pages[1..3], PAGE - 50, 200).expect("one run");
        assert_eq!(
            mapped_ranges_for_write(&spanning.write),
            vec![(at(a, 1) + PAGE - 50, 50), (b as usize + PAGE - 50, 50)]
        );
        assert_eq!(
            XMemMapPhysicalPages(at(a, 3) as *mut c_void, 1, a_pages[2..3].as_ptr()),
            at(a, 3) as *mut c_void
        );
        assert_eq!(
            mapped_ranges_for_write(&spanning.write),
            vec![
                (at(a, 1) + PAGE - 50, 50),
                (at(a, 3), 150),
                (b as usize + PAGE - 50, 50),
            ]
        );
        spanning.bytes().fill(0x42);
        drop(spanning);
        window.bytes().fill(0x24);
        drop(window);
        let payload = vec![0x55; total];
        assert!(write_physical(&write_pages, 100, &payload));

        assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
        assert_eq!(XMemMapPhysicalPages(b, 4, std::ptr::null()), b);
        // SAFETY: our own reservations.
        unsafe {
            assert_ne!(VirtualFree(a, 0, MEM_RELEASE), 0);
            assert_ne!(VirtualFree(b, 0, MEM_RELEASE), 0);
        }
        assert_eq!(XMemFreePhysicalPages(a_pages.len(), a_pages.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(b_pages.len(), b_pages.as_ptr()), 1);
    }

    #[test]
    fn concurrent_access_survives_whole_view_replacement() {
        use std::sync::atomic::AtomicBool;

        let pages_a = allocate(1);
        let pages_b = allocate(1);
        let pages_c = allocate(1);
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let target = reserve();
        let alias_src = reserve();
        assert!(!target.is_null() && !alias_src.is_null());

        let write_sentinel = |page: usize, val: u32| {
            let mut bytes = [0u8; PAGE];
            bytes[..4].copy_from_slice(&val.to_ne_bytes());
            bytes[PAGE - 4..].copy_from_slice(&val.to_ne_bytes());
            assert!(write_physical(&[page], 0, &bytes));
        };
        const VAL_A: u32 = 0x1111_2222;
        const VAL_B: u32 = 0x3333_4444;
        const VAL_C: u32 = 0x5555_6666;
        write_sentinel(pages_a[0], VAL_A);
        write_sentinel(pages_b[0], VAL_B);
        write_sentinel(pages_c[0], VAL_C);

        assert_eq!(XMemMapPhysicalPages(target, 1, pages_a.as_ptr()), target);
        assert_eq!(
            XMemMapPhysicalPages(alias_src, 1, pages_c.as_ptr()),
            alias_src
        );

        let started = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let target_addr = target as usize;
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let started = started.clone();
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let mut iters = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        // SAFETY: `target_addr` is continuously replaced with committed
                        // 64 KiB physical views while `RemapGuard` serializes faults.
                        unsafe {
                            let head = (target_addr as *const u32).read_volatile();
                            let tail = ((target_addr + PAGE - 4) as *const u32).read_volatile();
                            assert!(
                                head == VAL_A || head == VAL_B || head == VAL_C,
                                "unexpected head {head:#x}"
                            );
                            assert!(
                                tail == VAL_A || tail == VAL_B || tail == VAL_C,
                                "unexpected tail {tail:#x}"
                            );
                        }
                        if iters == 0 {
                            started.fetch_add(1, Ordering::Release);
                        }
                        iters += 1;
                    }
                    iters
                })
            })
            .collect();

        while started.load(Ordering::Acquire) < readers.len() {
            std::thread::yield_now();
        }

        for round in 0..200u32 {
            match round % 3 {
                0 => assert_eq!(XMemMapPhysicalPages(target, 1, pages_b.as_ptr()), target),
                1 => assert!(map_alias(target as u64, alias_src as u64, PAGE as u64)),
                _ => assert_eq!(XMemMapPhysicalPages(target, 1, pages_a.as_ptr()), target),
            }
        }

        stop.store(true, Ordering::Release);
        for reader in readers {
            let iters = reader
                .join()
                .expect("reader must survive whole-view replacement without access violations");
            assert!(iters > 0);
        }

        assert_eq!(XMemMapPhysicalPages(target, 1, std::ptr::null()), target);
        assert_eq!(
            XMemMapPhysicalPages(alias_src, 1, std::ptr::null()),
            alias_src
        );
        // SAFETY: our own reservations.
        unsafe {
            assert_ne!(VirtualFree(target, 0, MEM_RELEASE), 0);
            assert_ne!(VirtualFree(alias_src, 0, MEM_RELEASE), 0);
        }
        assert_eq!(XMemFreePhysicalPages(1, pages_a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, pages_b.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, pages_c.as_ptr()), 1);
    }
}

//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! unmapped are not asked about again until it maps them: view memory
//! comes from placeholders, which only the runtime's mapping fills.
//! shortcut: so memory committed otherwise into such a block (a range
//! reaching from a view into a plain reservation) is not seen.
//!
//! shortcut: writes into physical pages through another view (xmem's
//! `write_physical`, DirectStorage's page destinations) are not seen at
//! the guarded addresses; they go to the texture pool, whose textures are
//! hashed, not guarded. A system call writing into a guarded block fails
//! (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
});

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // `JOURNAL` is taken before `GUARDS` here as in `guard`; nothing takes
    // them the other way round.
    let known_view_pages = if guarded && first < last {
        let mut g = guards();
        // Guarded clean blocks + unmapped placeholder blocks: zero syscalls, answer from journal.
        let clean_or_placeholder = g.clean.range(first..last).count()
            + g.placeholder_absent.range(first..last).count();
        if clean_or_placeholder == last - first {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::WatchCleanHits, 1);
            j.asked += 1;
            return Some((
                j.asked,
                j.written
                    .range(first..last)
                    .map(|(&p, &w)| (p, w))
                    .collect(),
            ));
        }
        // Known view region (including split physical views): skip GetWriteWatch +
        // VirtualQuery and guard/update directly under the lock.
        if g.is_view(first) {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            let pages = guard_locked(&mut g, first, last);
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                let us = started.elapsed().as_micros() as u64;
                add(C::Watches, 1);
                add(C::WatchUs, us);
                add(C::WatchKnownView, 1);
                add(C::WatchKnownViewUs, us);
            }
            Some(pages)
        } else {
            None
        }
    } else {
        None
    };
    let (count, guarded_pages) = if let Some(pages) = known_view_pages {
        (0, pages)
    } else {
        let mut count = (last - first) * PAGE / KERNEL_PAGE;
        j.found.resize(count.max(j.found.len()), 0);
        let mut granularity = 0;
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(test)]
        tests::KERNEL_ASKED.with(|n| n.set(n.get() + 1));
        // SAFETY: the answer buffer holds `count` entries.
        let failed = unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                j.found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
        let guarded_pages = if !failed {
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add, direct_ok_range};
                let us = started.elapsed().as_micros() as u64;
                let len = (last - first) * PAGE;
                add(C::WatchDirectOk, 1);
                add(C::WatchDirectOkUs, us);
                add(C::WatchDirectBytes, len as u64);
                direct_ok_range(first * PAGE, len, us);
            }
            Vec::new()
        } else {
            count = 0;
            #[cfg(feature = "diagnostics")]
            let phase_started = std::time::Instant::now();
            match by_allocation(first * PAGE, last * PAGE, &mut j.found) {
                Some(pages) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        use crate::copystat::{C, add};
                        add(C::WatchByAlloc, 1);
                        add(C::WatchByAllocUs, phase_started.elapsed().as_micros() as u64);
                    }
                    pages
                }
                None if !guarded => return None,
                None => {
                    let pages = guard(first * PAGE, last * PAGE)?;
                    #[cfg(feature = "diagnostics")]
                    {
                        use crate::copystat::{C, add};
                        add(C::WatchGuardFallback, 1);
                        add(C::WatchGuardFallbackUs, phase_started.elapsed().as_micros() as u64);
                    }
                    pages
                }
            }
        };
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::WatchUs, started.elapsed().as_micros() as u64);
        }
        (count, guarded_pages)
    };
    j.asked += 1;
    let now = j.asked;
    for &at in &j.found[..count] {
        j.written.insert(at / PAGE, now);
    }
    for page in guarded_pages {
        j.written.insert(page, now);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// The pages of `[start, end)` written, asked allocation by allocation:
/// the kernel refuses a range over several (a title buffer spans as many
/// allocations as the runtime made for it). Parts never committed cannot
/// have been written, so a range none of which is committed (a reserved
/// allocation the title has not committed yet) has no writes. `None` if a
/// committed part is not write-watched (or the range is one committed
/// allocation, already refused).
fn by_allocation(start: usize, end: usize, found: &mut Vec<usize>) -> Option<Vec<usize>> {
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEM_COMMIT, MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let query = |at: usize| {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query about an address; nothing is dereferenced.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        (n != 0).then_some(info)
    };
    let (mut pages, mut at, mut allocations, mut any) = (Vec::new(), start, 0, false);
    while at < end {
        let info = query(at)?;
        let base = info.AllocationBase as usize;
        let mut committed = info.State == MEM_COMMIT;
        let mut to = (info.BaseAddress as usize + info.RegionSize).min(end);
        // The rest of the allocation's regions in the range.
        while info.State != MEM_FREE && to < end {
            let next = query(to)?;
            if next.State == MEM_FREE || next.AllocationBase as usize != base {
                break;
            }
            committed |= next.State == MEM_COMMIT;
            to = (next.BaseAddress as usize + next.RegionSize).min(end);
        }
        allocations += 1;
        any |= committed;
        if committed {
            let mut count = (to - at) / KERNEL_PAGE;
            found.resize(count.max(found.len()), 0);
            let mut granularity = 0;
            // SAFETY: the answer buffer holds `count` entries.
            let failed = unsafe {
                GetWriteWatch(
                    WRITE_WATCH_FLAG_RESET,
                    at as *const _,
                    to - at,
                    found.as_mut_ptr().cast(),
                    &mut count,
                    &mut granularity,
                )
            } != 0;
            if failed {
                return None;
            }
            pages.extend(found[..count].iter().map(|&a| a / PAGE));
        }
        at = to;
    }
    (allocations > 1 || !any).then_some(pages)
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;
// `writes` reads blocks as pages.
const _: () = assert!(BLOCK == PAGE);

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
    /// Blocks found unmapped inside a `MEM_MAPPED` view (`SEC_RESERVE`): may
    /// commit without a runtime mapping call, so checked via `VirtualQuery`.
    absent: BTreeSet<usize>,
    /// Unmapped placeholder / free blocks (`Type != MEM_MAPPED`) that only a
    /// runtime mapping (`title_memory_remapped`) can populate; skipped with
    /// zero `VirtualQuery` syscalls until remapped.
    placeholder_absent: BTreeSet<usize>,
    /// Blocks found to be in a view when an asking began there, until the
    /// runtime remaps them. Asking the system each time (`VirtualQuery`)
    /// cost 100-230 ms per world frame, more than the rest of the guard.
    views: BTreeSet<usize>,
    /// Disjoint mapped-view block ranges `[start_block, end_block)` discovered
    /// via `VirtualQuery`, until `title_memory_remapped` splits/clears them.
    view_ranges: BTreeMap<usize, usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
    absent: BTreeSet::new(),
    placeholder_absent: BTreeSet::new(),
    views: BTreeSet::new(),
    view_ranges: BTreeMap::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Guards {
    fn is_view(&self, block: usize) -> bool {
        self.views.contains(&block) || self.is_view_range(block, block + 1)
    }

    fn is_view_range(&self, first: usize, last: usize) -> bool {
        first < last
            && self
                .view_ranges
                .range(..=first)
                .next_back()
                .is_some_and(|(&s, &e)| s <= first && last <= e)
    }

    fn record_view_region(&mut self, mut first: usize, mut last: usize) {
        let mut to_remove = Vec::new();
        for (&s, &e) in self.view_ranges.range(..=last) {
            if e >= first {
                first = first.min(s);
                last = last.max(e);
                to_remove.push(s);
            }
        }
        for s in to_remove {
            self.view_ranges.remove(&s);
        }
        self.view_ranges.insert(first, last);
    }

    fn remove_view_range(&mut self, r_start: usize, r_end: usize) {
        let overlapping: Vec<(usize, usize)> = self
            .view_ranges
            .range(..r_end)
            .filter(|&(_, &e)| e > r_start)
            .map(|(&s, &e)| (s, e))
            .collect();
        for (s, e) in overlapping {
            self.view_ranges.remove(&s);
            if s < r_start {
                self.view_ranges.insert(s, r_start);
            }
            if e > r_end {
                self.view_ranges.insert(r_end, e);
            }
        }
    }
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    install();
    let mut g = guards();
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // Asked under the lock: a remap in between would leave a stale view.
    if !g.views.contains(&first) && !g.is_view_range(first, (first + 1).min(last)) {
        let (v_start, v_end) = view_region(start)?;
        g.views.insert(first);
        g.record_view_region(v_start, v_end);
    }
    Some(guard_locked(&mut g, first, last))
}

fn guard_locked(g: &mut Guards, first: usize, last: usize) -> Vec<usize> {
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let mut uncommitted_until = first;
    for block in first..=last {
        let mut open = block < last
            && !g.clean.contains(&block)
            && !g.placeholder_absent.contains(&block);
        if open && g.absent.contains(&block) {
            // Ordinary commits inside a `MEM_MAPPED` view need no runtime mapping
            // notification. Reuse this asking's query for following blocks.
            if block >= uncommitted_until {
                (uncommitted_until, _) = uncommitted_end(block);
            }
            open = block >= uncommitted_until;
            if open {
                g.absent.remove(&block);
            }
        }
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(g, from, block);
        }
    }
    written
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting unmapped placeholder blocks separately from uncommitted `SEC_RESERVE` blocks.
fn protect(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        } else {
            let (end, is_placeholder) = uncommitted_end(block);
            if block < end {
                if is_placeholder {
                    g.placeholder_absent.insert(block);
                } else {
                    g.absent.insert(block);
                }
            }
        }
    }
}

/// Returns `(end_block, is_placeholder)` for uncommitted blocks starting at `block`.
/// `is_placeholder` is true for `MEM_FREE` / `MEM_PRIVATE` placeholders that only
/// `title_memory_remapped` can fill; `SEC_RESERVE` views keep `is_placeholder = false`.
fn uncommitted_end(block: usize) -> (usize, bool) {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_MAPPED, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        let end = (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK;
        (end, info.Type != MEM_MAPPED)
    } else {
        (block, false)
    }
}

/// If `at` lies in a view (of a section), returns the view region's
/// `[start_block, end_block)` span.
fn view_region(at: usize) -> Option<(usize, usize)> {
    use windows_sys::Win32::System::Memory::{MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery};
    #[cfg(test)]
    tests::QUERIED.with(|n| n.set(n.get() + 1));
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && info.Type == MEM_MAPPED {
        let base = info.BaseAddress as usize;
        let end = base.saturating_add(info.RegionSize);
        let first = base / BLOCK;
        let last = end / BLOCK;
        if first < last {
            return Some((first, last));
        }
    }
    None
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        if unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) } == 0 {
            tracing::warn!(
                block = format_args!("{:#x}", block * BLOCK),
                error = %std::io::Error::last_os_error(),
                "watch::open VirtualProtect failed"
            );
        }
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let mut g = guards();
    let range = at / BLOCK..end.div_ceil(BLOCK);
    let absent: Vec<usize> = g.absent.range(range.clone()).copied().collect();
    for block in absent {
        g.absent.remove(&block);
    }
    let placeholder: Vec<usize> = g.placeholder_absent.range(range).copied().collect();
    for block in placeholder {
        g.placeholder_absent.remove(&block);
    }
    open(&mut g, at, end);
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped), and any placed textures backed by that range are evicted.
pub fn title_memory_remapped(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    crate::host::forget_unreadable(at, len);
    let end = at.saturating_add(len);
    {
        let mut g = guards();
        let range = at / BLOCK..end.div_ceil(BLOCK);
        let absent: Vec<usize> = g.absent.range(range.clone()).copied().collect();
        for block in absent {
            g.absent.remove(&block);
        }
        let placeholder: Vec<usize> = g.placeholder_absent.range(range.clone()).copied().collect();
        for block in placeholder {
            g.placeholder_absent.remove(&block);
        }
        let views: Vec<usize> = g.views.range(range.clone()).copied().collect();
        for block in views {
            g.views.remove(&block);
        }
        g.remove_view_range(range.start, range.end);
        open(&mut g, at, end);
    }
    crate::resource::evict_placed_textures(at as u64, end as u64);
    crate::descriptor::host::bindless_memory_remapped(at as u64, end as u64);
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
pub(crate) use tests::forget_journal;

#[cfg(test)]
pub(crate) fn kernel_asked() -> usize {
    tests::kernel_asked()
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    thread_local! {
        /// The kernel askings `writes` made on this thread (each test runs
        /// on its own).
        pub(super) static KERNEL_ASKED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// The view queries `guard` made on this thread (each test runs on
        /// its own).
        pub(super) static QUERIED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    pub(super) fn kernel_asked() -> usize {
        KERNEL_ASKED.with(|n| n.get())
    }

    fn queried() -> usize {
        QUERIED.with(|n| n.get())
    }

    /// A pagefile section of `blocks` blocks, mapped read-write.
    fn view(blocks: usize) -> usize {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        // SAFETY: a fresh section; the view is ours.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (blocks * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, blocks * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        view
    }

    pub fn forget_journal(start: usize, len: usize) {
        let pages = start / PAGE..(start + len).div_ceil(PAGE);
        JOURNAL
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .written
            .retain(|p, _| !pages.contains(p));
    }

    /// Unmaps a view from [`view`], first forgetting its pages' journal
    /// entries: the next allocation there is not written.
    fn unmap(view: usize, blocks: usize) {
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        forget_journal(view, blocks * BLOCK);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, blocks * BLOCK);
    }

    /// A clean guarded view is answered from the journal, without the
    /// kernel; a write to it (by the title, or announced) is still seen.
    #[test]
    fn clean_guarded_views_are_answered_from_the_journal() {
        let v = view(4);
        let (start, end) = (v, v + 4 * BLOCK);
        let first = v / PAGE;
        let asked = kernel_asked();
        // The first asking guards every block: all are written as far as
        // anyone knows.
        let (taken, all) = writes(start, end, true).expect("a view");
        assert_eq!(kernel_asked(), asked + 1);
        assert_eq!(all, (first..first + 4).map(|p| (p, taken)).collect::<Vec<_>>());
        // Clean since: the kernel is not asked, the journal answers as before.
        for _ in 0..3 {
            let (now, same) = writes(start, end, true).expect("a view");
            assert!(now > taken);
            assert_eq!(same, all);
        }
        let (_, part) = writes(v + BLOCK + 10, v + 2 * BLOCK - 10, true).expect("a view");
        assert_eq!(part, [(first + 1, taken)]);
        assert_eq!(kernel_asked(), asked + 1);

        // The title writes block 2: the handler opens it; because `v..v+4*BLOCK`
        // is already a known mapped view region, `writes` skips `GetWriteWatch`.
        // SAFETY: our view, guarded read-only; the handler gives write access.
        unsafe { ((v + 2 * BLOCK + 8) as *mut u8).write(7) };
        let (written, after) = writes(start, end, true).expect("a view");
        assert_eq!(kernel_asked(), asked + 1);
        assert_eq!(after[2], (first + 2, written));
        assert_eq!(after.iter().filter(|&&(_, w)| w == written).count(), 1);
        // SAFETY: as above, readable.
        assert_eq!(unsafe { ((v + 2 * BLOCK + 8) as *const u8).read() }, 7);
        // Guarded again, and clean.
        let (_, again) = writes(start, end, true).expect("a view");
        assert_eq!(again, after);
        assert_eq!(kernel_asked(), asked + 1);

        // Our own write through the kernel, announced.
        writing(v + 3 * BLOCK, 1);
        let (announced, after) = writes(start, end, true).expect("a view");
        assert_eq!(kernel_asked(), asked + 1);
        assert_eq!(after[3], (first + 3, announced));

        // The runtime remapped block 0 (clearing it from `view_ranges`): asked again.
        title_memory_remapped(v, BLOCK);
        let (remapped, after) = writes(start, end, true).expect("a view");
        assert_eq!(kernel_asked(), asked + 2);
        assert_eq!(after[0], (first, remapped));
        let (_, settled) = writes(start, end, true).expect("a view");
        assert_eq!(settled, after);
        assert_eq!(kernel_asked(), asked + 2);

        // An empty range is not answered from the guards.
        let _ = writes(v, v, true);
        assert_eq!(kernel_asked(), asked + 3);
        unmap(v, 4);
    }

    /// Expanding within the same mapped view region reuses the known region
    /// without extra `GetWriteWatch` calls; unguarded askers still query.
    #[test]
    fn partly_clean_ranges_ask_the_kernel() {
        let v = view(2);
        let asked = kernel_asked();
        // Only block 0 is guarded, discovering the whole 2-block view region.
        let _ = writes(v, v + BLOCK, true).expect("a view");
        assert_eq!(kernel_asked(), asked + 1);
        let (now, both) = writes(v, v + 2 * BLOCK, true).expect("a view");
        assert_eq!(kernel_asked(), asked + 1);
        assert_eq!(both.last(), Some(&(v / PAGE + 1, now)));
        // Both guarded and clean now.
        let _ = writes(v, v + 2 * BLOCK, true).expect("a view");
        assert_eq!(kernel_asked(), asked + 1);
        // Not guarded askers (texture hashes) never take the journal's word.
        assert!(writes(v, v + 2 * BLOCK, false).is_none());
        assert_eq!(kernel_asked(), asked + 2);
        unmap(v, 2);
    }

    /// A view region is queried once across all its blocks until the runtime
    /// remaps an overlapping block; other memory every time.
    #[test]
    fn views_are_queried_once_until_remapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        let at = queried();
        assert!(guard(view, view + 2 * BLOCK).is_some());
        assert!(guard(view + 8, view + BLOCK).is_some());
        assert_eq!(queried(), at + 1);
        // Another block in the same mapped region is already known.
        assert!(guard(view + BLOCK, view + 2 * BLOCK).is_some());
        assert!(guard(view + BLOCK, view + 2 * BLOCK).is_some());
        assert_eq!(queried(), at + 1);
        // The runtime remapped the first block: asked again.
        title_memory_remapped(view, BLOCK);
        assert!(guard(view, view + 2 * BLOCK).is_some());
        assert_eq!(queried(), at + 2);
        // Private memory is not a view, however often asked.
        let plain = vec![0u8; 2 * BLOCK];
        let p = plain.as_ptr() as usize;
        assert!(guard(p, p + 16).is_none());
        assert!(guard(p, p + 16).is_none());
        assert_eq!(queried(), at + 4);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            })
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn known_view_dirty_subranges_skip_kernel_asks() {
        let v = view(4);
        let asked = kernel_asked();
        let _ = writes(v, v + BLOCK, true).expect("a view");
        assert_eq!(kernel_asked(), asked + 1);
        // Dirtied subranges inside the same 4-block mapped view use zero kernel
        // write-watch asks while accurately reporting the written block.
        // SAFETY: `v + 2 * BLOCK` is mapped and our VEH opens the guarded block on write.
        unsafe { ((v + 2 * BLOCK + 16) as *mut u8).write(0x5a) };
        let (t1, found1) = writes(v + BLOCK, v + 3 * BLOCK, true).expect("a view");
        assert_eq!(kernel_asked(), asked + 1);
        assert!(found1.iter().any(|&(p, w)| p == (v / PAGE + 2) && w == t1));
        unmap(v, 4);
    }

    #[test]
    fn placeholder_backed_split_view_hits_clean_fast_path_without_syscalls() {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS,
            UnmapViewOfFile2, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        // SAFETY: fresh 4-block placeholder with 2 non-adjacent 1-block mapped sections.
        let (base, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                4 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let s = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!s.is_null());
            (p, s)
        };
        let map_block = |blk: usize, sec_idx: u64| {
            // SAFETY: splitting placeholder and mapping section block inside `base..base + 4*BLOCK`.
            unsafe {
                VirtualFree(
                    (base + blk * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                let v = MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (base + blk * BLOCK) as *const _,
                    sec_idx * BLOCK as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                assert_eq!(v.Value as usize, base + blk * BLOCK);
                v
            }
        };
        let v0 = map_block(0, 0);
        let v2 = map_block(2, 1);

        let asked = kernel_asked();
        let q = queried();
        // First pass: probes once, guards mapped blocks 0 & 2, notes 1 & 3 in placeholder_absent.
        let (t0, _) = writes(base, base + 4 * BLOCK, true).expect("placeholder split view");
        assert_eq!(kernel_asked(), asked + 1);
        let q_after_init = queried();
        assert!(q_after_init > q);

        // Repeat passes: both clean blocks and placeholder_absent blocks -> zero syscalls,
        // and no pages gain a newer write timestamp than `t0`.
        for _ in 0..4 {
            let (t, w) = writes(base, base + 4 * BLOCK, true).expect("fast path");
            assert!(t > t0);
            assert!(w.iter().all(|&(_, when)| when <= t0));
        }
        assert_eq!(kernel_asked(), asked + 1);
        assert_eq!(queried(), q_after_init);

        // Clean up mapped views & placeholder.
        // SAFETY: unmapping and freeing views/placeholders allocated above.
        // SAFETY: unmapping and freeing views/placeholders allocated above.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), v0, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), v2, MEM_PRESERVE_PLACEHOLDER);
            for blk in 0..4 {
                VirtualFree((base + blk * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            CloseHandle(section);
        }
        forget_journal(base, 4 * BLOCK);
        title_memory_remapped(base, 4 * BLOCK);
    }

    #[test]
    fn split_adjacent_views_skip_kernel_asks_when_dirty() {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS,
            UnmapViewOfFile2, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        // SAFETY: 2 adjacent independently-mapped views (2 VirtualQuery regions).
        let (base, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                2 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let s = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!s.is_null());
            (p, s)
        };
        let map_block = |blk: usize| {
            // SAFETY: splitting placeholder and mapping section block inside `base..base + 2*BLOCK`.
            unsafe {
                VirtualFree(
                    (base + blk * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                let v = MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (base + blk * BLOCK) as *const _,
                    (blk * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                assert_eq!(v.Value as usize, base + blk * BLOCK);
                v
            }
        };
        let v0 = map_block(0);
        let v1 = map_block(1);
        let asked = kernel_asked();
        let _ = writes(base, base + 2 * BLOCK, true).expect("split view");
        assert_eq!(kernel_asked(), asked + 1);
        // SAFETY: block 1 is mapped; our VEH opens the guarded block on write.
        unsafe { ((base + BLOCK + 8) as *mut u8).write(0x7c) };
        let (t, found) = writes(base, base + 2 * BLOCK, true).expect("dirty split view");
        assert_eq!(kernel_asked(), asked + 1, "g.is_view(first) skips GetWriteWatch");
        assert!(found.iter().any(|&(p, w)| p == base / PAGE + 1 && w == t));

        // SAFETY: unmapping and freeing views/placeholders allocated above.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), v0, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), v1, MEM_PRESERVE_PLACEHOLDER);
            VirtualFree(base as *mut _, 0, MEM_RELEASE);
            VirtualFree((base + BLOCK) as *mut _, 0, MEM_RELEASE);
            CloseHandle(section);
        }
        forget_journal(base, 2 * BLOCK);
        title_memory_remapped(base, 2 * BLOCK);
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        forget_journal(start, 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        forget_journal(start, 4 * PAGE);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// Two write-watched allocations side by side (the runtime reserves
    /// title memory allocation by allocation), asked about as one range, as
    /// a buffer spanning them is: the pages written in each are reported.
    #[test]
    fn a_range_over_several_watched_allocations_is_watched() {
        let (a, b) = loop {
            // SAFETY: a probe reservation, released at once to find room for two.
            let span =
                unsafe { VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE, PAGE_READWRITE) };
            assert!(!span.is_null());
            // SAFETY: the probe above.
            unsafe { VirtualFree(span, 0, MEM_RELEASE) };
            let at = |p: usize| {
                // SAFETY: a fixed address in the room just released.
                let got = unsafe {
                    VirtualAlloc(
                        p as *const _,
                        2 * PAGE,
                        MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                        PAGE_READWRITE,
                    )
                };
                got as usize
            };
            let a = at(span as usize);
            let b = at(span as usize + 2 * PAGE);
            if a != 0 && b != 0 {
                break (a, b);
            }
            for p in [a, b].into_iter().filter(|&p| p != 0) {
                // SAFETY: ours.
                unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
            }
        };
        forget_journal(a, 2 * PAGE);
        forget_journal(b, 2 * PAGE);
        let _ = writes(a, b + 2 * PAGE, false);
        // SAFETY: committed pages of ours.
        unsafe {
            ((a + 8) as *mut u8).write(1);
            ((b + PAGE + 8) as *mut u8).write(1);
        }
        let (now, found) = writes(a, b + 2 * PAGE, false).expect("watched");
        assert_eq!(found, [(a / PAGE, now), (b / PAGE + 1, now)]);
        // SAFETY: ours.
        unsafe {
            VirtualFree(a as *mut _, 0, MEM_RELEASE);
            VirtualFree(b as *mut _, 0, MEM_RELEASE);
        }
        forget_journal(a, 2 * PAGE);
        forget_journal(b, 2 * PAGE);
    }

    /// A range in an allocation reserved but not committed yet (as the
    /// title's are before it uses them) has no writes, and is asked again
    /// once committed.
    #[test]
    fn a_range_never_committed_has_no_writes() {
        // SAFETY: a fresh write-watched reservation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                2 * PAGE,
                MEM_RESERVE | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        forget_journal(p, 2 * PAGE);
        let (_, none) = writes(p, p + 2 * PAGE, false).expect("nothing can be written there");
        assert!(none.is_empty());
        // SAFETY: committing and writing a page of our reservation.
        unsafe {
            assert!(!VirtualAlloc(p as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            ((p + 8) as *mut u8).write(1);
        }
        let (now, found) = writes(p, p + 2 * PAGE, false).expect("watched");
        assert_eq!(found, [(p / PAGE, now)]);
        // SAFETY: ours.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        forget_journal(p, 2 * PAGE);
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn remap_restores_write_access_after_guard() {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of one block, mapped read-write.
        let (section, view) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                BLOCK as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK).Value as usize;
            (section, view)
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + BLOCK).is_some());
        assert!(!writable(view), "guard must make view read-only");
        title_memory_remapped(view, BLOCK);
        assert!(writable(view), "remap must restore PAGE_READWRITE");
        assert!(!guards().protected.contains(&(view / BLOCK)));
        // SAFETY: writable again after remap; write must succeed without faulting.
        unsafe {
            (view as *mut u8).write_volatile(0x42);
            assert_eq!((view as *const u8).read_volatile(), 0x42);
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            });
            CloseHandle(section);
        }
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    #[test]
    fn writing_clears_absent_and_opens_guarded_blocks() {
        let v = view(2);
        assert!(guard(v, v + 2 * BLOCK).is_some());
        let first = v / BLOCK;
        {
            let mut g = guards();
            assert!(g.clean.contains(&first));
            g.absent.insert(first + 1);
            g.clean.remove(&(first + 1));
            g.protected.remove(&(first + 1));
        }
        // Writing to both blocks wakes up guarded `first` and absent `first + 1`.
        writing(v + 128, BLOCK);
        {
            let g = guards();
            assert!(!g.clean.contains(&first), "clean cleared");
            assert!(!g.protected.contains(&first), "protection opened");
            assert!(!g.absent.contains(&(first + 1)), "absent cleared");
        }
        forget_journal(v, 2 * BLOCK);
        title_memory_remapped(v, 2 * BLOCK);
    }
}

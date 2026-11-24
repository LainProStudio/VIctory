//! Opt-in observation only. A fixed request has a finite set of declared table
//! slots. It still needs its bound table bases, relocations and view generations.
use crate::host::HostList;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use windows::core::Interface;

use victory_shader::resource::{Address, Resources, Role, Source};
use victory_shader::{Container, FourCC, HwStage, ResourceType, RootParam, RootType, UsedResource};

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_BINDLESS_ATTRIBUTION").as_deref() == Ok("1"))
}

// A dynamic access in any stage takes precedence over an unclassified stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum Access {
    Fixed,
    Unknown,
    Dynamic,
}

pub(crate) fn classify_container(bytes: &[u8]) -> Access {
    fn inspect(bytes: &[u8]) -> Option<Access> {
        let container = Container::parse(bytes).ok()?;
        // Follow the same native program, patched instructions and spill mode as emit::translate_with.
        let native = victory_shader::native_programs(container.part(FourCC::DXIL)?.data)
            .ok()?
            .into_iter()
            .find(|p| {
                matches!(
                    p.stage,
                    HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms
                )
            })?;
        let spilled = native.stage == HwStage::Cs
            && !native.layout.is_empty()
            && native
                .layout
                .iter()
                .all(|r| native.stage.user_sgpr(r.user_data) >= 16);
        let ir = victory_shader::ir::build_with(
            &native.patched_code(),
            native.wave64,
            native.vgprs,
            spilled,
        )
        .ok()?;
        let accesses = victory_shader::resource::track(&ir, &native.layout, native.stage).ok()?;
        Some(classify(&accesses, &native.layout, &native.used))
    }
    inspect(bytes).unwrap_or(Access::Unknown)
}

fn classify(accesses: &Resources, layout: &[RootParam], used: &[UsedResource]) -> Access {
    let addresses: HashMap<_, _> = accesses.scalar.iter().copied().collect();
    let leaf = |source: &Source| match source {
        Source::Built { address, .. }
            if layout
                .get(address.param)
                .is_some_and(|r| r.root == RootType::Descriptor) =>
        {
            Access::Fixed
        }
        Source::Loaded { load, first } => addresses
            .get(load)
            .map_or(Access::Unknown, |a| table_access(a, *first, layout, used)),
        Source::Fetched { load, .. }
            if addresses.get(load).is_some_and(|a| {
                a.dynamic.is_some()
                    && layout
                        .get(a.param)
                        .is_some_and(|r| matches!(r.root, RootType::ResourceHeap | RootType::Table))
            }) =>
        {
            Access::Dynamic
        }
        _ => Access::Unknown,
    };
    // Selection between fixed leaves is bounded. A single unresolved leaf forces fallback.
    accesses
        .descriptors
        .iter()
        .filter(|d| d.role == Role::Resource)
        .flat_map(|d| d.source.leaves())
        .map(leaf)
        .max()
        .unwrap_or(Access::Unknown)
}

fn table_access(a: &Address, first: u8, layout: &[RootParam], used: &[UsedResource]) -> Access {
    let Some(r) = layout.get(a.param) else {
        return Access::Unknown;
    };
    if a.dynamic.is_some() && matches!(r.root, RootType::Table | RootType::ResourceHeap) {
        return Access::Dynamic;
    }
    let Some(offset) = a.offset.checked_add(u32::from(first) * 4) else {
        return Access::Unknown;
    };
    if r.root == RootType::Table
        && offset.is_multiple_of(32)
        && used.iter().any(|u| {
            u.root == RootType::Table
                && u.slot == r.slot
                && u.user_data == r.user_data
                && u.table_offset_dwords == offset / 4
                && matches!(
                    u.kind,
                    ResourceType::Uav | ResourceType::Srv | ResourceType::Cbv
                )
        })
    {
        Access::Fixed
    } else {
        Access::Unknown
    }
}

// These hashes label dispatch counts only. They never influence classification.
const AUDITED: [&str; 3] = ["6a04b371a8e26853", "a06ef09446bd6e70", "7de0f9db96ddd295"];

#[derive(Clone, Copy)]
pub(crate) struct Request {
    heap: usize,
    access: Access,
    audited: Option<usize>,
}

impl Request {
    pub(crate) fn new(
        heap: &super::HostHeap,
        pipeline: &crate::pipeline::host::HostPipeline,
    ) -> Self {
        let access = pipeline
            .stages
            .iter()
            .filter(|s| s.translated().resource_heap)
            .map(|s| s.shader.heap_access())
            .max()
            .unwrap_or(Access::Unknown);
        let audited = if pipeline.compute {
            pipeline
                .stages
                .first()
                .and_then(|s| AUDITED.iter().position(|h| *h == s.shader.hash))
        } else {
            None
        };
        Self {
            heap: std::ptr::from_ref(heap) as usize,
            access,
            audited,
        }
    }
}

#[derive(Default, Clone)]
struct Counts {
    requests: [u64; 3],
    ran: [u64; 3],
    fixed_dispatches: u64,
    fixed_indirect: u64,
    audited: [u64; 3],
}

impl Counts {
    fn add(&mut self, rhs: &Self) {
        for i in 0..3 {
            self.requests[i] += rhs.requests[i];
            self.ran[i] += rhs.ran[i];
            self.audited[i] += rhs.audited[i];
        }
        self.fixed_dispatches += rhs.fixed_dispatches;
        self.fixed_indirect += rhs.fixed_indirect;
    }

    fn forced(&self) -> bool {
        self.requests[Access::Dynamic as usize] + self.requests[Access::Unknown as usize] > 0
    }

    fn class(&self) -> &'static str {
        match (self.requests[Access::Fixed as usize] > 0, self.forced()) {
            (true, false) => "fixed_only",
            (true, true) => "mixed",
            (false, true) => "full_only",
            (false, false) => "no_bindless",
        }
    }
}

// A list retains its HostHeap Arc through submission. Merge by host heap identity,
// including repeated executions of the same list; resetting the list clears its counts.
#[derive(Default)]
pub(crate) struct Usage {
    heaps: HashMap<usize, Counts>,
}

impl Usage {
    pub(crate) fn request(&mut self, request: Request) {
        self.heaps.entry(request.heap).or_default().requests[request.access as usize] += 1;
    }

    pub(crate) fn ran(&mut self, request: Request, compute: bool, indirect: bool) {
        let c = self.heaps.entry(request.heap).or_default();
        c.ran[request.access as usize] += 1;
        if compute && request.access == Access::Fixed {
            if indirect {
                c.fixed_indirect += 1;
            } else {
                c.fixed_dispatches += 1;
            }
        }
        if let Some(i) = request.audited {
            c.audited[i] += 1;
        }
    }

    fn merge(&mut self, list: &Self) {
        for (&heap, counts) in &list.heaps {
            self.heaps.entry(heap).or_default().add(counts);
        }
    }

    fn totals(&self) -> (Counts, u64) {
        let mut total = Counts::default();
        let mut same_heap_forced = 0;
        for c in self.heaps.values() {
            total.add(c);
            if c.forced() {
                same_heap_forced += c.fixed_dispatches + c.fixed_indirect;
            }
        }
        (total, same_heap_forced)
    }

    pub(crate) fn report(
        &self,
        sub: u64,
        lists: usize,
        plan_us: u64,
        flush_us: u64,
        placed_us: u64,
        cb_bytes: usize,
    ) {
        let (c, same_heap_forced) = self.totals();
        let [fixed_requests, unknown_requests, dynamic_requests] = c.requests;
        let [fixed_ran, unknown_ran, dynamic_ran] = c.ran;
        let [zero, silu, vecmat] = c.audited;
        tracing::info!(
            frame = crate::frame::current(),
            sub,
            lists,
            class = c.class(),
            heaps = self.heaps.len(),
            fixed_requests,
            dynamic_requests,
            unknown_requests,
            fixed_ran,
            dynamic_ran,
            unknown_ran,
            fixed_dispatches = c.fixed_dispatches,
            fixed_indirect = c.fixed_indirect,
            same_heap_forced,
            zero,
            silu,
            vecmat,
            plan_ms = plan_us as f64 / 1000.0,
            flush_ms = flush_us as f64 / 1000.0,
            placed_ms = placed_us as f64 / 1000.0,
            cb_mb = cb_bytes as f64 / 1048576.0,
            cb_bytes,
            "bindless submission attribution"
        );
    }
}

// Recording State and HostList reset together. Key by native list identity so
// replay and multi-list submissions can be attributed from the submitted slice.
type Recordings = HashMap<usize, Arc<Mutex<Usage>>>;
static RECORDINGS: LazyLock<Mutex<Recordings>> = LazyLock::new(Mutex::default);

pub(crate) struct Recording {
    key: usize,
    usage: Arc<Mutex<Usage>>,
}

impl Recording {
    pub(crate) fn new(list: &HostList) -> Self {
        let key = list.list.as_raw() as usize;
        let usage = Arc::new(Mutex::new(Usage::default()));
        RECORDINGS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, usage.clone());
        Self { key, usage }
    }

    pub(crate) fn with_usage(&self, f: impl FnOnce(&mut Usage)) {
        f(&mut self.usage.lock().unwrap_or_else(|e| e.into_inner()));
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        let mut records = RECORDINGS.lock().unwrap_or_else(|e| e.into_inner());
        if records
            .get(&self.key)
            .is_some_and(|entry| Arc::ptr_eq(entry, &self.usage))
        {
            records.remove(&self.key);
        }
    }
}

fn submitted(records: &Recordings, keys: impl Iterator<Item = usize>) -> Usage {
    let mut usage = Usage::default();
    for list in keys.filter_map(|key| records.get(&key)) {
        let list = list.lock().unwrap_or_else(|e| e.into_inner());
        usage.merge(&list);
    }
    usage
}

/// Main's submission hook supplies the same durations and bytes as its slow log.
pub(crate) fn finish_submission(
    sub: u64,
    runs: &[&HostList],
    plan_us: u64,
    flush_us: u64,
    placed_us: u64,
    cb_bytes: usize,
) {
    if enabled() {
        let records = RECORDINGS.lock().unwrap_or_else(|e| e.into_inner());
        let usage = submitted(
            &records,
            runs.iter().map(|list| list.list.as_raw() as usize),
        );
        drop(records);
        usage.report(sub, runs.len(), plan_us, flush_us, placed_us, cb_bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_shader::ir::Value;
    use victory_shader::resource::{Descriptor, Site};

    fn table() -> RootParam {
        RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 0,
        }
    }

    fn entry(offset: u32) -> UsedResource {
        UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: offset,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 4,
            slot: 0,
        }
    }

    fn loaded(site: Site) -> Source {
        Source::Loaded {
            load: site,
            first: 0,
        }
    }

    #[test]
    fn fixed_table_requires_a_declared_aligned_slot() {
        let a = Address {
            param: 0,
            offset: 32,
            dynamic: None,
        };
        assert_eq!(table_access(&a, 0, &[table()], &[entry(8)]), Access::Fixed);
        assert_eq!(
            table_access(&a, 1, &[table()], &[entry(8)]),
            Access::Unknown
        );
        assert_eq!(
            table_access(&a, 0, &[table()], &[entry(0)]),
            Access::Unknown
        );
        assert_eq!(table_access(&a, 0, &[], &[]), Access::Unknown);
        assert_eq!(classify_container(b"broken"), Access::Unknown);
    }

    #[test]
    fn dynamic_or_unresolved_alternative_forces_the_whole_stage() {
        let a = Address {
            param: 0,
            offset: 0,
            dynamic: None,
        };
        let mut res = Resources {
            scalar: vec![((0, 0), a), ((0, 1), Address { offset: 32, ..a })],
            descriptors: vec![Descriptor {
                site: (1, 0),
                role: Role::Resource,
                source: Source::Phi {
                    block: 1,
                    args: vec![loaded((0, 0)), loaded((0, 1))],
                },
            }],
            ..Default::default()
        };
        let used = [entry(0), entry(8)];
        assert_eq!(classify(&res, &[table()], &used), Access::Fixed);
        res.scalar[1].1.dynamic = Some(Value(3));
        assert_eq!(classify(&res, &[table()], &used), Access::Dynamic);
        res.scalar.pop();
        assert_eq!(classify(&res, &[table()], &used), Access::Unknown);
    }

    #[test]
    fn replays_merge_per_heap_and_failed_requests_never_count_as_ran() {
        let fixed = Request {
            heap: 1,
            access: Access::Fixed,
            audited: Some(0),
        };
        let dynamic = Request {
            heap: 2,
            access: Access::Dynamic,
            audited: None,
        };
        let mut list = Usage::default();
        list.request(fixed);
        list.ran(fixed, true, false);
        list.request(dynamic);
        let records = HashMap::from([(3, Arc::new(Mutex::new(list)))]);
        let mut submission = submitted(&records, [3, 4, 3].into_iter());
        let (c, forced) = submission.totals();
        assert_eq!(c.class(), "mixed");
        assert_eq!(c.fixed_dispatches, 2);
        assert_eq!(c.audited, [2, 0, 0]);
        assert_eq!(c.ran, [2, 0, 0]);
        assert_eq!(forced, 0);
        let mut second = Usage::default();
        second.request(Request { heap: 1, ..dynamic });
        submission.merge(&second);
        assert_eq!(submission.totals().1, 2);
        assert_eq!(Usage::default().totals().0.class(), "no_bindless");
    }

    #[test]
    fn unknown_request_is_forced_and_fixed_indirect_counts_once() {
        let mut list = Usage::default();
        let r = Request {
            heap: 7,
            access: Access::Fixed,
            audited: Some(2),
        };
        list.request(r);
        list.ran(r, true, true);
        let mut sub = Usage::default();
        sub.merge(&list);
        let c = sub.totals().0;
        assert_eq!(c.class(), "fixed_only");
        assert_eq!(
            (c.fixed_dispatches, c.fixed_indirect, c.audited),
            (0, 1, [0, 0, 1])
        );
        let mut other = Usage::default();
        other.request(Request { access: Access::Unknown, ..r });
        let c = other.totals().0;
        assert_eq!(c.class(), "full_only");
        assert_eq!(c.requests[Access::Unknown as usize], 1);
    }
}

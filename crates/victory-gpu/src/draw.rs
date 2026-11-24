//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, depth bounds, topology, targets) is set on the host
//! list as the title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_UNKNOWN,
};
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

const SHADER_RESOURCE: u32 = 0x40 | 0x80;
const DEPTH_WRITE: u32 = 0x10;
const DEPTH_READ: u32 = 0x20;

#[derive(Clone, Copy)]
struct DepthTarget {
    resource: usize,
    state: u32,
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: Option<DepthTarget>,
    /// The current draw also samples its read-only depth target.
    depth_sampled: bool,
    /// None leaves the host list's default bounds (0..1) in place.
    depth_bounds: Option<(f32, f32)>,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, what it holds) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u64), u32>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
    /// Last bindless view heap prepared on this list: `(heap, revision, copy_epoch)`.
    bindless_prepared: Option<(usize, (u64, u64, u64, u64), u64)>,
    /// Root SRVs bound as one-view tables in this list: (host heap, title
    /// address) → dynamic slot.
    root_srv_slots: std::collections::HashMap<(usize, u64), u32>,
    /// Root UAVs bound as one-view tables in this list: (host heap, title
    /// address) → dynamic slot.
    root_uav_slots: std::collections::HashMap<(usize, u64), u32>,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            crate::descriptor::release_dynamic(heap, first, n);
        }
    }
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootView {
    Cbv,
    Srv,
    Uav,
}

/// Permanently mapped zero-filled read-only fallback buffer (for unset or
/// unmapped root CBVs/SRVs) and writable sink buffer (for unset or unmapped
/// root UAVs, kept separate so shader writes never dirty the zero buffer).
fn fallback_root_va(uav: bool) -> Option<u64> {
    const SIZE: u64 = 4 << 20;
    static ZERO: std::sync::OnceLock<Option<(ID3D12Resource, u64)>> = std::sync::OnceLock::new();
    static SINK: std::sync::OnceLock<Option<(ID3D12Resource, u64)>> = std::sync::OnceLock::new();
    let slot = if uav { &SINK } else { &ZERO };
    slot.get_or_init(|| {
        let host = crate::host::host()?;
        let (buffer, data) = if uav {
            host.shared_buffer(SIZE)?
        } else {
            host.upload_buffer(SIZE)?
        };
        let label = if uav {
            "fallback sink root uav"
        } else {
            "fallback zero root view"
        };
        crate::host::name(&buffer, || label.into());
        // SAFETY: `data` points to `SIZE` mapped writable bytes, and `buffer` is live.
        let gpu = unsafe {
            data.write_bytes(0, SIZE as usize);
            buffer.GetGPUVirtualAddress()
        };
        Some((buffer, gpu))
    })
    .as_ref()
    .map(|&(_, gpu)| gpu)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory. Constant buffers refresh up to the D3D12 64 KiB
/// maximum; SRVs and UAVs refresh the full remaining extent of the backing
/// buffer (dirty-page watched) and UAVs also schedule write-back.
fn root_view(list: &HostList, address: u64, kind: RootView) -> Option<u64> {
    if address == 0 {
        return None;
    }
    let shadow = crate::shadow::find(address, 1)?;
    let gpu = shadow.gpu(address);
    match kind {
        RootView::Cbv => {
            let len = shadow.left(address).min(1 << 16);
            if gpu.is_multiple_of(256) {
                shadow.refresh(list, address, len);
                list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
                return Some(gpu);
            }
            let (gpu, to) = list.scratch(len)?;
            list.defer_fill(to, address as *const u8, len as usize);
            Some(gpu)
        }
        RootView::Srv => {
            if !gpu.is_multiple_of(4) {
                return None;
            }
            let len = shadow.left(address);
            shadow.refresh(list, address, len);
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
            Some(gpu)
        }
        RootView::Uav => {
            if !gpu.is_multiple_of(4) {
                return None;
            }
            let len = shadow.left(address);
            shadow.refresh(list, address, len);
            shadow.gpu_writes(list, address, len);
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_WRITE);
            Some(gpu)
        }
    }
}

fn emit_uav_barrier(list: &HostList) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    };
    // SAFETY: an open command list and a global UAV barrier.
    unsafe { list.list.ResourceBarrier(&[barrier]) };
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    #[cfg(test)]
    pub(crate) fn current_pipeline(&self) -> This {
        self.pipeline as This
    }

    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        self.bindless_prepared = None;
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.targets.clear();
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            if record.resource.is_null() {
                // A typed null RTV discards writes without changing the output slot.
                if let Some(desc) = record.desc() {
                    rtvs.push(host.rtv(None, Some(&desc)));
                }
                continue;
            }
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                continue;
            };
            if !crate::host::allows_rtv(texture) {
                continue;
            }
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            rtvs.push(host.rtv(Some(texture), record.desc().as_ref()));
        }
        self.depth = None;
        self.depth_sampled = false;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            let desc = record.depth_desc();
            self.depth = Some(DepthTarget {
                resource: record.resource as usize,
                state: depth_state(texture, desc.as_ref()),
            });
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some(host.dsv(texture, desc.as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// The value of user-data dword `u` of a translated shader: a root
    /// argument, or a bound heap's guard slot.
    fn user_dword(&self, compute: bool, u: &victory_shader::emit::UserData) -> u32 {
        let guard = |samplers: bool| {
            let h = self.heaps[usize::from(samplers)];
            (h != 0)
                .then(|| crate::descriptor::heap(h as This).host())
                .flatten()
                .map_or(0, |h| h.guard_slot())
        };
        match u.root {
            RootType::ResourceHeap => guard(false),
            RootType::SamplerHeap => guard(true),
            _ => self.user_value(compute, u.slot, u.root, u.dword),
        }
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    pub(crate) fn set_depth_bounds(&mut self, list: &HostList, min: f32, max: f32) {
        self.depth_bounds = Some((min, max));
        if list.depth_bounds.is_none() {
            crate::host::warn_depth_bounds();
        }
        self.apply_depth_bounds(list);
    }

    fn apply_depth_bounds(&self, list: &HostList) {
        if let (Some(list), Some((min, max))) = (&list.depth_bounds, self.depth_bounds) {
            // SAFETY: the host supports depth bounds and the list is open.
            unsafe { list.OMSetDepthBounds(min, max) };
        }
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(
                mesh,
                "a draw or mesh dispatch with another kind of pipeline"
            );
            return false;
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            // After `bind`: a shader may have written the buffer before.
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        self.ensure_depth_state(list);
        true
    }

    fn ensure_depth_state(&self, list: &HostList) {
        if let Some(depth) = self.depth {
            let state = depth.state | if self.depth_sampled { SHADER_RESOURCE } else { 0 };
            crate::list::ensure_state(list, depth.resource as This, state);
        }
    }

    fn read_only_depth(&self, compute: bool) -> Option<usize> {
        self.depth
            .filter(|depth| !compute && depth.state == DEPTH_READ)
            .map(|depth| depth.resource)
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    pub(crate) fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        self.depth_sampled = false;
        let l = &list.list;
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| {
                    let title = crate::descriptor::heap(h as This);
                    let heap = title.host()?;
                    Some(Some(heap.heap.clone()))
                })
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            // Also reached after the indirect clamp invalidates the pipeline.
            if !compute {
                self.apply_depth_bounds(list);
            }
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        let mut bound_params = 0u64;
        if p.stages.iter().any(|s| s.translated().resource_heap)
            && self.heaps[0] != 0
            && let Some(host_heap) = crate::descriptor::heap(self.heaps[0] as This).host_arc()
        {
            let epoch = list.title_write_epoch();
            let key = (self.heaps[0], host_heap.revision(), epoch);
            if self.bindless_prepared != Some(key) {
                host_heap.prepare_bindless_record(list);
                self.bindless_prepared = Some((self.heaps[0], host_heap.revision(), epoch));
            }
            let writes = p.stages.iter().any(|s| s.translated().writes_memory);
            list.defer_bindless(&host_heap, writes);
        }
        for stage in &p.stages {
            let t = stage.translated();
            let Some(made) = self.tables(list, &t.used, compute) else {
                return false;
            };
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    if u.root == RootType::Table
                        && fresh.is_none()
                        && let Some(&Root::Table(handle)) =
                            self.roots_ref(compute).get(usize::from(u.slot))
                        && let Some(heap) = self.heap_of(handle)
                        && let (Some(host), Some(index)) = (heap.host(), heap.index(handle))
                        && index >= host.count()
                    {
                        once!(
                            slot = u.slot,
                            index,
                            count = host.count(),
                            "untracked descriptor table past host heap: draw/dispatch dropped"
                        );
                        return false;
                    }
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_dword(compute, u),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, kind) in [
                (&stage.root_cbvs, RootView::Cbv),
                (&stage.root_srvs, RootView::Srv),
                (&stage.root_uavs, RootView::Uav),
            ] {
                for &(slot, param) in views {
                    if param < 64 {
                        if bound_params & (1u64 << param) != 0 {
                            continue;
                        }
                        bound_params |= 1u64 << param;
                    }
                    let uav = kind == RootView::Uav;
                    let use_table = (kind == RootView::Srv
                        && crate::pipeline::host::bounded_root_srvs())
                        || (uav && crate::pipeline::host::bounded_root_uavs());
                    if use_table {
                        let address = match self.roots_ref(compute).get(usize::from(slot)) {
                            Some(&Root::View(address)) => Some(address),
                            _ => {
                                once!(
                                    slot,
                                    ?kind,
                                    "a shader reads a root view the title did not set"
                                );
                                None
                            }
                        };
                        if !self.root_buffer_table(list, address, param, compute, uav) {
                            return false;
                        }
                        continue;
                    }
                    let gpu = match self.roots_ref(compute).get(usize::from(slot)) {
                        Some(&Root::View(address)) if address != 0 => {
                            match root_view(list, address, kind) {
                                Some(gpu) => gpu,
                                None => {
                                    once!(
                                        slot,
                                        address = format_args!("{address:#x}"),
                                        ?kind,
                                        "root view outside mapped title memory; binding zeroed fallback"
                                    );
                                    let Some(gpu) = fallback_root_va(uav) else {
                                        continue;
                                    };
                                    gpu
                                }
                            }
                        }
                        Some(Root::View(_)) => {
                            once!(
                                slot,
                                ?kind,
                                "null root view address; binding zeroed fallback"
                            );
                            let Some(gpu) = fallback_root_va(uav) else {
                                continue;
                            };
                            gpu
                        }
                        _ => {
                            once!(
                                slot,
                                ?kind,
                                "a shader reads a root view the title did not set; binding zeroed fallback"
                            );
                            let Some(gpu) = fallback_root_va(uav) else {
                                continue;
                            };
                            gpu
                        }
                    };
                    // SAFETY: an open list; `gpu` lies in a live host or fallback buffer.
                    unsafe {
                        match (compute, kind) {
                            (false, RootView::Cbv) => {
                                l.SetGraphicsRootConstantBufferView(param, gpu)
                            }
                            (false, RootView::Srv) => {
                                l.SetGraphicsRootShaderResourceView(param, gpu)
                            }
                            (false, RootView::Uav) => {
                                l.SetGraphicsRootUnorderedAccessView(param, gpu)
                            }
                            (true, RootView::Cbv) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, RootView::Srv) => l.SetComputeRootShaderResourceView(param, gpu),
                            (true, RootView::Uav) => {
                                l.SetComputeRootUnorderedAccessView(param, gpu)
                            }
                        }
                    }
                }
            }
            if let Some(param) = stage.stride_table {
                let gpu = (self.heaps[0] != 0)
                    .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                    .flatten()
                    .and_then(|h| h.stride_table_gpu());
                if let Some(gpu) = gpu {
                    // SAFETY: an open list; the stride table buffer lives with the host heap.
                    unsafe {
                        if compute {
                            l.SetComputeRootShaderResourceView(param, gpu);
                        } else {
                            l.SetGraphicsRootShaderResourceView(param, gpu);
                        }
                    }
                } else {
                    once!("a shader reads the runtime stride table without a host view heap");
                }
            }
        }
        true
    }

    /// Binds a root SRV or UAV as a one-view table (see
    /// [`crate::pipeline::host::bounded_root_srvs`] /
    /// [`crate::pipeline::host::bounded_root_uavs`]): a raw view of the
    /// title buffer's host copy from `address` to the copy's end, or a null
    /// view (no address, no host copy, or unaligned start). One slot per
    /// address per view kind per list; false if no dynamic slot could be had.
    fn root_buffer_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        uav: bool,
    ) -> bool {
        let Some((heap, host)) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find_map(|h| h.host().filter(|x| !x.is_sampler()).map(|x| (h, x)))
        else {
            once!(uav, "a root buffer view without a bound view heap: the draw is dropped");
            return false;
        };
        let shadow = address
            .filter(|&a| a != 0)
            .and_then(|a| crate::shadow::find(a, 1).map(|s| (a, s)));
        let key = (std::ptr::from_ref(heap) as usize, address.unwrap_or(0));
        let slots = if uav {
            &mut self.root_uav_slots
        } else {
            &mut self.root_srv_slots
        };
        let index = match slots.get(&key) {
            Some(&index) => index,
            None => {
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                self.dynamic.push((key.0, index, 1));
                slots.insert(key, index);
                let install_null = |index: u32| {
                    if uav {
                        host.null_raw_uav_at(index);
                    } else {
                        host.null_raw_at(index);
                    }
                };
                match &shadow {
                    Some((a, s)) => {
                        // SAFETY: a live buffer.
                        let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                        let offset = s.gpu(*a) - base;
                        // SAFETY: a live buffer.
                        let uav_ok = !uav
                            || (unsafe { s.buffer.GetDesc() }.Flags.0
                                & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0
                                != 0);
                        if offset.is_multiple_of(16) && uav_ok {
                            if uav {
                                host.raw_uav_at(index, &s.buffer, offset, s.left(*a));
                            } else {
                                host.raw_at(index, &s.buffer, offset, s.left(*a));
                            }
                        } else {
                            // Raw views start on 16 bytes; one from the
                            // aligned start would shift every access.
                            once!(
                                address = format_args!("{a:#x}"),
                                uav,
                                "a root buffer view unaligned or non-UAV falls back to null"
                            );
                            install_null(index);
                        }
                    }
                    None => install_null(index),
                }
                index
            }
        };
        if let Some((a, s)) = &shadow {
            let len = s.left(*a);
            s.refresh(list, *a, len);
            if uav {
                s.gpu_writes(list, *a, len);
                list.buffer_state(&s.buffer, crate::host::BUFFER_WRITE);
            } else {
                list.buffer_state(&s.buffer, crate::host::BUFFER_READ);
            }
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        if p.has_root_uavs() {
            emit_uav_barrier(list);
        }
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = self.user_dword(compute, u);
                let table = u.root == RootType::Table;
                let guard = matches!(u.root, RootType::ResourceHeap | RootType::SamplerHeap);
                let (kind, index) = if table {
                    ("table", v / 32)
                } else if guard {
                    ("guard", v)
                } else {
                    ("const", v)
                };
                let _ = write!(text, " u{}={v:#x} {kind}", u.index);
                if !guard && let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let at = if table { "heap" } else { "as heap index" };
                    let _ = write!(text, " [{at} {index}: {slot}]");
                }
            }
            for &(slot, _) in stage
                .root_cbvs
                .iter()
                .chain(&stage.root_srvs)
                .chain(&stage.root_uavs)
            {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        text
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    if crate::pipeline::host_pipeline(self.pipeline as This)
                        .is_some_and(|p| p.has_root_uavs())
                    {
                        emit_uav_barrier(list);
                    }
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&mut self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            self.depth_sampled |= readable(list, u, &view, self.read_only_depth(compute));
        } else if index < host.count() {
            host.make_null_kind(index, u.kind, u.dwords);
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    fn tables(&mut self, list: &HostList, used: &[UsedResource], compute: bool) -> Option<Vec<(u8, u32)>> {
        let read_only_depth = self.read_only_depth(compute);
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let Some(host) = heap.host() else {
                continue;
            };
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8 + 1)
                .max()
                .unwrap_or(1);
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                }
                (table, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        self.depth_sampled |= readable(list, u, &view, read_only_depth);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                return None;
            };
            self.fresh_tables.insert(key, first);
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, first, n));
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(View::Constant(address, size)) => {
                        let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                            host.make_null_kind(index, ResourceType::Cbv, 4);
                            continue;
                        };
                        list.defer_fill(to, address as *const u8, size as usize);
                        host.constant_at(index, gpu, size);
                    }
                    Some(view) => {
                        host.make_at(index, &view);
                        if let Some(u) = used {
                            self.depth_sampled |= readable(list, u, &view, read_only_depth);
                        }
                    }
                    None => {
                        if let Some(u) = used {
                            host.make_null_kind(index, u.kind, u.dwords);
                            once!(slot, "a table entry without a view");
                        } else {
                            host.make_null(index);
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        Some(made)
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_resource(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let rects = self.topology == RECTANGLES;
        let ran = self.prepare(list, false) && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            if crate::pipeline::host_pipeline(self.pipeline as This)
                .is_some_and(|p| p.has_root_uavs())
            {
                emit_uav_barrier(list);
            }
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// The vertices an indirect rectangle draw's expansion covers: the
    /// fewest that any buffer the pipeline reads per vertex holds, in whole
    /// rectangles (2^24 if it reads none: vertex ids only). None if a buffer
    /// it reads is not bound or holds no rectangle.
    /// shortcut: at most 4 MiB of expanded vertices per buffer, as the
    /// expansion copies the bound buffers whole on the CPU at submission;
    /// records past that draw nothing.
    fn rectangle_vertices(&self) -> Option<u32> {
        const EXPANDED: u32 = 4 << 20;
        let p = crate::pipeline::host_pipeline(self.pipeline as This)?;
        let mut vertices = 1 << 24;
        for e in p.inputs.iter().filter(|e| !e.per_instance) {
            let v = self.vertex[e.slot as usize];
            if v.address == 0 || v.stride == 0 {
                return None;
            }
            vertices = (v.size / v.stride)
                .min(EXPANDED / 2 / v.stride)
                .min(vertices);
        }
        (vertices >= 3).then_some(vertices / 3 * 3)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        if self.topology == RECTANGLES {
            once!("indexed rectangles not drawn");
        }
        let ran = self.topology != RECTANGLES && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            if crate::pipeline::host_pipeline(self.pipeline as This)
                .is_some_and(|p| p.has_root_uavs())
            {
                emit_uav_barrier(list);
            }
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        let matches_kind = match kind {
            DRAW | INDEXED => self.topology != RECTANGLES && !p.compute && !p.mesh,
            MESH => !p.compute && p.mesh,
            DISPATCH => p.compute,
            _ => false,
        };
        if !matches_kind {
            why(&format!(
                "pipeline mismatch (compute {}, mesh {}, topology {})",
                p.compute, p.mesh, self.topology
            ));
            return false;
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        // Bounded first: the clamp binds a compute pipeline of its own and
        // copies clamped active records into a scratch buffer at offset 0.
        let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
        let Some(bounded) = crate::indirect::bound(
            list,
            (&args.buffer, args.offset(arguments)),
            count_arg,
            (kind, stride, max),
        ) else {
            why("not bounded (no clamp, or arguments not 4-byte aligned)");
            return false;
        };
        // Rectangles: the draws of their expanded vertices, from records of
        // our own; the bounded scratch returns to common after rewriting.
        let rects = kind == DRAW && self.topology == RECTANGLES;
        let (mut from, mut at, mut signature) = (bounded.clone(), 0u64, signature);
        let mut expanded = 0;
        if rects {
            // SAFETY: `bounded` is a live buffer returned by `bound`.
            let bounded_gpu = unsafe { bounded.GetGPUVirtualAddress() };
            let made = self
                .rectangle_vertices()
                .zip(host_signature(DRAW, 16))
                .and_then(|(vertices, triangles)| {
                    let records = (stride, max, vertices);
                    let (buffer, offset) =
                        crate::indirect::rectangles(list, &bounded, bounded_gpu, records)?;
                    Some((buffer, offset, triangles, vertices))
                });
            let Some(made) = made else {
                crate::indirect::done(list, &bounded);
                why("rectangles not expanded");
                return false;
            };
            (from, at, signature, expanded) = made;
        }
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        let ran = match kind {
            DRAW if rects => self.prepare(list, false) && self.rectangles(list, expanded, 0),
            DRAW | INDEXED => self.topology != RECTANGLES && self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => match crate::pipeline::host_pipeline(self.pipeline as This) {
                Some(p) if p.compute => self.bind(list, p, true),
                _ => false,
            },
            _ => false,
        };
        if !ran {
            crate::indirect::done(list, &from);
            if rects {
                crate::indirect::done(list, &bounded);
            }
            #[cfg(feature = "diagnostics")]
            match crate::pipeline::host_pipeline(self.pipeline as This) {
                None => crate::pipeline::note_drop(self.pipeline as This),
                Some(p) => why(&format!(
                    "prepare failed (compute {}, mesh {})",
                    p.compute, p.mesh
                )),
            }
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(
                format!("indirect kind {kind} max {max} at {arguments:#x}"),
                kind == DISPATCH,
            )
        });
        // SAFETY: an open list with everything set; the scratch buffer and
        // host copies are kept alive by the list, and the signature is cached.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                &from,
                at,
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        crate::indirect::done(list, &from);
        if rects {
            crate::indirect::done(list, &bounded);
        }
        if crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.has_root_uavs())
        {
            emit_uav_barrier(list);
        }
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// The tracker uses whole resources, so both planes must be read-only
/// when the depth format carries stencil.
fn depth_state(texture: &ID3D12Resource, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) -> u32 {
    let Some(desc) = desc else {
        return DEPTH_WRITE;
    };
    let format = if desc.Format == DXGI_FORMAT_UNKNOWN {
        // SAFETY: a live texture; a default format inherits its format.
        unsafe { texture.GetDesc() }.Format
    } else {
        desc.Format
    };
    let mut required = D3D12_DSV_FLAG_READ_ONLY_DEPTH.0;
    if format != DXGI_FORMAT_D16_UNORM && format != DXGI_FORMAT_D32_FLOAT {
        required |= D3D12_DSV_FLAG_READ_ONLY_STENCIL.0;
    }
    if desc.Flags.0 & required == required {
        DEPTH_READ
    } else {
        DEPTH_WRITE
    }
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need. Returns whether the view
/// also reads the draw's bound depth target.
fn readable(list: &HostList, u: &UsedResource, view: &View, read_only_depth: Option<usize>) -> bool {
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_deref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((address, len)) = crate::descriptor::host::buffer_range(view)
        && len > 0
        && let Some(shadow) = crate::shadow::find(address, len)
    {
        shadow.refresh(list, address, len);
        if u.kind == ResourceType::Uav {
            shadow.gpu_writes(list, address, len);
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_WRITE);
        } else {
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
        }
    }
    match (view, u.kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            if let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This, t);
            }
            // Keep the same read bits the depth target will request at draw time.
            let samples_depth = read_only_depth == Some(*resource);
            let state = if samples_depth {
                DEPTH_READ | SHADER_RESOURCE
            } else {
                SHADER_RESOURCE
            };
            crate::list::ensure_state(list, *resource as This, state);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
            samples_depth
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            false
        }
        _ => false,
    }
}

#[cfg(test)]
mod depth_tests {
    use super::*;

    #[test]
    fn shader_table_keeps_read_only_depth_state() {
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_SAMPLE_DESC,
        };

        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            ..Default::default()
        };
        let resource_iid = ID3D12Resource::IID;
        let mut depth = null;
        assert_eq!(
            crate::resource::create_committed(
                null,
                (&raw const heap).cast(),
                0,
                (&raw const desc).cast(),
                D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32,
                null.cast(),
                (&raw const resource_iid).cast(),
                &mut depth,
            ),
            crate::object::S_OK
        );
        #[repr(C)]
        struct ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC,
            placement: [u64; 2],
        }
        let desc = ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            },
            placement: [0; 2],
        };
        let heap_iid = ID3D12DescriptorHeap::IID;
        let mut views = null;
        assert_eq!(
            crate::descriptor::create_heap(
                null,
                (&raw const desc).cast(),
                (&raw const heap_iid).cast(),
                &mut views,
            ),
            crate::object::S_OK
        );
        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        let srv = crate::descriptor::heap(views).start;
        crate::descriptor::create_srv(null, depth, (&raw const srv_desc).cast(), srv);
        let used = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };

        let shader_resource = (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
            .0 as u32;
        let combined = D3D12_RESOURCE_STATE_DEPTH_READ.0 as u32 | shader_resource;
        for (flags, unsampled_state, shader_state, draw_state) in [
            (
                D3D12_DSV_FLAG_READ_ONLY_DEPTH,
                DEPTH_READ,
                combined,
                combined,
            ),
            (
                D3D12_DSV_FLAG_NONE,
                DEPTH_WRITE,
                shader_resource,
                DEPTH_WRITE,
            ),
        ] {
            let list = host.new_list().expect("host list");
            let desc = [
                DXGI_FORMAT_D32_FLOAT.0 as u32,
                D3D12_DSV_DIMENSION_TEXTURE2D.0 as u32,
                flags.0 as u32,
                0,
                0,
                0,
            ];
            let mut slot = [0u64; 16];
            let dsv = slot.as_mut_ptr() as u64;
            crate::descriptor::create_dsv(null, depth, &desc, dsv);
            let mut state = State::default();
            state.set_targets(&list, &[], dsv);
            state.set_heaps(&[views]);
            state.set_table(false, 0, srv);
            crate::list::ensure_state(&list, depth, D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32);
            state.ensure_depth_state(&list);
            assert!(
                list.transition(depth, unsampled_state).is_none(),
                "unsampled depth state for {flags:?}"
            );
            state.tables(&list, &[used], false);
            assert!(
                list.transition(depth, shader_state).is_none(),
                "shader-table state for {flags:?}"
            );
            state.ensure_depth_state(&list);
            assert!(
                list.transition(depth, draw_state).is_none(),
                "depth state for {flags:?}"
            );
            list.close();
            host.execute(&[&list]);
        }
        let release = |object: This| {
            crate::object::slot::<extern "system" fn(This) -> u32>(object, 0x10)(object)
        };
        assert_eq!(release(views), 0);
        assert_eq!(release(depth), 0);
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_resource(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!(
                "{}x{} f{} flags {:#x}",
                d.Width, d.Height, d.Format.0, d.Flags.0
            )
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        Some(View::Unordered(r, ..)) => format!("uav {:?}", texture(*r as This)),
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_shader::native::{ComputeIo, StageIo};
    use victory_shader::{
        Container, HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource,
    };
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
        VirtualFree,
    };

    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    struct Allocation {
        ptr: *mut u8,
        len: usize,
    }

    impl Allocation {
        fn watched(len: usize) -> Self {
            // SAFETY: allocates a fresh write-watched virtual range freed in Drop.
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
            Self { ptr, len }
        }

        fn reserved_unmapped(len: usize) -> Self {
            // SAFETY: reserves an uncommitted VA range so accesses are unmapped.
            let ptr = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE, PAGE_NOACCESS) }
                .cast::<u8>();
            assert!(!ptr.is_null());
            Self { ptr, len }
        }

        fn addr(&self) -> u64 {
            self.ptr as u64
        }
    }

    impl Drop for Allocation {
        fn drop(&mut self) {
            crate::host::forget_unreadable(self.ptr as usize, self.len);
            crate::watch::forget_journal(self.ptr as usize, self.len);
            // SAFETY: `self.ptr` came from `VirtualAlloc`.
            unsafe {
                VirtualFree(self.ptr.cast(), 0, MEM_RELEASE);
            }
        }
    }

    /// Synthetic GFX10 compute shader reading Root CBV (slot 0, dword 0) and
    /// Root SRV (slot 1, byte offset `+1.5 MiB`), and storing `cbv ^ srv ^
    /// 0xA5A5_5A5A` to Root UAV (slot 2, byte offset `+1.5 MiB`).
    fn build_root_view_cs() -> (victory_shader::emit::HostShader, Vec<UsedResource>) {
        let s_mov_b32 =
            |sdst: u8, src0: u8| 0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32;
        let s_load_dword = |dst: u8, base: u8, byte_off: u32| -> [u32; 2] {
            [
                0x3D << 26 | (dst as u32) << 6 | (base as u32 >> 1),
                125 << 25 | (byte_off & 0x1F_FFFF),
            ]
        };
        let vop1 = |op: u16, vdst: u8, src0: u16| -> u32 {
            0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
        };
        let buffer_load_dword_offen = |vdst: u8, vaddr: u8, srsrc: u8| -> [u32; 2] {
            [
                0x38 << 26 | 0x0C << 18 | (1 << 12),
                0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdst as u32) << 8 | vaddr as u32,
            ]
        };
        let buffer_store_dword_offen = |vdata: u8, vaddr: u8, srsrc: u8| -> [u32; 2] {
            [
                0x38 << 26 | 0x1C << 18 | (1 << 12),
                0x80 << 24
                    | ((srsrc as u32 >> 2) & 0x1F) << 16
                    | (vdata as u32) << 8
                    | vaddr as u32,
            ]
        };

        let mut words = Vec::new();
        // s[0:1] = CBV 0, s[2:3] = SRV 1, s[4:5] = UAV 2.
        words.extend_from_slice(&s_load_dword(6, 0, 0));
        // SRV descriptor in s[8:11], UAV descriptor in s[12:15].
        words.push(s_mov_b32(8, 2));
        words.push(s_mov_b32(9, 3));
        words.push(s_mov_b32(10, 0xC1));
        words.extend_from_slice(&[s_mov_b32(11, 0xFF), 0x3101_4FAC]);
        words.push(s_mov_b32(12, 4));
        words.push(s_mov_b32(13, 5));
        words.push(s_mov_b32(14, 0xC1));
        words.extend_from_slice(&[s_mov_b32(15, 0xFF), 0x3101_4FAC]);
        let vop2 = |op: u8, vdst: u8, src0: u16, vsrc1: u8| -> u32 {
            (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
        };
        // v0 = 0x18_0000 (+1.5 MiB); v1 = srv[0x18_0000]; v2 = s6 ^ v1 ^ 0xA5A5_5A5A.
        words.extend_from_slice(&[vop1(0x001, 0, 0xFF), 0x0018_0000]);
        words.extend_from_slice(&buffer_load_dword_offen(1, 0, 8));
        words.push(vop2(0x1D, 2, 6, 1)); // v_xor_b32 v2, s6, v1
        words.extend_from_slice(&[vop2(0x1D, 2, 0xFF, 2), 0xA5A5_5A5A]); // v_xor_b32 v2, literal, v2
        words.extend_from_slice(&buffer_store_dword_offen(2, 0, 12));
        words.push(0xBF81_0000); // s_endpgm

        let used = vec![
            UsedResource {
                kind: ResourceType::Cbv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 2,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 4,
                slot: 2,
            },
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![
                RootParam {
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    sgprs: 2,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                RootParam {
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    sgprs: 2,
                    base_register: 1,
                    user_data: 2,
                    slot: 1,
                },
                RootParam {
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    sgprs: 2,
                    base_register: 2,
                    user_data: 4,
                    slot: 2,
                },
            ],
            used: used.clone(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: 6 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = victory_shader::ir::build(&native.code, false, 8).unwrap();
        let res = victory_shader::resource::track(&p, &native.layout, native.stage).unwrap();
        let io = victory_shader::io::raise(&p, &native).unwrap();
        let st = victory_shader::structure::structure(&p).unwrap();
        let c = Container {
            digest: [0; 16],
            version: (1, 0),
            total_size: 0,
            parts: Vec::new(),
        };
        let shader = victory_shader::emit::emit(&c, &native, &p, &st, &io, &res).unwrap();
        (shader, used)
    }

    #[test]
    fn table_entry_unbound_slots_install_typed_null_views_without_faulting() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let s_mov_b32 = |dst: u8, src: u8| -> u32 {
            0xBE80_0000 | (dst as u32) << 16 | 0x03 << 8 | src as u32
        };
        let s_load_dwordx4 = |dst: u8, base: u8, byte_off: u32| -> [u32; 2] {
            [
                0x3D << 26 | 2 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
                125 << 25 | (byte_off & 0x1F_FFFF),
            ]
        };
        let v_mov_b32 = |vdst: u8, src0: u16| -> u32 {
            0x7E00_0000 | (vdst as u32) << 17 | (1 << 9) | (src0 as u32 & 0x1FF)
        };
        let buffer_load_dword_offen = |vdst: u8, vaddr: u8, srsrc: u8| -> [u32; 2] {
            [
                0x38 << 26 | 0x0C << 18 | (1 << 12),
                0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdst as u32) << 8 | vaddr as u32,
            ]
        };
        let buffer_store_dword_offen = |vdata: u8, vaddr: u8, srsrc: u8| -> [u32; 2] {
            [
                0x38 << 26 | 0x1C << 18 | (1 << 12),
                0x80 << 24
                    | ((srsrc as u32 >> 2) & 0x1F) << 16
                    | (vdata as u32) << 8
                    | vaddr as u32,
            ]
        };

        // Table at s[0:1]: entry +0 (dwords 0..4, offset 0) is SRV t0,
        // entry +1 (dwords 8..12, offset 32 bytes) is UAV u0.
        let mut words = Vec::new();
        words.push(s_mov_b32(1, 0x80)); // high word = 0 -> s[0:1] is descriptor table
        words.extend_from_slice(&s_load_dwordx4(4, 0, 0));
        words.extend_from_slice(&s_load_dwordx4(8, 0, 32));
        words.push(v_mov_b32(0, 0x80)); // offset = 0
        words.extend_from_slice(&buffer_load_dword_offen(1, 0, 4));
        words.extend_from_slice(&buffer_store_dword_offen(1, 0, 8));
        words.push(0xBF81_0000); // s_endpgm

        let used = vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: used.clone(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = victory_shader::ir::build(&native.code, false, 4).unwrap();
        let res = victory_shader::resource::track(&p, &native.layout, native.stage).unwrap();
        let io = victory_shader::io::raise(&p, &native).unwrap();
        let st = victory_shader::structure::structure(&p).unwrap();
        let c = Container {
            digest: [0; 16],
            version: (1, 0),
            total_size: 0,
            parts: Vec::new(),
        };
        let shader = victory_shader::emit::emit(&c, &native, &p, &st, &io, &res).unwrap();
        let pso = crate::pipeline::test_compute_pipeline(shader, used);

        #[repr(C)]
        struct ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC,
            placement: [u64; 2],
        }
        let heap_desc = ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 8,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            },
            placement: [0; 2],
        };
        let heap_iid = ID3D12DescriptorHeap::IID;
        let mut views = std::ptr::null();
        assert_eq!(
            crate::descriptor::create_heap(
                std::ptr::null(),
                (&raw const heap_desc).cast(),
                (&raw const heap_iid).cast(),
                &mut views,
            ),
            crate::object::S_OK
        );
        let heap_start =
            crate::object::slot::<extern "system" fn(This) -> u64>(views, 0x48)(views);

        // Both table entries (slot 0 SRV at heap_start+0 and slot 1 UAV at heap_start+32)
        // are deliberately left uninitialized (`None` / null) -> `table_entry` must write
        // typed null buffer SRV / raw null buffer UAV and execute with 0 debug-layer errors.
        let list = host.new_list().expect("new_list");
        let mut state = State::default();
        state.set_heaps(&[views]);
        state.set_pipeline(pso);
        state.set_table(true, 0, heap_start);
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);
        assert_eq!(host.log_messages(), None, "no D3D12 debug errors expected");

        let release =
            |o: This| crate::object::slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(views), 0);
        assert_eq!(release(pso), 0);
    }

    #[test]
    fn null_and_unmapped_root_views_use_safe_distinct_fallbacks() {
        assert!(crate::shadow::find(0, 1).is_none());
        assert!(crate::shadow::find(0x1000, 64).is_none());
        assert!(crate::shadow::find(u64::MAX - 16, 1).is_none());
        assert!(crate::shadow::find(u64::MAX - (1 << 20), 1).is_none());

        let unmapped = Allocation::reserved_unmapped(8 << 20);
        let mid = unmapped.addr() + (4 << 20);
        assert!(crate::shadow::find(mid, 64).is_none());

        let Some(host) = crate::host::host() else {
            return;
        };
        let list = host.new_list().expect("new_list");
        assert!(root_view(&list, 0, RootView::Cbv).is_none());
        assert!(root_view(&list, 0, RootView::Srv).is_none());
        assert!(root_view(&list, 0, RootView::Uav).is_none());
        assert!(root_view(&list, mid, RootView::Srv).is_none());
        assert!(root_view(&list, mid, RootView::Uav).is_none());
        assert!(root_view(&list, u64::MAX - 16, RootView::Srv).is_none());
        assert!(root_view(&list, u64::MAX - 16, RootView::Uav).is_none());

        let zero_va = fallback_root_va(false).expect("zero fallback VA");
        let sink_va = fallback_root_va(true).expect("sink fallback VA");
        assert_ne!(zero_va, 0);
        assert_ne!(sink_va, 0);
        assert_ne!(zero_va, sink_va);
        assert!(zero_va.is_multiple_of(256));
        assert!(sink_va.is_multiple_of(256));
        list.close();
    }

    #[test]
    fn root_srv_refreshes_dirty_data_past_one_mib() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let len = 2 << 20;
        let alloc = Allocation::watched(len);
        let base = alloc.addr();
        let resource_id = alloc.ptr as usize;
        crate::shadow::register(resource_id, base, len as u64);

        let off_a = 0x18_0000usize; // +1.5 MiB
        let off_b = 0x1C_0000usize; // +1.75 MiB
        // SAFETY: `off_a` and `off_b` lie within the 2 MiB committed allocation.
        unsafe {
            alloc.ptr.add(off_a).cast::<u32>().write(0xCAFE_1234);
        }

        let list = host.new_list().expect("new_list");
        let shadow = crate::shadow::find(base, 4).expect("shadow");
        assert_eq!(
            root_view(&list, base, RootView::Srv),
            Some(shadow.gpu(base))
        );
        list.close();
        host.execute(&[&list]);
        assert_eq!(shadow.read_u32(off_a), 0xCAFE_1234);

        // Second submission mutates another page past +1 MiB and verifies
        // incremental dirty-page refresh updates it too.
        list.reset();
        // SAFETY: within the 2 MiB allocation.
        unsafe {
            alloc.ptr.add(off_b).cast::<u32>().write(0xBEEF_5678);
        }
        assert_eq!(
            root_view(&list, base, RootView::Srv),
            Some(shadow.gpu(base))
        );
        list.close();
        host.execute(&[&list]);
        assert_eq!(shadow.read_u32(off_a), 0xCAFE_1234);
        assert_eq!(shadow.read_u32(off_b), 0xBEEF_5678);

        crate::shadow::unregister(resource_id, base);
    }

    #[test]
    fn dispatch_binds_fallbacks_and_copies_root_uav_writes_back() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let (shader, used) = build_root_view_cs();
        let pso = crate::pipeline::test_compute_pipeline(shader, used);

        let unmapped = Allocation::reserved_unmapped(8 << 20);
        let mid_unmapped = unmapped.addr() + (4 << 20);

        let len = 2 << 20;
        let srv_buf = Allocation::watched(len);
        let uav_buf = Allocation::watched(len);
        let cbv_buf = Allocation::watched(64 << 10);
        crate::shadow::register(srv_buf.ptr as usize, srv_buf.addr(), len as u64);
        crate::shadow::register(uav_buf.ptr as usize, uav_buf.addr(), len as u64);
        crate::shadow::register(cbv_buf.ptr as usize, cbv_buf.addr(), 64 << 10);

        let off = 0x18_0000usize; // +1.5 MiB
        // SAFETY: offsets are inside the committed allocations.
        unsafe {
            cbv_buf.ptr.cast::<u32>().write(0x1357_2468);
            srv_buf.ptr.add(off).cast::<u32>().write(0xDEAD_BEEF);
        }

        #[repr(C)]
        struct ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC,
            placement: [u64; 2],
        }
        let heap_desc = ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 8,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            },
            placement: [0; 2],
        };
        let heap_iid = ID3D12DescriptorHeap::IID;
        let mut views = std::ptr::null();
        assert_eq!(
            crate::descriptor::create_heap(
                std::ptr::null(),
                (&raw const heap_desc).cast(),
                (&raw const heap_iid).cast(),
                &mut views,
            ),
            crate::object::S_OK
        );

        // 1) Dispatch with slot 0 unset, slot 1 = null (0x0), slot 2 = unmapped VA:
        //    ZERO fallback serves CBV/SRV as 0, SINK absorbs the UAV store at +1.5 MiB
        //    without faulting or dirtying ZERO.
        let list = host.new_list().expect("new_list");
        let mut state = State::default();
        state.set_heaps(&[views]);
        state.set_pipeline(pso);
        state.set_view(true, 1, 0);
        state.set_view(true, 2, mid_unmapped);
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);

        // 2) Dispatch with CBV & SRV still unbound (verifying ZERO remained 0!)
        //    and UAV bound to `uav_buf`: uav_buf[+1.5 MiB] must receive
        //    0 ^ 0 ^ 0xA5A5_5A5A = 0xA5A5_5A5A in CPU title memory.
        list.reset();
        state.reset(pso);
        state.set_heaps(&[views]);
        state.set_view(true, 2, uav_buf.addr());
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);
        // SAFETY: reading back the u32 at `+1.5 MiB` in title memory.
        let from_zero = unsafe { uav_buf.ptr.add(off).cast::<u32>().read() };
        assert_eq!(
            from_zero, 0xA5A5_5A5A,
            "sink UAV write must not dirty the zero fallback buffer"
        );

        // 3) Dispatch with all three root views bound (SRV reading +1.5 MiB and
        //    UAV writing +1.5 MiB) and verify full extent refresh + write-back.
        list.reset();
        state.reset(pso);
        state.set_heaps(&[views]);
        state.set_view(true, 0, cbv_buf.addr());
        state.set_view(true, 1, srv_buf.addr());
        state.set_view(true, 2, uav_buf.addr());
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);
        // SAFETY: reading back the u32 at `+1.5 MiB` in title memory.
        let got = unsafe { uav_buf.ptr.add(off).cast::<u32>().read() };
        assert_eq!(got, 0x1357_2468 ^ 0xDEAD_BEEF ^ 0xA5A5_5A5A);

        // 4) Indirect dispatch (`execute_indirect`) with Root UAVs bound also
        //    emits its UAV barrier cleanly and copies updated writes back.
        let args_buf = Allocation::watched(64 << 10);
        crate::shadow::register(args_buf.ptr as usize, args_buf.addr(), 64 << 10);
        // SAFETY: updating CBV[0] and writing [1, 1, 1] dispatch args inside committed allocations.
        unsafe {
            cbv_buf.ptr.cast::<u32>().write(0x2468_ACE0);
            args_buf.ptr.cast::<[u32; 3]>().write([1, 1, 1]);
        }
        list.reset();
        state.reset(pso);
        state.set_heaps(&[views]);
        state.set_view(true, 0, cbv_buf.addr());
        state.set_view(true, 1, srv_buf.addr());
        state.set_view(true, 2, uav_buf.addr());
        assert!(state.execute_indirect(&list, (2, 12, 1), args_buf.addr(), None));
        list.close();
        host.execute(&[&list]);
        // SAFETY: reading back the u32 at `+1.5 MiB` in title memory.
        let indirect_got = unsafe { uav_buf.ptr.add(off).cast::<u32>().read() };
        assert_eq!(indirect_got, 0x2468_ACE0 ^ 0xDEAD_BEEF ^ 0xA5A5_5A5A);
        crate::shadow::unregister(args_buf.ptr as usize, args_buf.addr());

        crate::shadow::unregister(srv_buf.ptr as usize, srv_buf.addr());
        crate::shadow::unregister(uav_buf.ptr as usize, uav_buf.addr());
        crate::shadow::unregister(cbv_buf.ptr as usize, cbv_buf.addr());
        let release =
            |o: This| crate::object::slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(views), 0);
        assert_eq!(release(pso), 0);
    }

    #[test]
    fn bounded_root_uav_drops_out_of_bounds_writes_without_faulting() {
        let Some(host) = crate::host::host() else {
            return;
        };
        // Synthesize a CS with 1 Root UAV (`slot=0`, user_data `0..2`):
        // writes 0xDEAD_0001 at `+0xFFFF_B940` (OOB wrapping), 0xDEAD_0002 at `+0x20_0000` (past 64 KiB buffer),
        // and 0x1234_5678 at `+0x100` (in bounds).
        let s_mov_b32 =
            |sdst: u8, src0: u8| 0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32;
        let vop1 = |op: u16, vdst: u8, src0: u16| -> u32 {
            0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
        };
        let buffer_store_dword_offen = |vdata: u8, vaddr: u8, srsrc: u8| -> [u32; 2] {
            [
                0x38 << 26 | 0x1C << 18 | (1 << 12),
                0x80 << 24
                    | ((srsrc as u32 >> 2) & 0x1F) << 16
                    | (vdata as u32) << 8
                    | vaddr as u32,
            ]
        };

        let mut words = Vec::new();
        words.push(s_mov_b32(4, 0));
        words.push(s_mov_b32(5, 1));
        words.push(s_mov_b32(6, 0xC1));
        words.extend_from_slice(&[s_mov_b32(7, 0xFF), 0x3101_4FAC]);
        // 1) OOB wrapping write at 0xFFFF_B940
        words.extend_from_slice(&[vop1(0x001, 0, 0xFF), 0xFFFF_B940]);
        words.extend_from_slice(&[vop1(0x001, 1, 0xFF), 0xDEAD_0001]);
        words.extend_from_slice(&buffer_store_dword_offen(1, 0, 4));
        // 2) OOB write past buffer end at +0x20_0000
        words.extend_from_slice(&[vop1(0x001, 0, 0xFF), 0x0020_0000]);
        words.extend_from_slice(&[vop1(0x001, 1, 0xFF), 0xDEAD_0002]);
        words.extend_from_slice(&buffer_store_dword_offen(1, 0, 4));
        // 3) Valid in-bounds write at +0x100
        words.extend_from_slice(&[vop1(0x001, 0, 0xFF), 0x0000_0100]);
        words.extend_from_slice(&[vop1(0x001, 1, 0xFF), 0x1234_5678]);
        words.extend_from_slice(&buffer_store_dword_offen(1, 0, 4));
        words.push(0xBF81_0000); // s_endpgm

        let used = vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 2,
            user_data: 0,
            slot: 0,
        }];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: used.clone(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: 2 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = victory_shader::ir::build(&native.code, false, 4).unwrap();
        let res = victory_shader::resource::track(&p, &native.layout, native.stage).unwrap();
        let io = victory_shader::io::raise(&p, &native).unwrap();
        let st = victory_shader::structure::structure(&p).unwrap();
        let c = Container {
            digest: [0; 16],
            version: (1, 0),
            total_size: 0,
            parts: Vec::new(),
        };
        let shader = victory_shader::emit::emit(&c, &native, &p, &st, &io, &res).unwrap();
        let pso = crate::pipeline::test_compute_pipeline(shader, used);

        let buf = Allocation::watched(64 << 10);
        crate::shadow::register(buf.ptr as usize, buf.addr(), 64 << 10);

        #[repr(C)]
        struct ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC,
            placement: [u64; 2],
        }
        let heap_desc = ViewHeap {
            desc: D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 8,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            },
            placement: [0; 2],
        };
        let heap_iid = ID3D12DescriptorHeap::IID;
        let mut views = std::ptr::null();
        assert_eq!(
            crate::descriptor::create_heap(
                std::ptr::null(),
                (&raw const heap_desc).cast(),
                (&raw const heap_iid).cast(),
                &mut views,
            ),
            crate::object::S_OK
        );

        let list = host.new_list().expect("new_list");
        let mut state = State::default();
        state.set_heaps(&[views]);
        state.set_pipeline(pso);
        state.set_view(true, 0, buf.addr());
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);
        assert_eq!(host.log_messages(), None);

        // SAFETY: reading back in-bounds DWORD at +0x100 written by the dispatch.
        assert_eq!(unsafe { buf.ptr.add(0x100).cast::<u32>().read() }, 0x1234_5678);

        crate::shadow::unregister(buf.ptr as usize, buf.addr());
        let release =
            |o: This| crate::object::slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(views), 0);
        assert_eq!(release(pso), 0);
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn typed_null_rtv_keeps_the_next_color_slot() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
        };
        use windows::core::s;

        struct Texture(This);

        impl Drop for Texture {
            fn drop(&mut self) {
                let release = crate::object::slot::<extern "system" fn(This) -> u32>(self.0, 0x10);
                release(self.0);
            }
        }

        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let device = &host.device;
        let list = host.new_list().expect("host command list");
        let l = &list.list;
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        let source = r#"
            [RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)")]
            float4 vs(uint id : SV_VertexID) : SV_Position {
                return float4(id == 2 ? 3 : -1, id == 1 ? -3 : 1, 0, 1);
            }
            struct Colors { float4 red : SV_Target0; float4 green : SV_Target1; };
            Colors ps() {
                Colors c;
                c.red = float4(1, 0, 0, 1);
                c.green = float4(0, 1, 0, 1);
                return c;
            }
        "#;
        let compile = |entry, target| {
            let mut code = None;
            // SAFETY: source and shader names are live, and code is our output pointer.
            unsafe {
                D3DCompile(
                    source.as_ptr().cast(),
                    source.len(),
                    s!("rtv-slot-test"),
                    None,
                    None,
                    entry,
                    target,
                    0,
                    0,
                    &mut code,
                    None,
                )
                .expect("compile fixture shader");
            }
            code.expect("fixture bytecode")
        };
        let vs = compile(s!("vs"), s!("vs_5_1"));
        let ps = compile(s!("ps"), s!("ps_5_1"));
        // SAFETY: compiled fixture shaders and their embedded root signature are live.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let bytecode = |blob: &ID3DBlob| D3D12_SHADER_BYTECODE {
                pShaderBytecode: blob.GetBufferPointer(),
                BytecodeLength: blob.GetBufferSize(),
            };
            let code = bytecode(&vs);
            let root = device
                .CreateRootSignature(
                    0,
                    std::slice::from_raw_parts(code.pShaderBytecode.cast(), code.BytecodeLength),
                )
                .expect("graphics root signature");
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code,
                PS: bytecode(&ps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 2,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            // D3D12 requires UNKNOWN in the PSO slot that binds a null RTV.
            desc.RTVFormats[1] = format;
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask =
                D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let pso = device
                .CreateGraphicsPipelineState(&desc)
                .expect("graphics pipeline");
            (root, pso)
        };

        const SIDE: u32 = 4;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(SIDE),
            Height: SIDE,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let mut target = std::ptr::null();
        let null = std::ptr::null();
        let iid = ID3D12Resource::IID;
        assert_eq!(
            crate::resource::create_committed(
                null,
                (&raw const heap).cast(),
                0,
                (&raw const desc).cast(),
                D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32,
                null.cast(),
                (&raw const iid).cast(),
                &mut target,
            ),
            crate::object::S_OK
        );
        let _target = Texture(target);
        let color = crate::resource::host_resource(target).expect("color target");
        let mut slots = [[0u64; 16]; 2];
        let handles = slots.each_mut().map(|slot| slot.as_mut_ptr() as u64);
        let null_desc = [
            format.0 as u32,
            D3D12_RTV_DIMENSION_TEXTURE2D.0 as u32,
            0,
            0,
            0,
            0,
        ];
        crate::descriptor::create_rtv(null, null, &null_desc, handles[0]);
        crate::descriptor::create_rtv(null, target, null.cast(), handles[1]);
        let rtv1_desc = crate::descriptor::rtv_record(handles[1]).and_then(|r| r.desc());
        crate::list::ensure_state(&list, target, D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32);
        // SAFETY: open list, live target and graphics pipeline.
        unsafe {
            l.ClearRenderTargetView(
                host.rtv(Some(color), rtv1_desc.as_ref()),
                &[0.0, 0.0, 1.0, 1.0],
                None,
            );
            l.SetGraphicsRootSignature(&root);
            l.SetPipelineState(&pso);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        }
        viewports(
            &list,
            &[D3D12_VIEWPORT {
                Width: SIDE as f32,
                Height: SIDE as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }],
        );
        scissors(
            &list,
            &[RECT {
                right: SIDE as i32,
                bottom: SIDE as i32,
                ..Default::default()
            }],
        );
        State::default().set_targets(&list, &handles, 0);
        // SAFETY: complete graphics state on the open list.
        unsafe { l.DrawInstanced(3, 1, 0, 0) };

        crate::list::ensure_state(&list, target, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
        let (readback, footprint) =
            crate::capture::read_back(device, l, color).expect("color readback");
        list.close();
        host.execute(&[&list]);
        let mut mapped = std::ptr::null_mut();
        let mut pixels = Vec::new();
        // SAFETY: execution has completed and all pixels lie in the mapped footprint.
        unsafe {
            readback
                .Map(0, None, Some(&mut mapped))
                .expect("map readback");
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let at = footprint.Offset as usize
                        + (y * footprint.Footprint.RowPitch + x * 4) as usize;
                    pixels.push(mapped.cast::<u8>().add(at).cast::<[u8; 4]>().read());
                }
            }
            readback.Unmap(0, None);
        }
        assert_eq!(pixels, [[0, 255, 0, 255]; (SIDE * SIDE) as usize]);
    }
}

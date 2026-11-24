//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap;
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    pub(crate) depth_bounds_supported: bool,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
    texture_uploads: Mutex<crate::texture::UploadPool>,
    /// Allocator and command list used once per submission for bindless texture
    /// uploads and initial `SHADER_RESOURCE` transitions before list preludes.
    bindless_prelude: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    /// Dedicated lists and pooled staging arenas for `Shadow` VRAM transfers,
    /// used only under `self.submitted`.
    shadow_upload_list: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    shadow_readback_list: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    shadow_upload: Mutex<StagingArena>,
    shadow_readback: Mutex<StagingArena>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
/// Host objects the title let go of while a submission may still use them
/// (a texture its descriptors name, a descriptor slot's old resource), by
/// the submission count they wait for. The title's descriptor writes and
/// releases reach the host at once, from its own threads, while the host
/// runs a submission for most of a second in the world: freeing there was
/// a GPU page fault on the freed texture (DXGI_ERROR_DEVICE_HUNG,
/// 2026-11-24, RX 7600).
static RETIRED: Mutex<Vec<(u64, Retired)>> = Mutex::new(Vec::new());
/// Submissions started and finished (waited for).
static STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FINISHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A retired object, dropped on whatever thread finishes the submission.
struct Retired(#[allow(dead_code)] Box<dyn std::any::Any>);
// SAFETY: what is retired is D3D12 objects (free-threaded) and the
// reference-counted wrappers holding them, dropped once.
unsafe impl Send for Retired {}

/// Drops `object` once every submission running now has finished: at once
/// if none is.
pub(crate) fn retire<T: 'static>(object: T) {
    let started = STARTED.load(Ordering::Acquire);
    if started <= FINISHED.load(Ordering::Acquire) {
        drop(object);
        return;
    }
    RETIRED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((started, Retired(Box::new(object))));
}

/// Drops all `objects` once every submission running now has finished, taking
/// the retirement lock at most once.
pub(crate) fn retire_batch<T: 'static>(objects: Vec<T>) {
    if objects.is_empty() {
        return;
    }
    let started = STARTED.load(Ordering::Acquire);
    if started <= FINISHED.load(Ordering::Acquire) {
        drop(objects);
        return;
    }
    let mut retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
    retired.reserve(objects.len());
    for object in objects {
        retired.push((started, Retired(Box::new(object))));
    }
}

/// Drops what waited for submissions up to `finished`.
fn bury(finished: u64) {
    let done: Vec<(u64, Retired)> = {
        let mut retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
        if retired.is_empty() {
            return;
        }
        let (done, keep) = std::mem::take(&mut *retired).into_iter().partition(|(at, _)| *at <= finished);
        *retired = keep;
        done
    };
    drop(done);
}

static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DEBUG");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if crate::switch::on("VICTORY_D3D12_GBV")
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP") {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let host = Host::from_device(device, messages)?;
        tracing::info!("host D3D12 device ready");
        Ok(host)
    }
}

impl Host {
    fn from_device(
        device: ID3D12Device,
        messages: Option<ID3D12InfoQueue>,
    ) -> windows::core::Result<Host> {
        // SAFETY: plain D3D12 queue, fence, event and descriptor heap creation.
        unsafe {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
            let depth_bounds_supported = device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS2,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
                .is_ok()
                && options.DepthBoundsTestSupported.as_bool();
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            let rtvs: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: RTV_RING,
                    ..Default::default()
                })?;
            let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            let dsvs: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                    NumDescriptors: RTV_RING,
                    ..Default::default()
                })?;
            let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
            let wave64_mode = recommended_wave64_mode(&device);
            crate::shader_cache::set_wave64_mode(wave64_mode);
            tracing::info!(?wave64_mode, "Wave64 compute lowering mode selected");
            let closed_list =
                || -> windows::core::Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)> {
                    let allocator: ID3D12CommandAllocator =
                        device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                    let list: ID3D12GraphicsCommandList = device.CreateCommandList(
                        0,
                        D3D12_COMMAND_LIST_TYPE_DIRECT,
                        &allocator,
                        None,
                    )?;
                    list.Close()?;
                    Ok((allocator, list))
                };
            let bindless_prelude = closed_list()?;
            let shadow_upload_list = closed_list()?;
            let shadow_readback_list = closed_list()?;
            Ok(Host {
                device,
                depth_bounds_supported,
                queue,
                fence,
                submitted: Mutex::new(0),
                event,
                rtvs,
                rtv_size,
                dsvs,
                dsv_size,
                next_dsv: AtomicU32::new(0),
                messages,
                next_rtv: AtomicU32::new(0),
                presenter: Mutex::new(None),
                texture_uploads: Mutex::new(crate::texture::UploadPool::new()),
                bindless_prelude,
                shadow_upload_list,
                shadow_readback_list,
                shadow_upload: Mutex::default(),
                shadow_readback: Mutex::default(),
            })
        }
    }

    /// Creates an isolated WARP-backed `Host` for unit tests without touching
    /// the process-global host singleton.
    #[cfg(test)]
    pub(crate) fn new_warp() -> Option<Host> {
        let adapter = warp_adapter().ok()?;
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: `adapter` is a verified software WARP adapter.
        unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()? };
        Host::from_device(device?, None).ok()
    }
}

/// Queries D3D12 wave lane capabilities and chooses the Wave64 lowering mode.
fn recommended_wave64_mode(device: &ID3D12Device) -> victory_shader::emit::Wave64Mode {
    use victory_shader::emit::Wave64Mode;
    let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: plain D3D12 feature query with valid struct size.
    let ok = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            (&raw mut opts).cast(),
            size_of_val(&opts) as u32,
        )
    }
    .is_ok();
    if !ok || !opts.WaveOps.as_bool() {
        return Wave64Mode::Emulate;
    }
    if opts.WaveLaneCountMin <= 64 && opts.WaveLaneCountMax >= 64 {
        Wave64Mode::Native
    } else if opts.WaveLaneCountMin <= 32 && opts.WaveLaneCountMax >= 32 {
        Wave64Mode::Wave32
    } else {
        Wave64Mode::Emulate
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sparse_copies_preserve_islands_and_observe_remaps() {
        use windows_sys::Win32::System::Memory::*;
        const N: usize = 5 << 16;
        // SAFETY: we reserve a private region, commit selected pages inside it,
        // and only directly access the pages while they are writable.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let mut expected = vec![0; N];
            for (offset, value) in [(0, 9), (2 << 16, 37), (N - 4096, 85)] {
                let at = base.add(offset);
                assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                at.write_bytes(value, 4096);
                expected[offset..offset + 4096].fill(value);
            }
            for remap in [false, true] {
                if remap {
                    let mut old = 0;
                    assert_ne!(VirtualProtect(base.cast(), 4096, PAGE_NOACCESS, &mut old), 0);
                    expected[..4096].fill(0);
                    let at = base.add(1 << 16);
                    assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                    at.write_bytes(0xc7, 4096);
                    expected[1 << 16..(1 << 16) + 4096].fill(0xc7);
                }
                let mut out = vec![0xee; N];
                super::copy_mapped(out.as_mut_ptr().add(27), base.add(27), N - 54, false);
                assert_eq!(out[27..N - 27], expected[27..N - 27]);
                assert!(out[..27].iter().chain(&out[N - 27..]).all(|&v| v == 0xee));
            }
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    use super::*;

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    /// A node of the device-removed data is a list by pointer or by name.
    /// The name is what still matches when the runtime records another
    /// object than the one we hold (the debug layer wraps ours), and a
    /// different list or an unnamed node never matches.
    #[test]
    fn a_breadcrumb_node_is_found_by_pointer_or_by_name() {
        let ours = [0x1000, 0x1008];
        let name = list_name(7);
        let is = |pointer, text| node_is_list(pointer, text, &ours, &name);
        assert_eq!(name, "victory host list 7");
        assert!(is(Some(0x1000), None));
        assert!(is(Some(0x1008), Some("other")));
        assert!(is(Some(0x9999), Some("victory host list 7")));
        assert!(is(None, Some("victory host list 7")));
        assert!(!is(Some(0x9999), Some("victory host list 70")));
        assert!(!is(Some(0x9999), None));
        assert!(!is(None, None));
        assert_eq!(list_name(70), "victory host list 70");
    }

    /// The pointers of one list: the interface pointers it answers with
    /// are all there, and another list's are not.
    #[test]
    fn a_list_knows_every_pointer_it_answers_with() {
        // SAFETY: plain device and list creation on the software adapter.
        unsafe {
            let Ok(adapter) = warp_adapter() else { return };
            let mut device: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                return;
            }
            let device = device.unwrap();
            // An open list holds its allocator: one each.
            let make = || -> ID3D12GraphicsCommandList {
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .unwrap()
            };
            let (a, b) = (make(), make());
            let pointers = identities(&a);
            assert!(pointers.contains(&(a.as_raw() as usize)));
            let unknown = a.cast::<windows::core::IUnknown>().unwrap();
            assert!(pointers.contains(&(unknown.as_raw() as usize)));
            let command_list = a.cast::<ID3D12CommandList>().unwrap();
            assert!(pointers.contains(&(command_list.as_raw() as usize)));
            let other = identities(&b);
            assert!(pointers.iter().all(|p| !other.contains(p)));
            // As a node: the same object by any of its pointers; a node
            // naming the other list by pointer and our name still matches.
            let name = list_name(3);
            let is = |pointer, text| node_is_list(pointer, text, &pointers, &name);
            assert!(is(Some(command_list.as_raw() as usize), None));
            assert!(!is(Some(b.as_raw() as usize), None));
            assert!(is(Some(b.as_raw() as usize), Some(&name)));
        }
    }

    /// A node as the runtime gives it, built by hand: `HostList::is_node`
    /// reads the pointer and the copied name, and the name alone finds a
    /// list the runtime records under another object.
    #[test]
    fn is_node_reads_the_name_the_runtime_copied() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let mut wide: Vec<u16> = list_name(list.id).encode_utf16().collect();
        wide.push(0);
        let mut node = D3D12_AUTO_BREADCRUMB_NODE::default();
        assert!(!list.is_node(&node), "an empty node is no list");
        node.pCommandListDebugNameW = windows::core::PCWSTR(wide.as_ptr());
        assert!(list.is_node(&node), "found by its name alone");
        let mut other: Vec<u16> = list_name(list.id + 1).encode_utf16().collect();
        other.push(0);
        node.pCommandListDebugNameW = windows::core::PCWSTR(other.as_ptr());
        assert!(!list.is_node(&node), "another list's name is no match");
    }

    #[test]
    fn buffer_states_transition_between_writes_and_reads() {
        let mut s = BufferStates::default();
        // First uses: the prelude's.
        assert_eq!(s.update(1, 'a', BUFFER_WRITE), None);
        assert_eq!(s.update(2, 'b', BUFFER_READ), None);
        assert_eq!(s.update(1, 'a', BUFFER_WRITE), None);
        assert_eq!(s.update(1, 'a', BUFFER_READ), Some(BUFFER_WRITE));
        assert_eq!(s.update(1, 'a', BUFFER_READ), None);
        assert_eq!(s.update(1, 'a', BUFFER_WRITE), Some(BUFFER_READ));
        let sorted = |mut v: Vec<(char, u32)>| {
            v.sort_unstable();
            v
        };
        let firsts = sorted(s.firsts().map(|(r, f)| (*r, f)).collect());
        assert_eq!(firsts, [('a', BUFFER_WRITE), ('b', BUFFER_READ)]);
        let lasts = sorted(s.lasts().map(|(r, l)| (*r, l)).collect());
        assert_eq!(lasts, [('a', BUFFER_WRITE), ('b', BUFFER_READ)]);
        s.clear();
        assert_eq!(s.firsts().count(), 0);
    }

    /// Set in the child process `in_child` starts.
    const CHILD: &str = "VICTORY_TEST_DEBUG_LAYER_CHILD";

    /// Runs the test `name` of this module alone in a child process and
    /// checks that it passed; false in that child. The debug layer is
    /// process-wide and must be on before the host device is made.
    fn in_child(name: &str) -> bool {
        if std::env::var_os(CHILD).is_some() {
            return false;
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([&format!("host::tests::{name}"), "--exact", "--nocapture"])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        eprintln!("{text}");
        assert!(out.status.success(), "the child test failed");
        assert!(text.contains("1 passed"), "the child ran no test");
        true
    }

    fn compile(source: &str, entry: windows::core::PCSTR, target: windows::core::PCSTR) -> Vec<u8> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
        // SAFETY: the source and names live through the call; out pointers
        // ours; the blobs' own bytes are read.
        unsafe {
            let made = D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                None,
                None,
                None,
                entry,
                target,
                0,
                0,
                &mut code,
                Some(&mut errors),
            );
            if let Some(e) = &errors {
                let text = std::slice::from_raw_parts(
                    e.GetBufferPointer().cast::<u8>(),
                    e.GetBufferSize(),
                );
                assert!(made.is_ok(), "{}", String::from_utf8_lossy(text));
            }
            made.unwrap();
            let code = code.unwrap();
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
                .to_vec()
        }
    }

    /// A buffer a shader writes and draws read as vertices, as the world's
    /// 174 MB heap is (#199, #264): the argument clamp writes it and
    /// indirect and plain draws read it, in two lists submitted together.
    /// The debug layer checks every barrier's state before against what
    /// the lists did (id 527, which #264 reported); it does not check
    /// vertex buffer states at draws on every system, so this cannot show
    /// #199's vertex buffer in unordered access.
    #[test]
    fn debug_layer_accepts_buffers_written_then_read_as_vertices() {
        use windows::core::s;
        if in_child("debug_layer_accepts_buffers_written_then_read_as_vertices") {
            return;
        }
        let mut layer: Option<ID3D12Debug> = None;
        // SAFETY: plain queries, before any device exists in this process.
        if unsafe { D3D12GetDebugInterface(&mut layer) }.is_err() {
            eprintln!("no D3D12 debug layer; skipped");
            return;
        }
        // SAFETY: as above.
        unsafe { layer.unwrap().EnableDebugLayer() };
        let Some(host) = host() else {
            eprintln!("no host device; skipped");
            return;
        };
        let messages: ID3D12InfoQueue = host.device.cast().unwrap();

        const SOURCE: &str = r#"
            #define RS "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)"
            [RootSignature(RS)] float4 vs(float4 p : POSITION) : SV_Position { return p; }
            [RootSignature(RS)] float4 ps() : SV_Target { return float4(1, 0, 0, 1); }
        "#;
        let (vs, ps) = (
            compile(SOURCE, s!("vs"), s!("vs_5_1")),
            compile(SOURCE, s!("ps"), s!("ps_5_1")),
        );
        let element = D3D12_INPUT_ELEMENT_DESC {
            SemanticName: s!("POSITION"),
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            ..Default::default()
        };
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let stencil = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        // SAFETY: plain object creation from valid descriptions; the
        // shader bytes outlive the calls.
        let (root, pso, target) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &vs).unwrap();
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: vs.as_ptr().cast(),
                    BytecodeLength: vs.len(),
                },
                PS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: ps.as_ptr().cast(),
                    BytecodeLength: ps.len(),
                },
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    FrontFace: stencil,
                    BackFace: stencil,
                    ..Default::default()
                },
                InputLayout: D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: &element,
                    NumElements: 1,
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            let pso: ID3D12PipelineState = host.device.CreateGraphicsPipelineState(&desc).unwrap();
            let mut target: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: 4,
                        Height: 4,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut target,
                )
                .unwrap();
            (root, pso, target.unwrap())
        };

        // Three vertices at 0, a draw's arguments (3 vertices, 1 instance)
        // at 256: one buffer, as title memory is.
        let (buffer, data) = host.shared_buffer(1 << 16).unwrap();
        let vertices: [f32; 12] = [
            -1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0,
        ];
        // SAFETY: inside the mapping.
        unsafe {
            std::ptr::copy_nonoverlapping(vertices.as_ptr().cast::<u8>(), data, 48);
            std::ptr::copy_nonoverlapping([3u32, 1, 0, 0].as_ptr().cast::<u8>(), data.add(256), 16);
        }
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let rtv = host.rtv(Some(&target), None);
        // SAFETY: one draw argument, no root arguments.
        let signature: ID3D12CommandSignature = unsafe {
            let argument = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
                ..Default::default()
            };
            let mut signature = None;
            host.device
                .CreateCommandSignature(
                    &D3D12_COMMAND_SIGNATURE_DESC {
                        ByteStride: 16,
                        NumArgumentDescs: 1,
                        pArgumentDescs: &argument,
                        NodeMask: 0,
                    },
                    None::<&ID3D12RootSignature>,
                    &mut signature,
                )
                .unwrap();
            signature.unwrap()
        };
        // The draw state, with the buffer's vertices bound as `draw.rs`
        // binds them.
        let bind = |list: &HostList| {
            list.buffer_state(&buffer, BUFFER_READ);
            // SAFETY: an open list; our objects, alive until the lists ran.
            unsafe {
                let l = &list.list;
                l.SetGraphicsRootSignature(&root);
                l.SetPipelineState(&pso);
                l.OMSetRenderTargets(1, Some(&rtv), false, None);
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    Width: 4.0,
                    Height: 4.0,
                    MaxDepth: 1.0,
                    ..Default::default()
                }]);
                l.RSSetScissorRects(&[RECT {
                    right: 4,
                    bottom: 4,
                    ..Default::default()
                }]);
                l.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                );
                l.IASetVertexBuffers(
                    0,
                    Some(&[D3D12_VERTEX_BUFFER_VIEW {
                        BufferLocation: gpu,
                        SizeInBytes: 48,
                        StrideInBytes: 16,
                    }]),
                );
            }
        };
        let draw = |list: &HostList| {
            bind(list);
            // SAFETY: an open list with the draw state set.
            unsafe { list.list.DrawInstanced(3, 1, 0, 0) };
        };
        // As `draw.rs` executes indirect draws: clamp, bind, execute, done.
        let indirect = |list: &HostList| {
            let bounded =
                crate::indirect::bound(list, (&buffer, 256), None, (0, 16, 1)).unwrap();
            bind(list);
            // SAFETY: open list with draw state bound and scratch buffer in INDIRECT_ARGUMENT.
            unsafe { list.list.ExecuteIndirect(&signature, 1, &bounded, 0, None, 0) };
            crate::indirect::done(list, &bounded);
        };
        // A list that draws, draws indirectly, draws; then one that starts
        // with an indirect draw, submitted together.
        let (a, b) = (host.new_list().unwrap(), host.new_list().unwrap());
        draw(&a);
        indirect(&a);
        draw(&a);
        a.close();
        indirect(&b);
        draw(&b);
        b.close();
        // SAFETY: plain queries.
        unsafe { messages.ClearStoredMessages() };
        host.execute(&[&a, &b]);

        let mut errors = Vec::new();
        // SAFETY: each message is read into a buffer of the size the queue
        // reports.
        unsafe {
            for i in 0..messages.GetNumStoredMessages() {
                let mut size = 0;
                if messages.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut bytes = vec![0u64; size.div_ceil(8)];
                let m = bytes.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if messages.GetMessage(i, Some(m), &mut size).is_ok()
                    && (*m).Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0
                {
                    let text =
                        std::slice::from_raw_parts((*m).pDescription, (*m).DescriptionByteLength);
                    errors.push(format!(
                        "id {}: {}",
                        (*m).ID.0,
                        String::from_utf8_lossy(text)
                    ));
                }
            }
        }
        assert!(errors.is_empty(), "debug layer errors: {errors:#?}");
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) depth_bounds: Option<ID3D12GraphicsCommandList1>,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Memory shaders write indirect arguments to, reused after a reset.
    arguments: Mutex<Arguments>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// The states of the host copies of title memory the list uses.
    buffers: Mutex<BufferStates<ID3D12Resource>>,
    /// Depth targets the list gave unordered access to since it last made
    /// them current: their twins hold their contents until copied back.
    twins: Mutex<Vec<HostCopy>>,
    /// The list that runs those transitions just before this one.
    prelude: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// Which list this is, for the name that tells it apart in device-removed
    /// data (see [`HostList::is_node`]).
    id: u64,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    clear_uav_cpu: OnceLock<Option<ID3D12DescriptorHeap>>,
    /// Device-local scratch buffer for clamped indirect arguments (`indirect`),
    /// grown to the largest indirect execution recorded on the list.
    indirect_args: Mutex<Option<(ID3D12Resource, u64)>>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

pub(crate) fn warn_depth_bounds() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| tracing::warn!("the host GPU has no depth bounds test; depth bounds are ignored"));
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

/// Every read of a host copy of title memory (vertex, index, constant and
/// shader-resource data, indirect arguments).
pub(crate) const BUFFER_READ: u32 = D3D12_RESOURCE_STATE_GENERIC_READ.0 as u32;
/// Shader writes to a host copy of title memory.
pub(crate) const BUFFER_WRITE: u32 = D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32;

/// The states of the host copies of title memory (buffers) one list uses,
/// by copy: the state it starts in and the one it is in now. A buffer is in
/// the common state between lists: the list's prelude moves it to its first
/// state and the list moves it back before it closes. A buffer a shader
/// writes and a later draw reads as vertices (#199, #264) is transitioned in
/// between; the implicit promotion from common that it relied on allows
/// either use in a list, not both.
///
/// shortcut: buffers read only through directly indexed heaps are not
/// tracked, and a command that both writes a buffer and reads it as
/// vertices, indices or arguments leaves it in the read state.
pub(crate) struct BufferStates<R> {
    by_copy: std::collections::HashMap<usize, (u32, u32, R)>,
}

impl<R> Default for BufferStates<R> {
    fn default() -> Self {
        BufferStates {
            by_copy: std::collections::HashMap::new(),
        }
    }
}

impl<R> BufferStates<R> {
    /// The list uses `copy` (`resource`) in `wanted`: the state it leaves,
    /// if a transition must be recorded now. A first use records nothing:
    /// the prelude moves the buffer there.
    pub(crate) fn update(&mut self, copy: usize, resource: R, wanted: u32) -> Option<u32> {
        match self.by_copy.entry(copy) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, resource));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        }
    }

    /// The prelude's transitions: each buffer from common to its first state.
    pub(crate) fn firsts(&self) -> impl Iterator<Item = (&R, u32)> {
        self.by_copy.values().map(|(first, _, r)| (r, *first))
    }

    /// The list's closing transitions: each buffer from its last state back
    /// to common.
    pub(crate) fn lasts(&self) -> impl Iterator<Item = (&R, u32)> {
        self.by_copy.values().map(|(_, last, r)| (r, *last))
    }

    pub(crate) fn clear(&mut self) {
        self.by_copy.clear();
    }
}

/// A title texture's host copy, as lists hold it.
type HostCopy = std::sync::Arc<crate::resource::HostTexture>;

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DRED"))
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMING"))
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// The id of the next host list (see [`HostList::is_node`]).
static NEXT_LIST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The name a host list gets with device-removed data on.
fn list_name(id: u64) -> String {
    format!("victory host list {id}")
}

/// The pointers one of our list objects answers to: its own and the one each
/// interface a runtime may record it by gives (COM identity is the
/// `IUnknown` one). Only our live object is queried: a node of the
/// device-removed data may name a list that is long released, so its pointer
/// is compared and never dereferenced.
fn identities(list: &impl Interface) -> Vec<usize> {
    let mut found = vec![list.as_raw() as usize];
    let mut add = |pointer: Option<*mut core::ffi::c_void>| {
        if let Some(pointer) = pointer {
            found.push(pointer as usize);
        }
    };
    add(list
        .cast::<windows::core::IUnknown>()
        .ok()
        .map(|i| i.as_raw()));
    add(list.cast::<ID3D12Object>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12DeviceChild>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12CommandList>().ok().map(|i| i.as_raw()));
    found.sort_unstable();
    found.dedup();
    found
}

/// Whether a node of the device-removed data is a list: by pointer, or by the
/// name the runtime copied when the list was submitted. With the debug layer
/// on, the object we hold is its wrapper and the runtime records another
/// one, so no pointer of ours matches; the name still does.
fn node_is_list(
    node_pointer: Option<usize>,
    node_name: Option<&str>,
    pointers: &[usize],
    name: &str,
) -> bool {
    node_pointer.is_some_and(|p| pointers.contains(&p)) || node_name == Some(name)
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(ID3D12Resource, usize, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

/// Chunks of [`ARGUMENT_CHUNK`] bytes, the first `filled` in use, the last
/// of them up to `used`.
#[derive(Default)]
struct Arguments {
    chunks: Vec<ID3D12Resource>,
    filled: usize,
    used: u64,
}

// SAFETY: D3D12 resources are free-threaded; used under the list's lock.
unsafe impl Send for Arguments {}

const ARGUMENT_CHUNK: u64 = 1 << 16;

const SCRATCH_CHUNK: u64 = 4 << 20;

/// Pooled staging chunks (4 MiB each) reused across submissions for `Shadow`
/// `UPLOAD` and `READBACK` transfers; oversized allocations live for one
/// submission and are released on reset.
#[derive(Clone, Copy)]
enum StagingKind {
    Upload,
    Readback,
}

#[derive(Default)]
struct StagingArena {
    chunks: Vec<(ID3D12Resource, usize, u64)>,
    current: usize,
    used: u64,
    transient: Vec<(ID3D12Resource, usize, u64)>,
}

// SAFETY: pointers are persistent mappings of D3D12 upload/readback buffers
// accessed only while holding the arena's mutex under `Host::submitted`.
unsafe impl Send for StagingArena {}

const STAGING_CHUNK: u64 = 4 << 20;
const MAX_POOLED_STAGING_CHUNKS: usize = 4;

impl StagingArena {
    fn alloc(
        &mut self,
        host: &Host,
        kind: StagingKind,
        len: usize,
    ) -> Option<(ID3D12Resource, u64, *mut u8)> {
        let alloc_len = (len as u64).max(1).next_multiple_of(16);
        let make = |size: u64| match kind {
            StagingKind::Upload => {
                let (buf, ptr) = host.upload_buffer(size)?;
                name(&buf, || format!("shadow upload staging {size:#x}"));
                Some((buf, ptr))
            }
            StagingKind::Readback => {
                let (buf, ptr) = host.readback_buffer(size)?;
                name(&buf, || format!("shadow readback staging {size:#x}"));
                Some((buf, ptr))
            }
        };
        if alloc_len > STAGING_CHUNK {
            let (buf, ptr) = make(alloc_len)?;
            self.transient.push((buf.clone(), ptr as usize, alloc_len));
            return Some((buf, 0, ptr));
        }
        if self.used + alloc_len > STAGING_CHUNK && self.current < self.chunks.len() {
            (self.current, self.used) = (self.current + 1, 0);
        }
        if self.current == self.chunks.len() {
            let (buf, ptr) = make(STAGING_CHUNK)?;
            self.chunks.push((buf, ptr as usize, STAGING_CHUNK));
            self.used = 0;
        }
        let (buf, data, _) = &self.chunks[self.current];
        let at = self.used;
        self.used += alloc_len;
        Some((buf.clone(), at, (*data + at as usize) as *mut u8))
    }

    fn reset(&mut self) {
        self.current = 0;
        self.used = 0;
        self.transient.clear();
        self.chunks.truncate(MAX_POOLED_STAGING_CHUNKS);
    }
}

/// One merged `Shadow` range to read back before and after the submission.
struct ReadbackPlan {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    from: usize,
    len: usize,
    before_res: ID3D12Resource,
    before_off: u64,
    before_ptr: *const u8,
    after_res: ID3D12Resource,
    after_off: u64,
    after_ptr: *const u8,
}

/// Default-heap UAV chunks for clamped indirect arguments, always in
/// `D3D12_RESOURCE_STATE_UNORDERED_ACCESS` between indirect executions and
/// reused after a reset.
/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date and, when dirty, copies
    /// it into the host texture on the list's prelude: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(
        std::sync::Arc<crate::texture::Staging>,
        std::sync::Arc<crate::resource::HostTexture>,
    ),
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// View heaps the list's draws index directly (`ResourceDescriptorHeap[]`),
    /// and whether any of their shaders may write memory: their slots are
    /// brought up to date once per submission, after every other step.
    bindless: Vec<(std::sync::Arc<crate::descriptor::host::HostHeap>, bool)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// Host `D3D12_FEATURE_DATA_D3D12_OPTIONS5` (cached once per device).
    pub(crate) fn options5(&self) -> D3D12_FEATURE_DATA_D3D12_OPTIONS5 {
        static OPTIONS5: OnceLock<D3D12_FEATURE_DATA_D3D12_OPTIONS5> = OnceLock::new();
        *OPTIONS5.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
            // SAFETY: writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS5,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            if ok {
                tracing::info!(
                    tier = options.RaytracingTier.0,
                    render_passes = options.RenderPassesTier.0,
                    "host raytracing and render-pass support"
                );
                options
            } else {
                D3D12_FEATURE_DATA_D3D12_OPTIONS5::default()
            }
        })
    }

    /// Host DXR tier (`0` = unsupported, `10` = Tier 1.0, `11` = Tier 1.1).
    pub(crate) fn raytracing_tier(&self) -> u32 {
        self.options5().RaytracingTier.0 as u32
    }

    /// `ID3D12Device5` for DXR 1.1 calls when the host supports at least Tier
    /// 1.0 raytracing.
    pub(crate) fn device5(&self) -> Option<ID3D12Device5> {
        if self.raytracing_tier() < D3D12_RAYTRACING_TIER_1_0.0 as u32 {
            return None;
        }
        self.device.cast::<ID3D12Device5>().ok()
    }

    /// A `DEFAULT`-heap UAV-capable buffer (used for DXR acceleration
    /// structures and scratch buffers), aligned up to 256 bytes.
    pub(crate) fn default_uav_buffer(
        &self,
        size: u64,
        initial: D3D12_RESOURCE_STATES,
    ) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(256).next_multiple_of(256),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain `DEFAULT` buffer creation.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                initial,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no default UAV buffer");
                None
            }
        }
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude_allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &prelude_allocator,
                None,
            )?;
            prelude.Close()?;
            let id = NEXT_LIST.fetch_add(1, Ordering::Relaxed);
            name(&list, || list_name(id));
            Ok(HostList {
                allocator,
                depth_bounds: self.depth_bounds_supported.then(|| list.cast()).transpose()?,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                arguments: Mutex::default(),
                states: Mutex::default(),
                buffers: Mutex::default(),
                twins: Mutex::default(),
                prelude: (prelude_allocator, prelude),
                pipelines: Mutex::default(),
                id,
                budget: OnceLock::new(),
                clear_uav_cpu: OnceLock::new(),
                indirect_args: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// Runs closed lists and waits for them.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let this_submission = STARTED.fetch_add(1, Ordering::AcqRel) + 1;
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut upload_pool = self.texture_uploads.lock().unwrap_or_else(|e| e.into_inner());
        let mut upload_arena = self.shadow_upload.lock().unwrap_or_else(|e| e.into_inner());
        let mut readback_arena = self
            .shadow_readback
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        #[cfg(feature = "diagnostics")]
        let (mut deferred_us, mut prelude_us, mut tex_uploads) = (0u64, 0u64, 0usize);
        let (
            mut list_uploads,
            bindless_prelude,
            bindless_timings,
            _plans,
            batch_uploads,
            back,
        ) = {
            // The copies below see title memory as at this submission: each
            // page's written state is asked of the kernel once for all of them
            // (again after a step that writes title memory).
            let sweep = crate::watch::Sweep::begin();
            let mut batch = Batch::default();
            let mut list_uploads = Vec::with_capacity(runs.len());
            for list in runs {
                #[cfg(feature = "diagnostics")]
                let t0 = std::time::Instant::now();
                let uploads =
                    list.run_deferred(&mut batch, self, &mut upload_pool, &mut upload_arena);
                #[cfg(feature = "diagnostics")]
                {
                    deferred_us += t0.elapsed().as_micros() as u64;
                    tex_uploads += uploads.len();
                }
                list_uploads.push(uploads);
            }
            let (bindless_prelude, bindless_timings) = batch.prepare_bindless(
                self,
                &mut upload_pool,
                &mut upload_arena,
                &mut list_uploads,
            );
            drop(sweep);
            #[cfg(feature = "diagnostics")]
            {
                tex_uploads += bindless_timings.uploads;
            }
            (
                list_uploads,
                bindless_prelude,
                bindless_timings,
                batch.plans,
                batch.uploads,
                batch.copy_back,
            )
        };
        // Merge shader-written ranges per shadow buffer (preserving first-declaration
        // order across shadows) and stage readback slots for pre-submission ("before")
        // and post-submission ("after") bytes.
        let mut readback_plans: Vec<ReadbackPlan> = Vec::new();
        let mut pending_shadows: Vec<std::sync::Arc<crate::shadow::Shadow>> = Vec::new();
        type ShadowRanges = (std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>);
        let mut grouped: Vec<ShadowRanges> = Vec::new();
        for (shadow, address, len) in back {
            if !pending_shadows
                .iter()
                .any(|s| std::sync::Arc::ptr_eq(s, &shadow))
            {
                pending_shadows.push(shadow.clone());
            }
            let (from, to) = shadow.inside(address, len);
            if from >= to {
                continue;
            }
            match grouped
                .iter_mut()
                .find(|(s, _)| std::sync::Arc::ptr_eq(s, &shadow))
            {
                Some((_, ranges)) => ranges.push((from, to)),
                None => grouped.push((shadow, vec![(from, to)])),
            }
        }
        for (shadow, mut ranges) in grouped {
            crate::shadow::merge_ranges(&mut ranges);
            for (from, to) in ranges {
                let len = to - from;
                if let (
                    Some((before_res, before_off, before_ptr)),
                    Some((after_res, after_off, after_ptr)),
                ) = (
                    readback_arena.alloc(self, StagingKind::Readback, len),
                    readback_arena.alloc(self, StagingKind::Readback, len),
                ) {
                    readback_plans.push(ReadbackPlan {
                        shadow: shadow.clone(),
                        from,
                        len,
                        before_res,
                        before_off,
                        before_ptr: before_ptr.cast_const(),
                        after_res,
                        after_off,
                        after_ptr: after_ptr.cast_const(),
                    });
                }
            }
        }
        let upload_list: Option<ID3D12CommandList> =
            if !batch_uploads.is_empty() || !readback_plans.is_empty() {
                let (alloc, list) = &self.shadow_upload_list;
                // SAFETY: previous submission has finished; all resources live through wait.
                let recorded = unsafe {
                    alloc.Reset().and_then(|()| {
                        list.Reset(alloc, None)?;
                        let mut uploaded = std::collections::HashSet::new();
                        for u in &batch_uploads {
                            list.CopyBufferRegion(
                                &u.shadow.buffer,
                                u.dst_offset,
                                &u.src,
                                u.src_offset,
                                u.len,
                            );
                            uploaded.insert(u.shadow.buffer.as_raw() as usize);
                        }
                        if !readback_plans.is_empty() {
                            let mut transitioned = std::collections::HashSet::new();
                            let mut barriers = Vec::new();
                            for r in &readback_plans {
                                let raw = r.shadow.buffer.as_raw() as usize;
                                if uploaded.contains(&raw) && transitioned.insert(raw) {
                                    barriers.push(transition(
                                        &r.shadow.buffer,
                                        D3D12_RESOURCE_STATE_COPY_DEST,
                                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                                    ));
                                }
                            }
                            if !barriers.is_empty() {
                                list.ResourceBarrier(&barriers);
                            }
                            for r in &readback_plans {
                                list.CopyBufferRegion(
                                    &r.before_res,
                                    r.before_off,
                                    &r.shadow.buffer,
                                    r.from as u64,
                                    r.len as u64,
                                );
                            }
                        }
                        list.Close()
                    })
                };
                if let Err(e) = recorded {
                    fail(format_args!("shadow upload prelude did not record ({e})"));
                }
                list.cast().ok()
            } else {
                None
            };
        let readback_list: Option<ID3D12CommandList> = if !readback_plans.is_empty() {
            let (alloc, list) = &self.shadow_readback_list;
            // SAFETY: previous submission has finished; all resources live through wait.
            let recorded = unsafe {
                alloc.Reset().and_then(|()| {
                    list.Reset(alloc, None)?;
                    for r in &readback_plans {
                        list.CopyBufferRegion(
                            &r.after_res,
                            r.after_off,
                            &r.shadow.buffer,
                            r.from as u64,
                            r.len as u64,
                        );
                    }
                    list.Close()
                })
            };
            if let Err(e) = recorded {
                fail(format_args!("shadow readback list did not record ({e})"));
            }
            list.cast().ok()
        } else {
            None
        };
        if let Some(prelude) = bindless_prelude {
            lists.push(Some(prelude));
        }
        for (i, (list, uploads)) in runs.iter().zip(list_uploads.drain(..)).enumerate() {
            firsts.push(if i == 0 { 0 } else { lists.len() });
            #[cfg(feature = "diagnostics")]
            let t1 = std::time::Instant::now();
            if let Some(prelude) = list.prelude(&uploads) {
                lists.push(Some(prelude));
            }
            #[cfg(feature = "diagnostics")]
            {
                prelude_us += t1.elapsed().as_micros() as u64;
            }
            lists.push(list.list.cast().ok());
        }
        let _ = bindless_timings;
        let prepared = std::time::Instant::now();
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timing() {
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else {
            vec![&lists[..]]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [&lists[..]];
        // Run shadow uploads in a separate ExecuteCommandLists call so all
        // default-heap buffers decay cleanly back to COMMON before the title's
        // lists run (no fence wait needed between calls on the same queue).
        if let Some(up) = upload_list {
            // SAFETY: closed list; all resources stay alive until after the fence wait.
            unsafe { self.queue.ExecuteCommandLists(&[Some(up)]) };
        }
        let wait_queue = |submitted: &mut u64| {
            *submitted += 1;
            // SAFETY: the fence and event are ours.
            unsafe {
                let waited = self
                    .queue
                    .Signal(&self.fence, *submitted)
                    .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
                match waited {
                    Ok(()) => {
                        WaitForSingleObject(self.event, INFINITE);
                    }
                    Err(e) => tracing::warn!(error = %e, "host submission not waited for"),
                }
            }
        };
        if groups.len() == 1 {
            // SAFETY: closed lists; separate ExecuteCommandLists calls let buffers
            // decay to COMMON between runs and readback with a single fence wait.
            unsafe {
                if !groups[0].is_empty() {
                    self.queue.ExecuteCommandLists(groups[0]);
                }
                if let Some(rb) = readback_list {
                    self.queue.ExecuteCommandLists(&[Some(rb)]);
                }
            }
            wait_queue(&mut submitted);
        } else {
            for (i, group) in groups.iter().enumerate() {
                let started = std::time::Instant::now();
                // SAFETY: closed lists.
                unsafe { self.queue.ExecuteCommandLists(group) };
                wait_queue(&mut submitted);
                #[cfg(feature = "diagnostics")]
                if timing() && started.elapsed() > std::time::Duration::from_millis(50) {
                    // SAFETY: a plain query.
                    let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                    let commands = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                    tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                    for (ms, op) in runs[i].slowest(frequency, 6) {
                        tracing::warn!("slow list command, {ms:.1} ms: {op}");
                    }
                }
                let _ = (i, started);
            }
            if let Some(rb) = readback_list {
                // SAFETY: closed list.
                unsafe { self.queue.ExecuteCommandLists(&[Some(rb)]) };
                wait_queue(&mut submitted);
            }
        }
        let _ = &firsts;
        FINISHED.store(this_submission, Ordering::Release);
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        bury(this_submission);
        upload_pool.reset();
        drop(upload_pool);
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done.
        let copying = std::time::Instant::now();
        for r in &readback_plans {
            // SAFETY: the fence completed and readback_arena is still alive and mapped.
            let (old, after) = unsafe {
                (
                    std::slice::from_raw_parts(r.before_ptr, r.len),
                    std::slice::from_raw_parts(r.after_ptr, r.len),
                )
            };
            let written = r.shadow.copy_back_range(r.from, old, after);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        }
        for shadow in pending_shadows {
            shadow.clear_pending();
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        drop(readback_plans);
        drop(batch_uploads);
        upload_arena.reset();
        readback_arena.reset();
        #[cfg(feature = "diagnostics")]
        {
            let prep_ms = (prepared - started).as_millis() as u64;
            let gpu_ms = gpu.as_millis() as u64;
            let copyback_ms = copying.elapsed().as_millis() as u64;
            if prep_ms >= 150 || gpu_ms >= 150 || copyback_ms >= 150 {
                let bindless_us = bindless_timings.plan_us
                    + bindless_timings.flush_us
                    + bindless_timings.snap_us
                    + bindless_timings.placed_us;
                tracing::warn!(
                    frame = crate::frame::current(),
                    sub = this_submission,
                    lists = runs.len(),
                    prep_ms,
                    deferred_ms = deferred_us / 1000,
                    prelude_ms = prelude_us / 1000,
                    bindless_ms = bindless_us / 1000,
                    plan_ms = bindless_timings.plan_us / 1000,
                    flush_ms = bindless_timings.flush_us / 1000,
                    snap_ms = bindless_timings.snap_us / 1000,
                    placed_ms = bindless_timings.placed_us / 1000,
                    bindless_uploads = bindless_timings.uploads,
                    gpu_ms,
                    copyback_ms,
                    tex_uploads,
                    "slow submission breakdown"
                );
            }
        }
        for list in runs {
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        self.log_messages();
    }

    /// Records the once-per-submission bindless texture uploads and initial
    /// `SHADER_RESOURCE` transitions onto `self.bindless_prelude` before any
    /// list's prelude runs.
    fn record_bindless_prelude(
        &self,
        uploads: &[BindlessUpload],
        transitions: &[(std::sync::Arc<crate::resource::HostTexture>, u32)],
    ) -> Option<ID3D12CommandList> {
        const COPY_DEST: u32 = D3D12_RESOURCE_STATE_COPY_DEST.0 as u32;
        let mut pre_barriers = Vec::with_capacity(uploads.len());
        let mut post_barriers = Vec::with_capacity(uploads.len() + transitions.len());
        for (_, copy, _, _, srv_state) in uploads {
            let real = copy.state.swap(COPY_DEST, Ordering::Relaxed);
            if real != COPY_DEST {
                pre_barriers.push(transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATES(real as i32),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ));
            }
            if *srv_state != 0 && *srv_state != COPY_DEST {
                copy.state.store(*srv_state, Ordering::Relaxed);
                post_barriers.push(transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATES(*srv_state as i32),
                ));
            }
        }
        for (copy, srv_state) in transitions {
            if *srv_state == 0 {
                continue;
            }
            let real = copy.state.swap(*srv_state, Ordering::Relaxed);
            if real != *srv_state {
                post_barriers.push(transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATES(real as i32),
                    D3D12_RESOURCE_STATES(*srv_state as i32),
                ));
            }
        }
        if uploads.is_empty() && pre_barriers.is_empty() && post_barriers.is_empty() {
            return None;
        }
        let (allocator, list) = &self.bindless_prelude;
        // SAFETY: `self.submitted` guarantees the previous submission finished;
        // every upload buffer and host texture stays alive until `finished`.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                if !pre_barriers.is_empty() {
                    list.ResourceBarrier(&pre_barriers);
                }
                for (staging, copy, upload, base_offset, _) in uploads {
                    staging.record_copies(list, &copy.texture, upload, *base_offset);
                }
                if !post_barriers.is_empty() {
                    list.ResourceBarrier(&post_barriers);
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("bindless prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            let com_identity = |obj: &ID3D12GraphicsCommandList| -> Option<*mut std::ffi::c_void> {
                obj.cast::<windows::core::IUnknown>().ok().map(|u| u.as_raw())
            };
            let list_index = |p: &D3D12_AUTO_BREADCRUMB_NODE| -> Option<usize> {
                runs.iter().position(|r| r.is_node(p))
            };
            let list_label = |p: &D3D12_AUTO_BREADCRUMB_NODE| -> String {
                let ptr = p.pCommandList.as_ref().and_then(&com_identity);
                runs.iter()
                    .enumerate()
                    .find_map(|(i, r)| {
                        if r.is_node(p) {
                            Some(format!("Some({i})"))
                        } else if ptr.is_some() && com_identity(&r.prelude.1) == ptr {
                            Some(format!("prelude {i}"))
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| "None".into())
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let label = list_label(p);
                progress.push(format!("{done}/{} {label}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let run = list_index(p)
                        .map(|i| runs[i])
                        .or_else(|| (runs.len() == 1 && p.BreadcrumbCount > 2).then_some(runs[0]));
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource or a typed null view.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    pub(crate) fn close(&self) {
        // Lists end, and so start, with depth targets current, not twins.
        let twins = std::mem::take(&mut *self.twins.lock().unwrap_or_else(|e| e.into_inner()));
        for depth in &twins {
            if let Some(twin) = &depth.twin {
                self.copy_texture(depth, twin);
            }
        }
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // The buffers it used back to common, where the next list's prelude
        // expects them.
        let back: Vec<D3D12_RESOURCE_BARRIER> = self
            .buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .lasts()
            .map(|(b, last)| {
                transition(
                    b,
                    D3D12_RESOURCE_STATES(last as i32),
                    D3D12_RESOURCE_STATE_COMMON,
                )
            })
            .collect();
        if !back.is_empty() {
            // SAFETY: an open list and buffers the list holds.
            unsafe { self.list.ResourceBarrier(&back) };
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        *self.deferred() = Deferred::default();
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.buffers.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
        }
        {
            let mut arguments = self.arguments.lock().unwrap_or_else(|e| e.into_inner());
            (arguments.filled, arguments.used) = (0, 0);
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission. A depth target with a twin is given
    /// unordered access through the twin ([`HostList::sync_twin`]).
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        if let Some(twin) = &copy.twin
            && self.sync_twin(&copy, twin, wanted)
        {
            // Unordered access goes to the twin; the depth stays as it is.
            return None;
        }
        let wanted = legal_state(texture, wanted);
        self.track(&copy, wanted).map(|before| (texture, before))
    }

    /// Notes that the list has `copy` in `wanted` from here on: the state it
    /// leaves, if that is a change (none on the list's first use).
    fn track(&self, copy: &HostCopy, wanted: u32) -> Option<u32> {
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        }
    }

    /// Records that this list starts and ends with `copy` in `wanted` state
    /// (test helper matching `transition`'s first-use tracking).
    #[cfg(test)]
    pub(crate) fn expect_texture_state(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
        wanted: u32,
    ) {
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        states
            .entry(std::sync::Arc::as_ptr(copy) as usize)
            .or_insert_with(|| (wanted, wanted, copy.clone()));
    }

    #[cfg(test)]
    pub(crate) fn has_deferred_work(&self) -> bool {
        !self.deferred().copies.is_empty()
    }

    /// Moves a host copy of title memory to `wanted` ([`BUFFER_READ`] or
    /// [`BUFFER_WRITE`]) in this list: see [`BufferStates`].
    pub(crate) fn buffer_state(&self, buffer: &ID3D12Resource, wanted: u32) {
        let mut buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(before) = buffers.update(buffer.as_raw() as usize, buffer.clone(), wanted) {
            let barrier = transition(
                buffer,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(wanted as i32),
            );
            // SAFETY: an open list and a buffer the list holds.
            unsafe { self.list.ResourceBarrier(&[barrier]) };
        }
    }

    /// [`HostList::track`], with the barrier to record if it is a change.
    fn move_to(&self, copy: &HostCopy, wanted: u32) -> Option<D3D12_RESOURCE_BARRIER> {
        let before = self.track(copy, wanted)?;
        Some(transition(
            &copy.texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(wanted as i32),
        ))
    }

    /// Keeps a depth target and its unordered access twin coherent in this
    /// list: unordered access (`wanted`) after other use copies the depth
    /// into the twin, other use after unordered access copies the twin back
    /// (as `close` does). Whether `wanted` is unordered access.
    /// shortcut: a dispatch that also reads the depth through a shader view
    /// readied after its unordered access view loses its writes (the read
    /// copies the twin back first); whole-texture states cannot have both.
    fn sync_twin(&self, depth: &HostCopy, twin: &HostCopy, wanted: u32) -> bool {
        const UNORDERED_ACCESS: u32 = 0x8;
        let unordered = wanted & UNORDERED_ACCESS != 0;
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let ahead = twins.iter().position(|d| std::sync::Arc::ptr_eq(d, depth));
        match (unordered, ahead) {
            (true, None) => {
                twins.push(depth.clone());
                drop(twins);
                self.copy_texture(twin, depth);
            }
            (false, Some(i)) => {
                twins.swap_remove(i);
                drop(twins);
                self.copy_texture(depth, twin);
            }
            _ => {}
        }
        if unordered && let Some(barrier) = self.move_to(twin, UNORDERED_ACCESS) {
            // SAFETY: an open list and a live texture.
            unsafe { self.list.ResourceBarrier(&[barrier]) };
        }
        unordered
    }

    /// Records a copy of all of `src` into `dst` (a depth target and its
    /// twin), with the transitions it needs.
    fn copy_texture(&self, dst: &HostCopy, src: &HostCopy) {
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let barriers: Vec<D3D12_RESOURCE_BARRIER> =
            [self.move_to(dst, COPY_DEST), self.move_to(src, COPY_SOURCE)]
                .into_iter()
                .flatten()
                .collect();
        // SAFETY: an open list and live textures, in the copy states by then.
        unsafe {
            if !barriers.is_empty() {
                self.list.ResourceBarrier(&barriers);
            }
            self.list.CopyResource(&dst.texture, &src.texture);
        }
    }

    /// At submission: the list that copies any newly dirtied textures from
    /// their staging buffers into their host textures and moves each texture
    /// this list uses from its real state to the one the list starts with;
    /// the textures' real states become those the list ends with (and each
    /// host buffer copy moves from common to its first state).
    fn prelude(
        &self,
        uploads: &[(
            std::sync::Arc<crate::texture::Staging>,
            std::sync::Arc<crate::resource::HostTexture>,
            ID3D12Resource,
            u64,
        )],
    ) -> Option<ID3D12CommandList> {
        const COPY_DEST: u32 = D3D12_RESOURCE_STATE_COPY_DEST.0 as u32;
        let mut pre_barriers = Vec::new();
        for (_, copy, _, _) in uploads {
            let real = copy.state.swap(COPY_DEST, Ordering::Relaxed);
            if real != COPY_DEST {
                pre_barriers.push(transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATES(real as i32),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ));
            }
        }
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        let post_barriers: Vec<D3D12_RESOURCE_BARRIER> = states
            .values()
            .filter_map(|(first, last, copy)| {
                let real = copy.state.swap(*last, Ordering::Relaxed);
                (real != *first).then(|| {
                    transition(
                        &copy.texture,
                        D3D12_RESOURCE_STATES(real as i32),
                        D3D12_RESOURCE_STATES(*first as i32),
                    )
                })
            })
            .chain(buffers.firsts().map(|(b, first)| {
                transition(
                    b,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATES(first as i32),
                )
            }))
            .collect();
        if uploads.is_empty() && post_barriers.is_empty() {
            return None;
        }
        let (allocator, list) = &self.prelude;
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures and staging buffers are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                if !pre_barriers.is_empty() {
                    list.ResourceBarrier(&pre_barriers);
                }
                for (staging, copy, upload, base_offset) in uploads {
                    staging.record_copies(list, &copy.texture, upload, *base_offset);
                }
                if !post_barriers.is_empty() {
                    list.ResourceBarrier(&post_barriers);
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if dred_on() {
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        }
    }

    /// Whether `node` of the device-removed data (its breadcrumbs) is this
    /// list.
    fn is_node(&self, node: &D3D12_AUTO_BREADCRUMB_NODE) -> bool {
        // SAFETY: only the pointer and the name text are read: the pointer is
        // compared, never used, and the name is the runtime's copy.
        let (pointer, text) = unsafe {
            (
                node.pCommandList.as_ref().map(|l| l.as_raw() as usize),
                (!node.pCommandListDebugNameW.is_null())
                    .then(|| node.pCommandListDebugNameW.to_string().unwrap_or_default()),
            )
        };
        node_is_list(
            pointer,
            text.as_deref(),
            &identities(&self.list),
            &list_name(self.id),
        )
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        let (p_start, p_end) = (
            (dst as usize) / crate::watch::PAGE,
            (dst as usize).saturating_add(len).div_ceil(crate::watch::PAGE),
        );
        deferred.refreshed.retain(|&(_, addr, l)| {
            let e_start = addr / crate::watch::PAGE;
            let e_end = addr.saturating_add(l).div_ceil(crate::watch::PAGE);
            e_end <= p_start || p_end <= e_start
        });
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        self.deferred()
            .copies
            .push(Step::GpuWrite(shadow.clone(), address, len));
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().uploads.push(buffer);
            return Some((gpu, data));
        }
        if s.used + len > SCRATCH_CHUNK && s.current < s.chunks.len() {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let (buffer, data) = host.upload_buffer(SCRATCH_CHUNK)?;
            name(&buffer, || "list scratch".into());
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            s.chunks.push((buffer, data as usize, gpu));
            s.used = 0;
        }
        let (_, data, gpu) = s.chunks[s.current];
        let at = s.used;
        s.used += len;
        Some((gpu + at, (data + at as usize) as *mut u8))
    }

    /// The CPU-only UAV slot used when recording a clear. D3D12 uses
    /// this descriptor during recording, so the next clear may reuse it.
    pub(crate) fn clear_uav_cpu(&self) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
        let heap = self
            .clear_uav_cpu
            .get_or_init(|| {
                // SAFETY: plain allocation of a CPU-only view heap.
                unsafe {
                    host()?.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: 1,
                        ..Default::default()
                    })
                }
                .ok()
            })
            .as_ref()?;
        // SAFETY: the list owns this heap until it is dropped.
        Some(unsafe { heap.GetCPUDescriptorHandleForHeapStart() })
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred
                .copies
                .push(Step::Host(Box::new(move || unsafe { (at as *mut u32).write_volatile(0) })));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// A device-local unordered-access buffer of at least `size` bytes for
    /// clamped indirect arguments, reused across executions and resets of the
    /// list. Smaller buffers replaced mid-list stay alive until
    /// [`HostList::reset`].
    pub(crate) fn indirect_scratch(&self, size: u64) -> Option<ID3D12Resource> {
        let needed = size.max(64 << 10);
        let mut slot = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, cap)) = &*slot
            && *cap >= needed
        {
            return Some(buffer.clone());
        }
        let prev = slot.as_ref().map_or(0, |(_, cap)| *cap);
        let alloc = prev
            .saturating_mul(2)
            .max(needed)
            .next_multiple_of(64 << 10);
        let buffer = host()?.uav_buffer(alloc)?;
        name(&buffer, || format!("indirect arguments {alloc:#x}"));
        let old = slot.replace((buffer.clone(), alloc));
        drop(slot);
        if let Some((old_buf, _)) = old {
            self.keep(&old_buf);
        }
        Some(buffer)
    }

    /// `len` bytes (at most 64 KiB) the list's shaders write and its
    /// indirect executions read: a buffer and the offset in it, the list's
    /// until it is reset. The buffer starts each submission in the common
    /// state.
    pub(crate) fn argument_space(&self, len: u64) -> Option<(ID3D12Resource, u64)> {
        if len > ARGUMENT_CHUNK {
            return None;
        }
        let mut a = self.arguments.lock().unwrap_or_else(|e| e.into_inner());
        if a.filled == 0 || a.used + len > ARGUMENT_CHUNK {
            if a.filled == a.chunks.len() {
                let (buffer, _) = host()?.shared_buffer(ARGUMENT_CHUNK)?;
                name(&buffer, || "indirect arguments".into());
                a.chunks.push(buffer);
            }
            a.filled += 1;
            a.used = 0;
        }
        let at = a.used;
        a.used += len.next_multiple_of(16);
        Some((a.chunks[a.filled - 1].clone(), at))
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Run(Box::new(work)));
        deferred.refreshed.clear();
    }

    /// Queues submission-time materialization and refresh of the bindless
    /// slots in `heap`, once per list; `writes`: the draw's shaders may write
    /// memory, so the slots' unordered access ranges go back to title memory.
    pub(crate) fn defer_bindless(
        &self,
        heap: &std::sync::Arc<crate::descriptor::host::HostHeap>,
        writes: bool,
    ) {
        let mut deferred = self.deferred();
        match deferred
            .bindless
            .iter_mut()
            .find(|(h, _)| std::sync::Arc::ptr_eq(h, heap))
        {
            Some((_, w)) => *w |= writes,
            None => deferred.bindless.push((heap.clone(), writes)),
        }
    }

    /// Whether each heap queued for bindless preparation may be written.
    #[cfg(test)]
    pub(crate) fn bindless_queued(&self) -> Vec<bool> {
        self.deferred().bindless.iter().map(|&(_, w)| w).collect()
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted (and copies it to `texture` on the prelude only if dirty),
    /// keeping both alive until the list is reset.
    pub(crate) fn defer_texture(
        &self,
        staging: std::sync::Arc<crate::texture::Staging>,
        texture: std::sync::Arc<crate::resource::HostTexture>,
    ) {
        self.deferred()
            .copies
            .push(Step::Texture(staging, texture));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges). Returns the textures
    /// whose staging buffers changed and must be copied onto the prelude.
    fn run_deferred(
        &self,
        batch: &mut Batch,
        host: &Host,
        pool: &mut crate::texture::UploadPool,
        upload_arena: &mut StagingArena,
    ) -> Vec<(
        std::sync::Arc<crate::texture::Staging>,
        std::sync::Arc<crate::resource::HostTexture>,
        ID3D12Resource,
        u64,
    )> {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        let mut uploads = Vec::new();
        let deferred = self.deferred();
        for (heap, writes) in &deferred.bindless {
            batch.bindless(heap, *writes);
        }
        let mut i = 0;
        while i < deferred.copies.len() {
            if matches!(deferred.copies[i], Step::Texture(..)) {
                let run_start = i;
                while i < deferred.copies.len() && matches!(deferred.copies[i], Step::Texture(..))
                {
                    i += 1;
                }
                let mut due = Vec::new();
                for step in &deferred.copies[run_start..i] {
                    if let Step::Texture(staging, copy) = step
                        && batch.texture_due_range(staging, staging.memory_range())
                    {
                        due.push((staging, copy, staging.memory_range()));
                    }
                }
                let mut spans: Ranges = due.iter().map(|&(_, _, r)| r).collect();
                let prefetched: Vec<PrefetchedWatch> = if due.len() > 1 {
                    coalesce_ranges(&mut spans, crate::watch::PAGE);
                    if spans.len() < due.len() {
                        spans
                            .into_iter()
                            .map(|(f, t)| (f, t, crate::watch::writes(f, t, true)))
                            .collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };
                let mut work_items = Vec::new();
                for (due_idx, (staging, _, (from, to))) in due.iter().enumerate() {
                    let idx = prefetched.partition_point(|&(sf, _, _)| sf <= *from);
                    let found = idx
                        .checked_sub(1)
                        .and_then(|pi| {
                            let (_, st, ref res) = prefetched[pi];
                            (st >= *to).then_some(res.as_ref()).flatten()
                        })
                        .map(|(now, list)| (*now, list.as_slice()));
                    match staging.check_watch_clean(found) {
                        Some(None) => {}
                        Some(Some(now)) => {
                            work_items.push((due_idx, (*staging).clone(), Some(now)))
                        }
                        None => work_items.push((due_idx, (*staging).clone(), None)),
                    }
                }
                for (due_idx, upload, base_offset) in
                    crate::texture::refresh_batch(host, pool, work_items)
                {
                    let (staging, copy, _) = due[due_idx];
                    uploads.push((staging.clone(), copy.clone(), upload, base_offset));
                }
                continue;
            }
            let step = &deferred.copies[i];
            i += 1;
            let started = std::time::Instant::now();
            let copied = match step {
                &Step::Copy(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.copy_written(host, upload_arena, dst, len);
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    len
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work) | Step::Host(work) => {
                    if matches!(step, Step::Run(_)) {
                        batch.title_written(host, upload_arena);
                    }
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len) => {
                    batch.gpu_write(host, upload_arena, shadow, *address, *len);
                    continue;
                }
                Step::Texture(..) => continue,
            };
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
        uploads
    }
}

/// A submission's refreshes not done yet, per buffer copy, merged where
/// they share or touch a 64 KiB page: done before a step that writes title
/// memory or gives a range to shaders, and before the lists run. Ranges
/// pages apart stay apart: one range covering a big buffer's low and high
/// ends would have its untouched middle checked (and its stale pages
/// copied) too.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
#[derive(Default)]
struct Batch {
    /// Per copy, `[from, to)` ranges in the order they came.
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, Ranges)>,
    /// Staged GPU upload copies to record on the shadow upload prelude list.
    uploads: Vec<crate::shadow::UploadCopy>,
    /// Staging buffers (by address) refreshed since the last overlapping title
    /// write, mapped to the title memory `(start, end)` they cover.
    textures: std::collections::HashMap<usize, (usize, usize)>,
    /// View heaps the submission's draws index directly, and whether their
    /// shaders may write memory.
    bindless: Vec<(std::sync::Arc<crate::descriptor::host::HostHeap>, bool)>,
    /// Their plans, which hold the host resources the slots view until the
    /// submission has run.
    plans: Vec<std::sync::Arc<crate::descriptor::host::BindlessPlan>>,
    /// Write ranges already declared in this submission (shadow identity, address, bytes).
    write_ranges: std::collections::HashSet<(usize, usize, usize)>,
    /// The same ranges, holding the shadows in first-declaration order for copy-back.
    copy_back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
}

/// `[from, to)` ranges of title memory.
type Ranges = Vec<(usize, usize)>;

/// Sorts `[from, to)` ranges and merges those whose `page`-sized pages
/// overlap or touch; empty ranges go.
pub(crate) fn coalesce_ranges(ranges: &mut Ranges, page: usize) {
    ranges.retain(|&(from, to)| from < to);
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(from, to) in ranges.iter() {
        match merged.last_mut() {
            Some((_, end)) if from / page <= end.div_ceil(page) => *end = (*end).max(to),
            _ => merged.push((from, to)),
        }
    }
    *ranges = merged;
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        if len == 0 {
            return;
        }
        let range = (address, address.saturating_add(len));
        match self
            .refreshes
            .iter_mut()
            .find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow))
        {
            Some((_, ranges)) => ranges.push(range),
            None => self.refreshes.push((shadow.clone(), vec![range])),
        }
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed for test helpers / unbounded callers).
    #[cfg(test)]
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.texture_due_range(staging, (0, usize::MAX))
    }

    /// Whether `staging` (backed by title range `(start, end)`) is still to be
    /// refreshed in this submission, noting its backing range so disjoint
    /// buffer writes leave it valid.
    fn texture_due_range<T>(
        &mut self,
        staging: &std::sync::Arc<T>,
        range: (usize, usize),
    ) -> bool {
        self.textures
            .insert(std::sync::Arc::as_ptr(staging) as *const () as usize, range)
            .is_none()
    }

    /// Host heaps indexed directly by draws/dispatches of this submission.
    fn bindless(&mut self, heap: &std::sync::Arc<crate::descriptor::host::HostHeap>, writes: bool) {
        match self
            .bindless
            .iter_mut()
            .find(|(h, _)| std::sync::Arc::ptr_eq(h, heap))
        {
            Some((_, w)) => *w |= writes,
            None => self.bindless.push((heap.clone(), writes)),
        }
    }

    fn record_write(
        &mut self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if self.write_ranges.insert(key) {
            shadow.mark_pending(address, len);
            self.copy_back.push((shadow.clone(), address, len));
        }
    }

    fn gpu_write(
        &mut self,
        host: &Host,
        upload_arena: &mut StagingArena,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        // Flush only pending refreshes on `shadow` whose pages overlap this
        // GPU write range before declaring it busy.
        self.flush_shadow_overlapping(host, upload_arena, shadow, address, address.saturating_add(len));
        self.record_write(shadow, address, len);
    }

    /// Before a known copy into title memory `[start, start + len)`: flush only
    /// overlapping pending refreshes, invalidate overlapping textures and
    /// write-watch sweep pages.
    fn copy_written(&mut self, host: &Host, upload_arena: &mut StagingArena, start: usize, len: usize) {
        if len == 0 {
            return;
        }
        let end = start.saturating_add(len);
        self.flush_overlapping(host, upload_arena, start, end);
        crate::watch::Sweep::invalidate(start, len);
        let (p_start, p_end) = (start / crate::watch::PAGE, end.div_ceil(crate::watch::PAGE));
        self.textures.retain(|_, &mut (ts, te)| {
            if ts >= te {
                return true;
            }
            let t_start = ts / crate::watch::PAGE;
            let t_end = te.div_ceil(crate::watch::PAGE);
            t_end <= p_start || p_end <= t_start
        });
    }

    /// Before a step that may arbitrarily write title memory (`Step::Run`):
    /// pending refreshes are done first, and textures and write-watch sweep
    /// pages are checked again after it.
    fn title_written(&mut self, host: &Host, upload_arena: &mut StagingArena) {
        self.flush(host, upload_arena);
        self.textures.clear();
        crate::watch::Sweep::title_written();
    }

    fn flush(&mut self, host: &Host, upload_arena: &mut StagingArena) {
        self.flush_matching(host, upload_arena, |_, _, _| true);
    }

    fn flush_overlapping(
        &mut self,
        host: &Host,
        upload_arena: &mut StagingArena,
        start: usize,
        end: usize,
    ) {
        let (p_start, p_end) = (start / crate::watch::PAGE, end.div_ceil(crate::watch::PAGE));
        self.flush_matching(host, upload_arena, |_, from, to| {
            from / crate::watch::PAGE < p_end && p_start < to.div_ceil(crate::watch::PAGE)
        });
    }

    fn flush_shadow_overlapping(
        &mut self,
        host: &Host,
        upload_arena: &mut StagingArena,
        target: &std::sync::Arc<crate::shadow::Shadow>,
        start: usize,
        end: usize,
    ) {
        if start >= end {
            return;
        }
        let (p_start, p_end) = (start / crate::watch::PAGE, end.div_ceil(crate::watch::PAGE));
        self.flush_matching(host, upload_arena, |s, from, to| {
            std::sync::Arc::ptr_eq(s, target)
                && from / crate::watch::PAGE < p_end
                && p_start < to.div_ceil(crate::watch::PAGE)
        });
    }

    fn flush_matching(
        &mut self,
        host: &Host,
        upload_arena: &mut StagingArena,
        mut pred: impl FnMut(&std::sync::Arc<crate::shadow::Shadow>, usize, usize) -> bool,
    ) {
        let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, Ranges)> = Vec::new();
        let mut total_due_ranges = 0usize;
        let mut all_spans: Ranges = Vec::new();
        for (shadow, ranges) in std::mem::take(&mut self.refreshes) {
            let (mut due, keep): (Ranges, Ranges) = ranges
                .into_iter()
                .partition(|&(from, to)| pred(&shadow, from, to));
            if !due.is_empty() {
                coalesce_ranges(&mut due, crate::watch::PAGE);
                for &(from, to) in &due {
                    total_due_ranges += 1;
                    if shadow.can_use_prefetched(from, to) {
                        all_spans.push((from, to));
                    }
                }
                work.push((shadow.clone(), due));
            }
            if !keep.is_empty() {
                self.refreshes.push((shadow, keep));
            }
        }
        let prefetched: Vec<PrefetchedWatch> = if !all_spans.is_empty() {
            coalesce_ranges(&mut all_spans, crate::watch::PAGE);
            if all_spans.len() < total_due_ranges {
                all_spans
                    .into_iter()
                    .map(|(f, t)| (f, t, crate::watch::writes(f, t, true)))
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        for (shadow, due) in work {
            for (from, to) in due {
                update_with_prefetched(
                    &shadow,
                    from,
                    to,
                    &prefetched,
                    &mut |n| upload_arena.alloc(host, StagingKind::Upload, n),
                    &mut self.uploads,
                );
            }
        }
    }

    /// After every list's steps (no title copy follows): each directly
    /// indexed heap's slots, once per submission however many lists and
    /// draws use it. Plan reads coalesce into `self.refreshes` before a single
    /// flush, writable ranges mark pending once per shadow, and deduplicated
    /// placed textures refresh & upload once per submission via prefetched watch.
    fn prepare_bindless(
        &mut self,
        host: &Host,
        pool: &mut crate::texture::UploadPool,
        upload_arena: &mut StagingArena,
        list_uploads: &mut [Vec<ListUpload>],
    ) -> (Option<ID3D12CommandList>, BindlessTimings) {
        #[allow(unused_mut)]
        let mut timings = BindlessTimings::default();
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let mut pending_plans = Vec::new();
        for (heap, writes) in std::mem::take(&mut self.bindless) {
            let Some(plan) = heap.bindless_plan() else {
                continue;
            };
            plan.copy_constants();
            for (shadow, address, len) in &plan.reads {
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Refreshes, 1);
                    add(C::RefreshBytes, *len as u64);
                }
                self.add(shadow, *address, *len);
            }
            pending_plans.push((plan, writes));
        }
        #[cfg(feature = "diagnostics")]
        {
            timings.plan_us = started.elapsed().as_micros() as u64;
        }

        #[cfg(feature = "diagnostics")]
        let t_flush = std::time::Instant::now();
        self.flush(host, upload_arena);
        #[cfg(feature = "diagnostics")]
        {
            timings.flush_us = t_flush.elapsed().as_micros() as u64;
        }
        if pending_plans.is_empty() {
            return (None, timings);
        }

        #[cfg(feature = "diagnostics")]
        let t_snap = std::time::Instant::now();
        for (plan, writes) in &pending_plans {
            if !*writes {
                continue;
            }
            for &(ref shadow, address, len) in &plan.writes {
                if len == 0 {
                    continue;
                }
                self.record_write(shadow, address, len);
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            timings.snap_us = t_snap.elapsed().as_micros() as u64;
        }

        #[cfg(feature = "diagnostics")]
        let t_placed = std::time::Instant::now();
        let prefetched: Vec<PrefetchedWatch> = if !crate::texture::hash_textures() {
            let mut spans: Ranges = pending_plans
                .iter()
                .flat_map(|(plan, _)| plan.placed_watch_spans.iter().copied())
                .collect();
            if !spans.is_empty() {
                if pending_plans.len() > 1 {
                    coalesce_ranges(&mut spans, crate::watch::PAGE);
                }
                spans
                    .into_iter()
                    .map(|(f, t)| (f, t, crate::watch::writes(f, t, true)))
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        let mut seen_copies = std::collections::HashSet::<usize>::new();
        let mut unique_targets = Vec::new();
        let mut work_items = Vec::new();
        for (plan, _) in &pending_plans {
            for target in &plan.placed_textures {
                if !seen_copies.insert(std::sync::Arc::as_ptr(&target.copy) as usize) {
                    continue;
                }
                let target_idx = unique_targets.len();
                unique_targets.push(target);
                if let (Some(staging), Some((from, to))) = (&target.staging, target.watch_range)
                    && !crate::texture::is_host_written(target.resource)
                    && self.texture_due_range(staging, (from, to))
                {
                    let idx = prefetched.partition_point(|&(sf, _, _)| sf <= from);
                    let found = idx
                        .checked_sub(1)
                        .and_then(|pi| {
                            let (_, st, ref res) = prefetched[pi];
                            (st >= to).then_some(res.as_ref()).flatten()
                        })
                        .map(|(now, list)| (*now, list.as_slice()));
                    match staging.check_watch_clean(found) {
                        Some(None) => {}
                        Some(Some(now)) => {
                            work_items.push((target_idx, staging.clone(), Some(now)))
                        }
                        None => work_items.push((target_idx, staging.clone(), None)),
                    }
                }
            }
        }
        let mut uploaded_flags = vec![false; unique_targets.len()];
        let mut bindless_uploads: Vec<BindlessUpload> = Vec::new();
        let mut state_transitions: Vec<(std::sync::Arc<crate::resource::HostTexture>, u32)> =
            Vec::new();
        for (target_idx, upload, base_offset) in
            crate::texture::refresh_batch(host, pool, work_items)
        {
            let target = unique_targets[target_idx];
            if let Some(staging) = &target.staging {
                uploaded_flags[target_idx] = true;
                for uploads in list_uploads.iter_mut() {
                    uploads.retain(|(_, c, _, _)| !std::sync::Arc::ptr_eq(c, &target.copy));
                }
                bindless_uploads.push((
                    staging.clone(),
                    target.copy.clone(),
                    upload,
                    base_offset,
                    target.srv_state,
                ));
            }
        }
        for (target_idx, target) in unique_targets.into_iter().enumerate() {
            if !uploaded_flags[target_idx]
                && target.srv_state != 0
                && target.copy.state.load(Ordering::Relaxed) != target.srv_state
            {
                state_transitions.push((target.copy.clone(), target.srv_state));
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            timings.uploads = bindless_uploads.len();
        }
        for (plan, _) in pending_plans {
            self.plans.push(plan);
        }
        let prelude = host.record_bindless_prelude(&bindless_uploads, &state_transitions);
        #[cfg(feature = "diagnostics")]
        {
            timings.placed_us = t_placed.elapsed().as_micros() as u64;
            let total_us =
                timings.plan_us + timings.flush_us + timings.snap_us + timings.placed_us;
            crate::copystat::add(crate::copystat::C::BindlessUs, total_us);
        }
        (prelude, timings)
    }
}

type BindlessUpload = (
    std::sync::Arc<crate::texture::Staging>,
    std::sync::Arc<crate::resource::HostTexture>,
    ID3D12Resource,
    u64,
    u32,
);

type ListUpload = (
    std::sync::Arc<crate::texture::Staging>,
    std::sync::Arc<crate::resource::HostTexture>,
    ID3D12Resource,
    u64,
);

#[derive(Default)]
struct BindlessTimings {
    #[cfg(feature = "diagnostics")]
    plan_us: u64,
    #[cfg(feature = "diagnostics")]
    flush_us: u64,
    #[cfg(feature = "diagnostics")]
    snap_us: u64,
    #[cfg(feature = "diagnostics")]
    placed_us: u64,
    #[cfg(feature = "diagnostics")]
    uploads: usize,
}

type WatchAnswer = Option<(u64, Vec<(usize, u64)>)>;
type PrefetchedWatch = (usize, usize, WatchAnswer);

/// Brings `[from, to)` of a host copy up to date, reusing a pre-fetched
/// cross-shadow `watch::writes` answer when `[from, to)` is covered.
fn update_with_prefetched(
    shadow: &std::sync::Arc<crate::shadow::Shadow>,
    from: usize,
    to: usize,
    prefetched: &[PrefetchedWatch],
    alloc: &mut impl FnMut(usize) -> Option<(ID3D12Resource, u64, *mut u8)>,
    out: &mut Vec<crate::shadow::UploadCopy>,
) {
    let idx = prefetched.partition_point(|&(sf, _, _)| sf <= from);
    let found = idx
        .checked_sub(1)
        .and_then(|i| {
            let (_, st, ref res) = prefetched[i];
            (st >= to).then_some(res.as_ref()).flatten()
        })
        .map(|(now, list)| (*now, list.as_slice()));
    let started = std::time::Instant::now();
    let copied = shadow.update_with_watch(from, to - from, found, alloc, out);
    let us = started.elapsed().as_micros() as u64;
    #[cfg(feature = "diagnostics")]
    crate::copystat::range(from, to - from, us, 0);
    PREPARING[0].fetch_add(us, Ordering::Relaxed);
    PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped takes nothing from its
/// first unmapped page on.
///
/// The copy runs in the kernel (`ReadProcessMemory`), which reports an
/// unmapped page of the source, or of the destination it writes, instead of
/// faulting. Asking `VirtualQuery` first cost about 2 ms per copy in the
/// title's big pools (it walks the page tables to the region's end), which
/// made copies run at 100-200 MB/s; a title destination is asked only once
/// its copy has failed (asked before each, the world's 80 copies per frame
/// cost 60 ms).
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    if title_dst {
        // The kernel writes the destination: it must not be guarded.
        crate::watch::writing(dst as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    // The copy stopped at an unmapped page. One of a title destination ends
    // it: the retries below write the destination themselves (zeros for
    // what cannot be read), and would fault there.
    if title_dst && !crate::texture::committed(dst, len) {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    if !title_dst && len >= 64 << 10 {
        read_sparse(src, dst, len);
    } else {
        read_pieces(src, dst, len);
    }
}

/// After a failed bulk read into host memory, skip contiguous holes. Never
/// cache the query across copies: the title may remap the same address.
fn read_sparse(src: *const u8, dst: *mut u8, len: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualQuery,
    };
    let mut offset = 0;
    while offset < len {
        let at = src.wrapping_add(offset);
        let to = dst.wrapping_add(offset);
        let remaining = len - offset;
        // SAFETY: the query inspects an address and fills a local structure.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let size = unsafe { VirtualQuery(at.cast(), &mut info, size_of_val(&info)) };
        let n = (info.BaseAddress as usize).saturating_add(info.RegionSize)
            .saturating_sub(at as usize).min(remaining);
        if size != size_of_val(&info) || n == 0 {
            read_pieces(at, to, remaining);
            break;
        }
        if info.State != MEM_COMMIT || info.Protect == PAGE_NOACCESS {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
            // SAFETY: this path is only used with a writable host destination.
            unsafe { to.write_bytes(0, n) };
        } else if !read(at, to, n) {
            // A mapping may change between query and read.
            read_pieces(at, to, n);
        }
        offset += n;
    }
}

fn read_pieces(src: *const u8, dst: *mut u8, len: usize) {
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read. Pages found unreadable
    // are remembered for a while and not asked again: each failed read is
    // a kernel call (a server round trip under Wine), and the world's
    // copies cross the same unmapped pages every frame.
    let zero = |to: *mut u8, n: usize| {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
        // SAFETY: inside the destination, which is mapped.
        unsafe { to.write_bytes(0, n) };
    };
    let mut unreadable = UNREADABLE.lock().unwrap_or_else(|e| e.into_inner());
    let now = std::time::Instant::now();
    let known = |page: usize, map: &std::collections::BTreeMap<usize, std::time::Instant>| {
        map.get(&page)
            .is_some_and(|&seen| now.duration_since(seen) < UNREADABLE_FOR)
    };
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        let any_known = (at / PAGE..(at + n).div_ceil(PAGE)).any(|p| known(p, &unreadable));
        if !any_known && read(at as *const u8, to, n) {
            return;
        }
        for_pieces(at, n, PAGE, |at, n| {
            let to = dst.wrapping_add(at - src as usize);
            if known(at / PAGE, &unreadable) {
                zero(to, n);
            } else if !read(at as *const u8, to, n) {
                unreadable.insert(at / PAGE, now);
                zero(to, n);
            }
        });
    });
}

/// 4 KiB pages.
const PAGE: usize = 4 << 10;
/// How long a page [`read_pieces`] found unreadable is taken as unmapped
/// without asking again, unless the runtime remaps it first
/// ([`forget_unreadable`]); after that it is asked again, for memory that
/// is mapped another way.
const UNREADABLE_FOR: std::time::Duration = std::time::Duration::from_secs(2);
/// Pages [`read_pieces`] found unreadable, by page number, and when.
static UNREADABLE: Mutex<std::collections::BTreeMap<usize, std::time::Instant>> =
    Mutex::new(std::collections::BTreeMap::new());

/// The runtime mapped or unmapped `[at, at + len)`: its pages are asked
/// again.
pub(crate) fn forget_unreadable(at: usize, len: usize) {
    let mut unreadable = UNREADABLE.lock().unwrap_or_else(|e| e.into_inner());
    let pages: Vec<usize> = unreadable
        .range(at / PAGE..at.saturating_add(len).div_ceil(PAGE))
        .map(|(&p, _)| p)
        .collect();
    for page in pages {
        unreadable.remove(&page);
    }
    // Entries past their time are dropped here too, so the map stays small.
    let now = std::time::Instant::now();
    unreadable.retain(|_, seen| now.duration_since(*seen) < UNREADABLE_FOR);
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` in the kernel: false if a source
/// page is not readable or a destination page not writable (the bytes
/// before that page may have been copied).
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is `len` bytes the caller may write, host
    // memory or title memory; the kernel checks both ranges, and a page it
    // cannot read or write fails the call instead of faulting.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
///
/// Through [`read`]: the kernel refuses an unmapped destination page as it
/// does a source page. `WriteProcessMemory` makes the same copy, but asks
/// for the destination's region first, at `VirtualQuery`'s cost (see
/// [`copy_mapped`]): 0.35 ms for each piece of the world's copy-back, 39 ms
/// per frame.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    let write = |at: usize, n: usize| read(src.wrapping_add(at - dst as usize), at as *mut u8, n);
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

#[cfg(test)]
mod title_copies {
    use super::{copy_mapped, write_mapped};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualProtect,
    };

    const BLOCK: usize = 64 << 10;

    /// Three reserved blocks, the first and the last committed and filled
    /// with `fill`: title memory with an unmapped block in it.
    fn holed(fill: u8) -> *mut u8 {
        // SAFETY: a fresh reservation; the committed blocks are ours to fill.
        unsafe {
            let base =
                VirtualAlloc(std::ptr::null(), 3 * BLOCK, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            for block in [0, 2] {
                let at = base.add(block * BLOCK);
                assert!(!VirtualAlloc(at.cast(), BLOCK, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.write_bytes(fill, BLOCK);
            }
            base
        }
    }

    fn block<'a>(base: *const u8, n: usize) -> &'a [u8] {
        // SAFETY: tests ask for committed blocks only.
        unsafe { std::slice::from_raw_parts(base.add(n * BLOCK), BLOCK) }
    }

    fn release(base: *mut u8) {
        super::forget_unreadable(base as usize, 3 * BLOCK);
        // SAFETY: a reservation of `holed`.
        assert_ne!(unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) }, 0);
    }

    fn pattern() -> Vec<u8> {
        (0..3 * BLOCK).map(|i| (i % 251) as u8 + 1).collect()
    }

    #[test]
    fn title_copies_reach_mapped_destinations_and_stop_at_unmapped_pages() {
        let from = pattern();
        // All mapped: copied, through a destination that was never asked.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), BLOCK, true);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        release(to);
        // An unmapped block in the destination: nothing faults, and nothing
        // arrives from that block on.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), 3 * BLOCK, true);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
        // An unmapped block in the source reads as zeros.
        let holes = holed(7);
        let mut to = vec![0xeeu8; 3 * BLOCK];
        copy_mapped(to.as_mut_ptr(), holes, 3 * BLOCK, true);
        assert!(to[..BLOCK].iter().all(|&b| b == 7));
        assert!(to[BLOCK..2 * BLOCK].iter().all(|&b| b == 0));
        assert!(to[2 * BLOCK..].iter().all(|&b| b == 7));
        release(holes);
    }

    #[test]
    fn copy_back_writes_mapped_pages_only() {
        let from = pattern();
        let to = holed(0xee);
        write_mapped(to, from.as_ptr(), 3 * BLOCK);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        assert_eq!(block(to, 2), &from[2 * BLOCK..]);
        release(to);
        // Neither aligned nor whole: the bytes around the range stay.
        let to = holed(0xee);
        // SAFETY: inside the first block.
        write_mapped(unsafe { to.add(100) }, from.as_ptr(), BLOCK - 300);
        let first = block(to, 0);
        assert_eq!(first[100..BLOCK - 200], from[..BLOCK - 300]);
        let around = first[..100].iter().chain(&first[BLOCK - 200..]);
        assert!(around.into_iter().all(|&b| b == 0xee));
        // A page that is mapped but not writable is not written either
        // (guarded title blocks are opened before they are copied back).
        let mut old = 0;
        // SAFETY: a committed block of ours.
        let protected =
            unsafe { VirtualProtect(to.add(2 * BLOCK).cast(), BLOCK, PAGE_READONLY, &mut old) };
        assert_ne!(protected, 0);
        // SAFETY: the third block.
        write_mapped(unsafe { to.add(2 * BLOCK) }, from.as_ptr(), BLOCK);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
    }
}

impl Host {
    /// A buffer in the upload heap, mapped for writing.
    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
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
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    hwnd: HWND,
    meter: Meter,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect",
                making = crate::pipeline::making()
            );
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::log(frames);
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown; else, in diagnostics builds, whether the frame was black.
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            };
            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            let list = &p.copy.list;
            list.ResourceBarrier(&[
                transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
            ]);
            // Diagnostics: `VICTORY_SHOW_TARGET` shows one of the title's render targets instead.
            #[cfg(feature = "diagnostics")]
            let shown = crate::capture::show(list, &back);
            #[cfg(not(feature = "diagnostics"))]
            let shown = false;
            if !shown {
                list.CopyResource(&back, texture);
            }
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = crate::capture::probe(&self.device, list, texture);
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            list.ResourceBarrier(&[
                transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ),
            ]);
            p.copy.close();
            self.execute(&[&p.copy]);
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                .cast::<IDXGISwapChain3>()?;
            let copy = self.new_list()?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                hwnd,
                meter: Meter::new(),
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// Whether `texture` was created with `ALLOW_RENDER_TARGET` (some formats
/// lose it when the host refuses that use).
pub(crate) fn allows_rtv(texture: &ID3D12Resource) -> bool {
    // SAFETY: a plain query.
    (unsafe { texture.GetDesc() }.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
        != D3D12_RESOURCE_FLAG_NONE
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// A buffer in GPU-local VRAM (`D3D12_HEAP_TYPE_DEFAULT`) with unordered
    /// access enabled: for host copies of title buffers, uploaded via staging
    /// copies on submission. Starts in `COMMON` (buffers promote/decay cleanly).
    pub(crate) fn default_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        self.uav_buffer(size)
    }

    /// A buffer in the readback heap, persistently mapped for CPU reading.
    pub(crate) fn readback_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
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
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a readback buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no readback buffer");
                None
            }
        }
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }

    /// A device-local buffer shaders may write (unordered access) and indirect
    /// execution may read, starting in the common state.
    pub(crate) fn uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a device-local buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no unordered-access buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    let start = if done > from + 16 { done - 16 } else { from };
    for i in start..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 16 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::{Batch, coalesce_ranges};
    use std::sync::Arc;

    const PAGE: usize = crate::watch::PAGE;

    fn coalesced(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
        coalesce_ranges(&mut ranges, PAGE);
        ranges
    }

    #[test]
    fn empty_ranges_are_dropped() {
        assert_eq!(coalesced(vec![]), []);
        assert_eq!(coalesced(vec![(5, 5), (PAGE, PAGE)]), []);
        assert_eq!(coalesced(vec![(10, 10), (20, 30)]), [(20, 30)]);
    }

    #[test]
    fn ranges_in_one_page_merge() {
        assert_eq!(coalesced(vec![(100, 200), (10, 20), (150, 400)]), [(10, 400)]);
        // Nested and repeated.
        assert_eq!(coalesced(vec![(0, PAGE), (8, 16), (0, PAGE)]), [(0, PAGE)]);
    }

    #[test]
    fn ranges_in_touching_pages_merge() {
        // Ends in page 0, starts in page 1.
        assert_eq!(coalesced(vec![(PAGE + 8, PAGE + 16), (0, 8)]), [(0, PAGE + 16)]);
        // Ends exactly at page 1's start, starts in page 1.
        assert_eq!(coalesced(vec![(0, PAGE), (PAGE + 100, 2 * PAGE)]), [(0, 2 * PAGE)]);
        // A chain of ranges each in the page after the last.
        let chain: Vec<_> = (0..8).map(|p| (p * PAGE + 10, p * PAGE + 20)).collect();
        assert_eq!(coalesced(chain), [(10, 7 * PAGE + 20)]);
    }

    #[test]
    fn ranges_pages_apart_stay_apart() {
        // A whole untouched page between them.
        assert_eq!(coalesced(vec![(2 * PAGE, 2 * PAGE + 1), (0, 1)]), [(0, 1), (2 * PAGE, 2 * PAGE + 1)]);
        // The low and high ends of a 256 MiB buffer.
        let (low, high) = (1 << 30, (1 << 30) + (256 << 20) - 64);
        assert_eq!(
            coalesced(vec![(high, high + 64), (low, low + 64), (low + 32, low + 128)]),
            [(low, low + 128), (high, high + 64)]
        );
    }

    #[test]
    fn ranges_at_the_top_of_the_address_space_do_not_overflow() {
        let top = usize::MAX;
        assert_eq!(coalesced(vec![(top - 8, top), (top - 100, top - 50)]), [(top - 100, top)]);
    }

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        assert!(batch.refreshes.is_empty());
        batch.textures.clear();
        crate::watch::Sweep::title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    #[test]
    fn copy_written_invalidates_only_overlapping_textures() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new(1u32), Arc::new(2u32));
        let p = crate::watch::PAGE;
        assert!(batch.texture_due_range(&a, (0x10_0000, 0x10_0000 + p)));
        assert!(batch.texture_due_range(&b, (0x20_0000, 0x20_0000 + p)));
        let invalidate = |batch: &mut Batch, start: usize, len: usize| {
            let end = start.saturating_add(len);
            let (p_start, p_end) = (start / crate::watch::PAGE, end.div_ceil(crate::watch::PAGE));
            batch.textures.retain(|_, &mut (ts, te)| {
                if ts >= te {
                    return true;
                }
                let t_start = ts / crate::watch::PAGE;
                let t_end = te.div_ceil(crate::watch::PAGE);
                t_end <= p_start || p_end <= t_start
            });
        };
        // Writing to an unrelated range leaves both textures cached.
        invalidate(&mut batch, 0x30_0000, 4096);
        assert!(!batch.texture_due_range(&a, (0x10_0000, 0x10_0000 + p)));
        assert!(!batch.texture_due_range(&b, (0x20_0000, 0x20_0000 + p)));
        // Writing onto `a`'s page invalidates only `a`.
        invalidate(&mut batch, 0x10_0100, 256);
        assert!(batch.texture_due_range(&a, (0x10_0000, 0x10_0000 + p)));
        assert!(!batch.texture_due_range(&b, (0x20_0000, 0x20_0000 + p)));
    }

    #[test]
    fn cross_shadow_coalescing_batches_watch_ask() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let Some(host) = super::host() else {
            return;
        };
        let p = crate::watch::PAGE;
        // SAFETY: fresh 2-page write-watched region freed at the end.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                2 * p,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0);
        // SAFETY: committed read-write pages above.
        unsafe {
            (base as *mut u8).write_bytes(0x11, p);
            ((base + p) as *mut u8).write_bytes(0x22, p);
        }
        crate::shadow::register(0x7001_0001, base as u64, p as u64);
        crate::shadow::register(0x7001_0002, (base + p) as u64, p as u64);
        let s1 = crate::shadow::find(base as u64, p as u64).unwrap();
        let s2 = crate::shadow::find((base + p) as u64, p as u64).unwrap();

        let asked = crate::watch::kernel_asked();
        let mut arena = super::StagingArena::default();
        let mut batch = Batch::default();
        batch.add(&s1, base, p);
        batch.add(&s2, base + p, p);
        batch.flush(host, &mut arena);
        assert_eq!(crate::watch::kernel_asked(), asked + 1);
        assert_eq!(batch.uploads.len(), 2);
        let read_staged_u32 = |u: &crate::shadow::UploadCopy| -> u32 {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: upload buffer mapped for at least `src_offset + 4` bytes.
            unsafe {
                u.src.Map(0, None, Some(&mut ptr)).unwrap();
                let val = ptr.cast::<u8>().add(u.src_offset as usize).cast::<u32>().read_unaligned();
                u.src.Unmap(0, None);
                val
            }
        };
        assert_eq!(read_staged_u32(&batch.uploads[0]), 0x1111_1111);
        assert_eq!(read_staged_u32(&batch.uploads[1]), 0x2222_2222);

        crate::shadow::unregister(0x7001_0001, base as u64);
        crate::shadow::unregister(0x7001_0002, (base + p) as u64);
        // SAFETY: freeing our own allocation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }
}

#[cfg(test)]
mod unreadable_pages {
    use super::{PAGE, copy_mapped, forget_unreadable};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    #[test]
    fn an_unmapped_page_is_remembered_until_remapped() {
        // Four pages: 0 and 3 committed, 1 and 2 not.
        // SAFETY: plain reservation and commits of our own pages.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE, PAGE_READWRITE) };
        assert!(!base.is_null());
        let page = |i: usize| base.cast::<u8>().wrapping_add(i * PAGE);
        let commit = |i: usize, value: u8| {
            // SAFETY: inside the reservation above.
            unsafe {
                assert!(!VirtualAlloc(page(i).cast(), PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
                page(i).write_bytes(value, PAGE);
            }
        };
        let filled = |out: &[u8], i: usize, value: u8| {
            out[i * PAGE..(i + 1) * PAGE].iter().all(|&b| b == value)
        };
        commit(0, 1);
        commit(3, 4);
        let mut out = vec![0xffu8; 4 * PAGE];
        copy_mapped(out.as_mut_ptr(), page(0), 4 * PAGE, false);
        assert!(
            filled(&out, 0, 1)
                && filled(&out, 1, 0)
                && filled(&out, 2, 0)
                && filled(&out, 3, 4)
        );
        // Page 1 mapped behind the runtime's back while page 2 stays
        // unmapped: the copy falls back again and takes page 1 as unmapped.
        commit(1, 2);
        out.fill(0xff);
        copy_mapped(out.as_mut_ptr(), page(0), 4 * PAGE, false);
        assert!(
            filled(&out, 0, 1)
                && filled(&out, 1, 0)
                && filled(&out, 2, 0)
                && filled(&out, 3, 4)
        );
        // Remapped through the runtime: page 1 is read again.
        forget_unreadable(page(1) as usize, PAGE);
        out.fill(0xff);
        copy_mapped(out.as_mut_ptr(), page(0), 4 * PAGE, false);
        assert!(
            filled(&out, 0, 1)
                && filled(&out, 1, 2)
                && filled(&out, 2, 0)
                && filled(&out, 3, 4)
        );
        forget_unreadable(base as usize, 4 * PAGE);
        // SAFETY: the reservation above, no longer used.
        unsafe { VirtualFree(base, 0, MEM_RELEASE) };
    }
}

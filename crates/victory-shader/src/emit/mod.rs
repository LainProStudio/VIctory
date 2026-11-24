//! DXIL emission (decision 0003, step 3): a console vertex, pixel, compute
//! or mesh shader as a PC DXIL container (SM 6.6) that host D3D12 accepts.
//! Compute shaders keep the console's thread-group size; one that depends
//! on the lane layout (cross-lane operations) declares the console's wave
//! size (`WaveSize`), so the host device must support it.
//!
//! What the host binds for a translated shader ([`HostShader`]):
//!
//! * **User data** at `b0, space 0` (32-bit root constants or a constant
//!   buffer, per stage): dword `index` holds a root parameter's value, as
//!   [`UserData`] lists. A root-constant dword is the constant. A
//!   descriptor table is the byte offset of its first descriptor in its
//!   heap: the descriptor index times 32 for CBV/SRV/UAV tables, times 16
//!   for sampler tables (the console's descriptor sizes; the code adds
//!   offsets in those units). The heap bases of dynamic resources are 0
//!   and not listed. An entry of root type `ResourceHeap` (`SamplerHeap`)
//!   is a guard slot of the bound view (sampler) heap, past the title's
//!   console dwords: every heap index the code computes at or above it
//!   reads that slot instead (an index past a heap hangs some GPUs). The
//!   host keeps a view heap's guard slot null and gives no table or title
//!   descriptor a slot at or above it; any sampler slot will do.
//! * **Descriptors** through SM 6.6 `ResourceDescriptorHeap[]` and
//!   `SamplerDescriptorHeap[]`, at the index the console address gives, so
//!   the host heaps keep the title's descriptor order. The root signature
//!   needs `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` and
//!   `SAMPLER_HEAP_DIRECTLY_INDEXED`.
//! * **Immediate samplers** as static samplers `s<n>, space 0`, `n` the
//!   position in [`HostShader::samplers`] (the console sampler words) and
//!   [`HostShader::static_samplers`] (the same decoded into a D3D12 static
//!   sampler description). Comparison sampling (added 2026-11-23): an
//!   immediate sampler it uses has a comparison filter there (bit 0x80);
//!   a heap sampler it uses must be a comparison sampler on the host.
//! * **Root CBVs, root SRVs and root UAVs** (raw buffers) as root descriptors
//!   `b<i>` / `t<i>` / `u<i>` in [`ROOT_CBV_SPACE`], `i` the title's root
//!   parameter index ([`HostShader::root_cbvs`], [`HostShader::root_srvs`],
//!   [`HostShader::root_uavs`]); the address the title sets is the buffer's.
//! * **Vertex fetch** (added 2026-11-23). Grand Theft Auto VI's mesh and
//!   vertex programs fetch vertices themselves with typed-format loads
//!   (`tbuffer_load_format_*`): the format (8_8_8_8 unorm, 16_16 snorm,
//!   2_10_10_10 unorm, 8_8_8_8 and 16_16_16_16 uint...) is in the
//!   instruction, so the translation reads raw bytes and converts them
//!   itself. Their descriptors are built around a root SRV (stride 1, no
//!   record limit): the host binds the root SRV `t<slot>` above with the
//!   address the title sets, nothing more. The vertex index, the stride
//!   and the attribute offset are the shader's own arithmetic on data in
//!   that buffer (per-stream headers), not descriptor fields.
//! * **Buffer views in the heap** are declared by how the shader reads
//!   them: raw (ByteAddressBuffer) for dword loads and stores, atomics and
//!   typed-format loads without an index; structured with the used-resource
//!   entry's stride when indexed; typed with four 32-bit float channels for
//!   `buffer_load_format_*` and `buffer_store_format_*`. The host's view at
//!   that index must be of that kind and start at the console descriptor's
//!   base address (raw: `R32_TYPELESS` with the raw flag), whatever kind the
//!   title's view description says. Typed-format loads from heap
//!   descriptors (1,024 world compute programs, ray-tracing material
//!   resolves) read raw views this way.
//!
//! * **Texture atomics** use a scalar integer UAV. The host binds R32_UINT
//!   for 32-bit operations and R32G32_UINT for 64-bit operations. The latter
//!   require SM 6.6 atomic support on typed and descriptor-heap resources.
//!
//! Entry points: [`translate`] (container bytes as the title passes them
//! at pipeline creation → [`HostShader`]), [`emit`] (the same with the
//! earlier stages already run). The root constants of each stage must
//! cover [`HostShader::user_data_dwords`]. Stability: these types and the
//! binding scheme are what pipeline creation builds on; changes are
//! announced.
//! * **Signatures** as the console container has them (same semantics,
//!   registers and masks), so the input layout and stage linking are the
//!   title's; pixel outputs are plain 32-bit values.
//! * **Mesh shaders** ([`HwStage::Ms`], `ms_6_6`): user data and
//!   descriptors as above (the root constants visible to the mesh stage);
//!   the pipeline is a mesh pipeline (pipeline-state stream, no input
//!   layout) with the title's pixel shader. The thread group is the
//!   console's in lanes (`[numthreads]` in the DXIL), the vertex and
//!   primitive maxima its own, triangles only, no wave size required. The
//!   console numbers groups in x only: the draw path passes the title's
//!   count as `DispatchMesh(x, 1, 1)`. Per-primitive outputs (`PSG1`) are
//!   the console's, without `SV_PrimitiveID` and `SV_CullPrimitive` (the
//!   console never exports them).

mod bitcode;
mod container;
#[cfg(all(test, windows))]
mod diff_tests;
#[cfg(all(test, windows))]
mod image_tests;
#[cfg(test)]
mod memory_tests;
#[cfg(all(windows, any(test, feature = "host")))]
pub mod host;
mod lower;
mod sampler;
#[cfg(all(test, windows))]
mod scratch_tests;
mod shader;
mod spin;
mod threading;
#[cfg(windows)]
pub mod validator;
#[cfg(all(test, windows))]
mod valu_tests;
mod wave;
#[cfg(all(test, windows))]
mod wave_tests;
#[cfg(test)]
mod wave64_tests;

use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io, ir, resource, structure};

pub use sampler::StaticSampler;
pub use shader::{ROOT_CBV_SPACE, SAMPLER_SPACE, STRIDE_SPACE, USER_DATA_SPACE};

/// The total loop step budget per invocation without an override: 2^20.
pub const DEFAULT_LOOP_LIMIT: u32 = lower::LOOP_LIMIT;

/// The total back-edge step count after which translated loops are left, per
/// invocation: [`DEFAULT_LOOP_LIMIT`].
///
/// `VICTORY_SHADER_LOOP_LIMIT` (decimal or `0x` hex, at least 1) sets it
/// instead in any build. A loop that hits the limit in every lane of a
/// full-screen pass still runs for seconds (2^20 times a 4K screen is about
/// 10^13 iterations); a much lower limit shows whether such loops are what
/// makes a frame slow. The value is part of the translation, so the shader
/// cache must key on it (`victory-gpu`'s does).
pub fn loop_limit() -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        parse_loop_limit(std::env::var("VICTORY_SHADER_LOOP_LIMIT").ok().as_deref())
    })
}

/// The spin-wait limit without an override: 4096.
pub const DEFAULT_SPIN_LIMIT: u32 = 1 << 12;

/// The iteration count after which a loop that only waits for memory to
/// change (a spin wait, see `spin.rs`) is left: [`DEFAULT_SPIN_LIMIT`], or
/// `VICTORY_SHADER_SPIN_LIMIT` (decimal or `0x` hex, at least 1), never
/// more than [`loop_limit`]. On the console the write such a loop waits
/// for comes within microseconds; on the host, when it never comes (the
/// writer's view is another host resource, null, or the load was hoisted),
/// 2^20 uncached reads per lane cost about a second per wave, which is
/// what world frames spent per host submission. 4096 reads are a few
/// milliseconds. Part of the translation: the shader cache keys on it.
pub fn spin_limit() -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        let v = std::env::var("VICTORY_SHADER_SPIN_LIMIT").ok();
        parse_limit(v.as_deref(), DEFAULT_SPIN_LIMIT).min(loop_limit())
    })
}

/// `VICTORY_SHADER_LOOP_LIMIT`'s value; the default when it is unset,
/// unreadable or 0. Pure, for testing.
fn parse_loop_limit(value: Option<&str>) -> u32 {
    parse_limit(value, lower::LOOP_LIMIT)
}

/// A limit's value from its variable; `default` when it is unset,
/// unreadable or 0.
fn parse_limit(value: Option<&str>, default: u32) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    match parsed {
        Some(n) if n > 0 => n,
        _ => default,
    }
}

/// A translated shader and what the host binds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostShader {
    pub stage: HwStage,
    /// The DXIL container, hashed as D3D12 expects.
    pub dxil: Vec<u8>,
    /// The user-data dwords the shader reads.
    pub user_data: Vec<UserData>,
    /// Immediate samplers: the four sampler words of `s<n>`.
    pub samplers: Vec<[u32; 4]>,
    /// The same samplers decoded, `s<n>` at position n: the static sampler
    /// descriptions the host's root signature declares.
    pub static_samplers: Vec<StaticSampler>,
    /// Root CBVs the shader reads: root parameter indices, each bound as
    /// `b<index>` in [`ROOT_CBV_SPACE`].
    pub root_cbvs: Vec<u8>,
    /// Root SRVs (raw buffers) the shader reads: `t<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_srvs: Vec<u8>,
    /// True when the shader reads the runtime stride table (`t0` in
    /// [`STRIDE_SPACE`]).
    pub stride_table: bool,
    /// Root UAVs (raw buffers) the shader reads or writes: `u<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_uavs: Vec<u8>,
}

impl HostShader {
    /// Whether the DXIL requires direct indexing of the resource and sampler
    /// heaps, in that order. Static samplers need no indexed sampler heap.
    pub fn directly_indexed_heaps(&self) -> (bool, bool) {
        let features = Container::parse(&self.dxil)
            .ok()
            .and_then(|c| {
                let part = c.part(FourCC::SFI0)?;
                crate::dxbc::u64_at(part.data, 0).ok()
            })
            .unwrap_or(0);
        (
            features & shader::flags::FEATURE_RESOURCE_HEAP != 0,
            features & shader::flags::FEATURE_SAMPLER_HEAP != 0,
        )
    }

    /// Size of the user-data buffer the shader declares, in dwords (a
    /// multiple of 4; 0 when it reads none). Root constants must cover it.
    pub fn user_data_dwords(&self) -> u32 {
        let n = self
            .user_data
            .iter()
            .map(|u| u.index + 1)
            .max()
            .unwrap_or(0);
        n.next_multiple_of(4)
    }
}

/// One dword of user data and the root parameter that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserData {
    /// Dword in the user-data buffer (the console's user-data index).
    pub index: u32,
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// `Constants` or `Table`, or `ResourceHeap` / `SamplerHeap` for a
    /// heap's guard slot (see the module notes for the value).
    pub root: RootType,
    /// Dword within the parameter.
    pub dword: u32,
}

/// How a Wave64 compute shader is mapped onto the host device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Wave64Mode {
    /// Emit native `WaveSize(64)` when compute lane semantics require it.
    #[default]
    Native,
    /// Map virtual 64-lane waves onto 32-lane host waves (`WaveSize(32)`).
    Wave32,
    /// Emulate 64-lane waves in group-shared memory without `WaveSize`.
    Emulate,
}

/// Options controlling DXIL emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TranslateOptions {
    pub wave64: Wave64Mode,
}

/// Translates the vertex, pixel, compute or mesh program of a console
/// shader container with default options ([`Wave64Mode::Native`]).
pub fn translate(container: &[u8]) -> Result<HostShader, ShaderError> {
    translate_with(container, TranslateOptions::default())
}

/// Translates the vertex, pixel, compute or mesh program of a console
/// shader container using `options`.
pub fn translate_with(
    container: &[u8],
    options: TranslateOptions,
) -> Result<HostShader, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c
        .part(FourCC::DXIL)
        .ok_or(ShaderError::Malformed("no DXIL part"))?;
    let native = crate::native_programs(dxil.data)?
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms
            )
        })
        .ok_or(ShaderError::Unsupported {
            offset: 0,
            what: "a container without a vertex, pixel, compute or mesh program",
        })?;
    // A compute program whose root arguments all sit past the 16 user SGPRs
    // finds them through the spill table `s[0:1]` points to.
    let spilled = native.stage == HwStage::Cs
        && !native.layout.is_empty()
        && native
            .layout
            .iter()
            .all(|r| native.stage.user_sgpr(r.user_data) >= 16);
    let p = ir::build_with(&native.patched_code(), native.wave64, native.vgprs, spilled)?;
    let res = resource::track(&p, &native.layout, native.stage)?;
    let io = io::raise(&p, &native)?;
    let st = structure::structure(&p)?;
    emit_with(&c, &native, &p, &st, &io, &res, options)
}

/// Emits a program whose earlier stages already ran with default options.
pub fn emit(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<HostShader, ShaderError> {
    emit_with(c, native, p, st, io, res, TranslateOptions::default())
}

/// Emits a program whose earlier stages already ran using `options`.
pub fn emit_with(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    options: TranslateOptions,
) -> Result<HostShader, ShaderError> {
    let original = emit_program(c, native, p, st, io, res, options);
    // Retry unsupported programs only. Threading can separate an unknown
    // register value from its apparent use on a mutually exclusive path.
    if !matches!(original, Err(ShaderError::Unsupported { .. })) {
        return original;
    }
    let edges = threading::constant_edges(p, st);
    if edges.is_empty() {
        return original;
    }
    let p = ir::thread_edges(p, &edges)?;
    let st = structure::structure(&p)?;
    let io = io::raise(&p, native)?;
    let res = resource::track(&p, &native.layout, native.stage)?;
    emit_program(c, native, &p, &st, &io, &res, options)
}

fn emit_program(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    options: TranslateOptions,
) -> Result<HostShader, ShaderError> {
    let sig = |fourcc| -> Result<Vec<container::SigElement>, ShaderError> {
        match c.part(fourcc) {
            Some(part) => signature(part.data),
            None => Ok(Vec::new()),
        }
    };
    let pixel = native.stage == HwStage::Ps;
    let mut inputs = sig(FourCC::ISG1)?;
    let mut outputs = sig(FourCC::OSG1)?;
    // Interpolation as DXC declares it by default; pixel inputs that are
    // read get the mode their barycentrics say. Conservative depth outputs
    // (`SV_DepthGreaterEqual`/`SV_DepthLessEqual`, kinds 18/19) require
    // `SV_Position` to use `linear_noperspective_centroid` (5) rather than
    // `linear_noperspective` (4) unless the shader runs at sample frequency.
    let conservative_depth = pixel && outputs.iter().any(|o| matches!(o.kind, 18 | 19));
    let psv_modes = c
        .part(FourCC::PSV0)
        .and_then(|p| psv_interp(p.data, pixel));
    let interpolated = if pixel { &mut inputs } else { &mut outputs };
    for e in interpolated.iter_mut() {
        e.interp = match (e.kind, e.comp) {
            (3, _) => {
                if conservative_depth {
                    5
                } else {
                    4
                }
            }
            // Constant, as DXC declares it (the AMD compiler crashed on a
            // front face without a mode).
            (13, _) => 1,
            // Until a read says which mode (`lower`).
            (28, _) => 2,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
        if let Some(modes) = &psv_modes
            && let Some(&(_, _, _, m)) = modes
                .iter()
                .find(|&&(row, col, kind, m)| row == e.row && col == e.col && kind == e.kind && m != 0)
        {
            e.interp = if e.kind == 3 && conservative_depth && m == 4 {
                5
            } else {
                m
            };
        }
    }
    harmonize_row_interp(interpolated);
    // Per-primitive outputs are constant. The console never exports a
    // primitive id or cull flag a mesh shader writes (it culls with the
    // primitive export), so those rows are left out.
    let mut prims = sig(FourCC::PSG1)?;
    prims.retain(|e| !matches!(e.kind, 10 | 30));
    for e in &mut prims {
        e.interp = 1;
    }
    // A mesh group's id arrives multiplied by max(threads, vertices,
    // primitives) (the code divides by that); the thread count is in PSV0
    // from runtime info version 2 on.
    // shortcut: older containers fall back to the group's lane count, right
    // when the HLSL group fills its waves (Grand Theft Auto VI's do).
    let mesh_stride = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            threads.map_or(ms.threads, |t| {
                t.max(ms.max_vertices).max(ms.max_primitives)
            })
        }
        _ => 0,
    };
    let l = lower::lower(&lower::Parts {
        native,
        p,
        st,
        io,
        res,
        inputs,
        outputs,
        prims,
        mesh_stride,
        options,
    })?;
    Ok(HostShader {
        stage: native.stage,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: l
            .samplers
            .iter()
            .enumerate()
            .map(|(n, &w)| {
                let mut s = StaticSampler::from_words(w);
                if l.cmp_samplers >> n & 1 != 0 {
                    s.filter |= 0x80;
                }
                s
            })
            .collect(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        stride_table: l.stride_table,
        root_uavs: l.root_uavs,
    })
}

/// DXIL requires every signature element packed into the same row to share an
/// interpolation mode. Integer attributes and `SV_IsFrontFace` can only be
/// `Constant` (1), so any float sibling packed into their row must be `1` as
/// well; after pixel-shader lowering, unread siblings also inherit the actual
/// interpolation mode of a read element in the same row.
pub(super) fn harmonize_row_interp(elements: &mut [container::SigElement]) {
    let mut rows: Vec<u32> = elements.iter().filter_map(|e| e.row).collect();
    rows.sort_unstable();
    rows.dedup();
    for row in rows {
        let forced_constant = elements
            .iter()
            .any(|e| e.row == Some(row) && (e.kind == 13 || (e.kind == 0 && e.comp != 9)));
        let mode = if forced_constant {
            Some(1)
        } else {
            elements
                .iter()
                .find(|e| e.row == Some(row) && e.usage != 0 && e.interp != 0)
                .map(|e| e.interp)
        };
        if let Some(m) = mode {
            for e in elements.iter_mut() {
                if e.row == Some(row) && e.interp != 0 && e.usage == 0 {
                    e.interp = m;
                }
            }
        }
    }
}

/// Interpolation modes of the input (for pixel) or output (for vertex/mesh)
/// signature elements carried in a container's `PSV0` part: `(row, col, kind,
/// interp)`.
fn psv_interp(part: &[u8], pixel: bool) -> Option<Vec<(Option<u32>, u8, u8, u8)>> {
    use crate::dxbc::u32_at;
    let info_sz = u32_at(part, 0).ok()? as usize;
    if info_sz < 32 {
        return None;
    }
    let info = part.get(4..4 + info_sz)?;
    let (n_in, n_out, n_prim) = (info[28] as usize, info[29] as usize, info[30] as usize);
    let mut pos = 4 + info_sz;
    let n_bind = u32_at(part, pos).ok()? as usize;
    pos += 4;
    if n_bind > 0 {
        let bsz = u32_at(part, pos).ok()? as usize;
        pos = pos.checked_add(4)?.checked_add(n_bind.checked_mul(bsz)?)?;
    }
    let str_sz = u32_at(part, pos).ok()? as usize;
    pos = pos.checked_add(4)?.checked_add(str_sz)?;
    let idx_cnt = u32_at(part, pos).ok()? as usize;
    pos = pos.checked_add(4)?.checked_add(idx_cnt.checked_mul(4)?)?;
    if n_in + n_out + n_prim == 0 {
        return Some(Vec::new());
    }
    let esz = u32_at(part, pos).ok()? as usize;
    pos += 4;
    if esz < 14 {
        return None;
    }
    let (start_i, count) = if pixel { (0, n_in) } else { (n_in, n_out) };
    let mut out = Vec::with_capacity(count);
    for i in start_i..start_i + count {
        let off = pos.checked_add(i.checked_mul(esz)?)?;
        let e = part.get(off..off + esz)?;
        let alloc = e[10] & 0x40 != 0;
        let row = alloc.then_some(u32::from(e[9]));
        let col = if alloc { (e[10] >> 4) & 3 } else { 0 };
        out.push((row, col, e[11], e[13]));
    }
    Some(out)
}

/// The rows of an `ISG1`/`OSG1` part.
fn signature(part: &[u8]) -> Result<Vec<container::SigElement>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        let mask = *part
            .get(at + 24)
            .ok_or(ShaderError::Malformed("truncated"))?;
        if u32_at(part, at + 28)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a minimum-precision signature element",
            });
        }
        if u32_at(part, at)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a signature element of another stream",
            });
        }
        let kind = match u32_at(part, at + 12)? {
            0 => 0,
            1 => 3,
            2 => 6,
            3 => 7,
            4 => 4,
            5 => 5,
            6 => 1,
            7 => 10,
            8 => 2,
            9 => 13,
            10 => 12,
            64 => 16,
            65 => 17,
            66 => 14,
            67 => 19,
            68 => 18,
            69 => 20,
            70 => 15,
            23 => 28, // barycentrics
            25 => 30, // cull primitive
            _ => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "a system value in the signature",
                });
            }
        };
        // 16-bit elements are declared 32-bit; reading or writing one
        // converts (16-bit values travel packed on the console).
        let raw = u32_at(part, at + 16)?;
        let comp = match (kind, raw) {
            // DXIL declares the front face a uint (the validator insists).
            (13, _) => 5,
            (_, 1 | 4) => 5,
            (_, 2 | 5) => 4,
            (_, 3 | 6) => 9,
            (_, c) => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match c {
                        0 => "a signature element of unknown type",
                        _ => "a 64-bit signature element",
                    },
                });
            }
        };
        let register = u32_at(part, at + 20)?;
        out.push(container::SigElement {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            kind,
            comp,
            interp: 0,
            row: (register != u32::MAX).then_some(register),
            col: mask.trailing_zeros().min(3) as u8,
            cols: mask.count_ones().max(1) as u8,
            usage: 0,
            half: (4..=6).contains(&raw),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn direct_heap_requirements_are_independent() {
        let mut shader = super::HostShader {
            stage: crate::HwStage::Cs,
            dxil: super::container::build(&[]),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            stride_table: false,
            root_uavs: Vec::new(),
        };
        assert_eq!(shader.directly_indexed_heaps(), (false, false));
        for (features, expected) in [
            (0u64, (false, false)),
            (0x0200_0000, (true, false)),
            (0x0400_0000, (false, true)),
            (0x0600_0000, (true, true)),
        ] {
            shader.dxil = super::container::build(&[(b"SFI0", &features.to_le_bytes())]);
            assert_eq!(shader.directly_indexed_heaps(), expected);
        }
    }

    use super::bitcode::Attr;
    use super::lower::LOOP_LIMIT;
    use super::parse_loop_limit;

    #[test]
    fn the_loop_limit_switch_parses_or_falls_back() {
        assert_eq!(parse_loop_limit(None), LOOP_LIMIT);
        assert_eq!(parse_loop_limit(Some("4096")), 4096);
        assert_eq!(parse_loop_limit(Some(" 0x1000 ")), 4096);
        assert_eq!(parse_loop_limit(Some("1")), 1);
        for bad in ["", "0", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_limit(Some(bad)), LOOP_LIMIT, "{bad:?}");
        }
    }

    #[test]
    fn the_spin_limit_parses_with_its_own_default() {
        use super::{DEFAULT_SPIN_LIMIT, parse_limit};
        assert_eq!(parse_limit(None, DEFAULT_SPIN_LIMIT), 4096);
        assert_eq!(parse_limit(Some("0x40"), DEFAULT_SPIN_LIMIT), 64);
        assert_eq!(parse_limit(Some("0"), DEFAULT_SPIN_LIMIT), 4096);
    }
    use super::container::SigElement;
    use super::shader::Shader;
    use super::*;
    use crate::native::{ComputeIo, StageIo};

    /// A pixel shader writing SV_StencilRef (channel y of `exp mrtz`).
    /// DXC sets ShaderFlags.StencilRef (bit 11) for a pixel shader whose
    /// output signature has SV_StencilRef, and the feature bit
    /// D3D_SHADER_FEATURE_STENCIL_REF (0x200) in SFI0; the validator checks
    /// both against the module.
    #[test]
    fn a_stencil_export_declares_the_stencil_ref_feature() {
        let words = [
            0x7E00_02F2u32, // v_mov_b32 v0, 1.0
            0x7E02_0285,    // v_mov_b32 v1, 5
            0xF800_1883,
            0x0000_0100, // exp mrtz v0, v1, off, off done vm
            0xBF81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, kind, comp| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp: 0,
            row: None,
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let osg1 =
            container::signature(&[el("SV_Depth", 17, 9), el("SV_StencilRef", 20, 5)], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();
        // dxil.dll, where there is one, checks the flags against the module.
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
        let host = Container::parse(&out.dxil).unwrap();
        let sfi0 = host.part(FourCC::SFI0).unwrap().data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_ne!(features & 0x200, 0, "SFI0 {features:#x}: no StencilRef");
    }

    /// A wave64 compute program that writes only EXEC's high dword
    /// (`s_mov_b32 exec_hi, 0`) before a vector instruction. The new mask was
    /// opaque and keyed by the unchanged low dword, so building its lane bit
    /// needed the mask's own value: lowering recursed until the stack
    /// overflowed, which takes the whole process down (found by
    /// random-input testing). Now that low dword, seen with two high
    /// dwords, has no mask, and the instruction fails like any other that
    /// cannot be expressed: its result is missing, and only a side effect
    /// that uses it would stop the translation.
    #[test]
    fn an_exec_high_write_does_not_overflow_the_stack() {
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, v0; s_endpgm
        let words = [0xBEFF_0380u32, 0x7E02_0300, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        if let Err(e) = emit(&c, &native, &p, &st, &io, &res) {
            assert!(e.to_string().contains("its own value"), "{e}");
        }
    }

    /// A wave64 compute program: `body`, then a store of v1 to a UAV.
    fn stores_v1(body: &[u32]) -> Result<(), ShaderError> {
        use crate::native::RootParam;
        use crate::{ResourceType, RootType, UsedResource};
        // s_mov_b32 s1, 0; s_load_dwordx4 s[8:11], s[0:1], 0
        let mut words = vec![0xBE81_0380, 0xF408_0200, 0xFA00_0000];
        words.extend_from_slice(body);
        // v_lshlrev_b32 v3, 2, v0; buffer_store_dword v1, v3, s[8:11], 0 offen;
        // s_endpgm
        words.extend_from_slice(&[0x3406_0082, 0xE070_1000, 0x8002_0103, 0xBF81_0000]);
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 4,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 2,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        emit(&c, &native, &p, &st, &io, &res).map(|_| ())
    }

    /// The high dword of a wave64 mask rewritten alone (`s_mov_b32 vcc_hi`
    /// or `exec_hi`): lanes 32-63 took the mask the low dword was first
    /// written with (the compare's, EXEC's on entry). Now the store refuses;
    /// without the write it translates.
    #[test]
    fn a_mask_whose_high_dword_is_rewritten_is_refused() {
        // v_cmp_eq_u32 vcc, 0, v0; v_mov_b32 v2, 11; v_cndmask_b32 v1, 0, v2, vcc
        let select = [0x7D84_0080, 0x7E04_028B, 0x0202_0480];
        // The same with v_add_co_ci_u32 v1, vcc, 0, v2, vcc.
        let carry = [0x7D84_0080, 0x7E04_028B, 0x5002_0480];
        // v_mov_b32 v1, 7
        let mov = [0x7E02_0287];
        for body in [&select[..], &carry, &mov] {
            stores_v1(body).unwrap();
        }
        // s_mov_b32 vcc_hi, 0 after the compare.
        let select_hi = [0x7D84_0080, 0xBEEB_0380, 0x7E04_028B, 0x0202_0480];
        let carry_hi = [0x7D84_0080, 0xBEEB_0380, 0x7E04_028B, 0x5002_0480];
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, 7, then the same after
        // s_mov_b64 s[2:3], exec.
        let exec_hi = [0xBEFF_0380, 0x7E02_0287];
        let exec_hi_saved = [0xBE82_047E, 0xBEFF_0380, 0x7E02_0287];
        for body in [&select_hi[..], &carry_hi, &exec_hi, &exec_hi_saved] {
            assert!(stores_v1(body).is_err(), "{body:x?}");
        }
    }

    /// A carry-in reading a join of compares after a divergent branch. A
    /// lane that skipped the inner compare has VCC clear there (VOPC clears
    /// inactive lanes), not the first compare's bit, so the join is no
    /// per-lane phi: lowering refuses it.
    #[test]
    fn a_carry_in_from_a_divergent_join_is_refused() {
        let body = [
            0x7E04_028B, // v_mov_b32 v2, 11
            0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
            0xBE82_376A, // s_andn1_saveexec_b64 s[2:3], vcc
            0xBF88_0001, // s_cbranch_execz 1
            0x7D84_0081, // v_cmp_eq_u32 vcc, 1, v0
            0x88FE_027E, // s_or_b64 exec, exec, s[2:3]
            0x5002_0480, // v_add_co_ci_u32 v1, vcc, 0, v2, vcc
        ];
        let e = stores_v1(&body).unwrap_err().to_string();
        assert!(e.contains("a carry-in from an untracked mask"), "{e}");
    }

    #[test]
    fn indexed_move_initializes_an_input_before_its_first_use() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Case {
            Complete,
            NonzeroBase,
            WrongStart,
            SkippedSlot,
            Decrement,
            Wrap,
            UnknownStart,
            ShortLoop,
            ClobberedScc,
            EarlyExit,
            DivergentExit,
            MaskedWrite,
            SourceIsOld,
            ReadBeforeWrite,
            ReadOnBackedge,
        }
        use Case::*;

        // A two-block do-while loop writes v[M0] from v2, then exports v1.
        // On the first two passes M0 selects v0 and v1 respectively. Both
        // registers initially contain raw perspective-center barycentrics.
        let words = |case| {
            let init = match case {
                WrongStart => 0xbe80_0382,   // s_mov_b32 s0, 2
                Wrap => 0xbe80_03c1,         // s_mov_b32 s0, -1
                UnknownStart => 0x8000_8080, // s_add_u32 s0, 0, 0
                _ => 0xbe80_0380,            // s_mov_b32 s0, 0
            };
            let mut w = vec![init, 0x7e04_0281, 0xbf82_0000]; // v_mov_b32 v2, 1; s_branch header
            let latch = w.len();
            if case == EarlyExit {
                w.extend([0xbf06_8000, 0xbf85_0000]); // s_cmp_eq_u32 s0, 0; s_cbranch_scc1 exit
            }
            w.push(0xbe80_0302); // s_mov_b32 s0, s2
            let header = w.len();
            w[2] |= (header - 3) as u32;
            w.push(0xbefc_0300); // s_mov_b32 m0, s0
            w.push(match case {
                SkippedSlot => 0x8102_8200, // s_add_i32 s2, s0, 2
                Decrement => 0x8102_c100,   // s_add_i32 s2, s0, -1
                _ => 0x8102_8100,           // s_add_i32 s2, s0, 1
            });
            if case == ReadBeforeWrite {
                w.push(0x7e06_0301);
            } // v_mov_b32 v3, v1
            if case == MaskedWrite {
                w.extend([0x7d84_0480, 0xbe84_3c6a]); // v_cmp_eq_u32 vcc, 0, v2; s_and_saveexec_b32 s4, vcc_lo
            }
            w.push(match case {
                SourceIsOld => 0x7e00_8501, // v_movreld_b32 v0, v1
                NonzeroBase => 0x7e02_8502, // v_movreld_b32 v1, v2
                _ => 0x7e00_8502,           // v_movreld_b32 v0, v2
            });
            if case == MaskedWrite {
                w.push(0x887e_047e);
            } // s_or_b32 exec_lo, exec_lo, s4
            if case == ReadOnBackedge {
                w.push(0x7e06_0301);
            } // v_mov_b32 v3, v1
            let branch = if case == DivergentExit {
                w.push(0x7d84_0480); // v_cmp_eq_u32 vcc, 0, v2
                0xbf87_0000 // s_cbranch_vccnz latch
            } else {
                w.push(if matches!(case, ShortLoop | NonzeroBase) {
                    0xbf06_8102
                } else {
                    0xbf06_8402
                }); // s_cmp_eq_u32 s2, 1/4
                if case == ClobberedScc {
                    w.push(0xbf06_8000); // s_cmp_eq_u32 s0, 0 exits on the first pass
                }
                0xbf84_0000 // s_cbranch_scc0 latch
            };
            let rel = latch as i32 - (w.len() as i32 + 1);
            w.push(branch | u32::from(rel as u16));
            let exit = w.len();
            if case == EarlyExit {
                w[latch + 1] |= (exit - latch - 2) as u32;
            }
            let output = if matches!(case, ReadBeforeWrite | ReadOnBackedge) {
                3
            } else {
                1
            };
            w.extend([0xf800_1801, output, 0xbf81_0000]); // exp mrt0.x done vm; s_endpgm
            w
        };
        let target = SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let sig = super::container::signature(&[target], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &sig)]);
        let c = Container::parse(&bytes).unwrap();
        for case in [
            Complete,
            NonzeroBase,
            WrongStart,
            SkippedSlot,
            Decrement,
            Wrap,
            UnknownStart,
            ShortLoop,
            ClobberedScc,
            EarlyExit,
            DivergentExit,
            MaskedWrite,
            SourceIsOld,
            ReadBeforeWrite,
            ReadOnBackedge,
        ] {
            let native = NativeProgram {
                stage: HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: StageIo::Ps {
                    inputs: vec![],
                    input_addr: 8,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words(case).iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let res = resource::track(&p, &native.layout, native.stage).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let st = structure::structure(&p).unwrap();
            let shader = emit(&c, &native, &p, &st, &io, &res);
            if matches!(case, Complete | NonzeroBase) {
                let shader = shader.unwrap();
                assert!(
                    Container::parse(&shader.dxil)
                        .unwrap()
                        .part(FourCC::DXIL)
                        .is_some()
                );
                #[cfg(windows)]
                if let Ok(mut validator) =
                    validator::Validator::load(validator::Validator::default_path().as_deref())
                {
                    let mut dxil = shader.dxil;
                    validator.validate(&mut dxil).unwrap();
                    eprintln!("synthetic {case:?} SDK valid bytes={}", dxil.len());
                }
            } else {
                eprintln!("synthetic {case:?} {:?}", shader.as_ref().err());
                assert!(
                    matches!(
                        shader,
                        Err(ShaderError::Unsupported {
                            what: "pull-model barycentrics",
                            ..
                        })
                    ),
                    "{case:?}"
                );
            }
        }
    }

    /// A pixel shader writing a constant colour.
    fn constant_color() -> Vec<u8> {
        let mut s = Shader::new(true);
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        });
        let b = s.entry;
        for (c, v) in [1.0f32, 0.5, 0.25, 1.0].into_iter().enumerate() {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(c as u8),
                s.m.f32(v.to_bits()),
            ];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, super::bitcode::Term::Ret);
        s.finish()
    }

    #[test]
    #[cfg(windows)]
    fn stub_pipeline_is_created() {
        let Ok(host) = super::host::Host::new(false) else {
            eprintln!("no SM 6.6 device; skipped");
            return;
        };
        let ps = super::HostShader {
            stage: crate::HwStage::Ps,
            dxil: constant_color(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            stride_table: false,
            root_uavs: Vec::new(),
        };
        let file = |var: &str, stage| {
            std::env::var(var).ok().map(|p| super::HostShader {
                stage,
                dxil: std::fs::read(p).unwrap(),
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                stride_table: false,
                root_uavs: Vec::new(),
            })
        };
        let ps = file("VICTORY_REF_PS", crate::HwStage::Ps).unwrap_or(ps);
        let vs = file("VICTORY_REF_VS", crate::HwStage::Vs)
            .unwrap_or_else(|| super::host::stub_vs(&ps).unwrap());
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil.clone()).unwrap();
        }
        host.pipeline(&vs, &ps).unwrap();
        // The runtime checks the container hash: no hash or a wrong one is
        // refused. (Recent runtimes also take sixteen 0x01 bytes, a bypass
        // value; ours works with older ones too.)
        let mut wrong = ps.dxil[4..20].to_vec();
        wrong[0] ^= 1;
        for digest in [&[0u8; 16][..], &wrong] {
            let mut other = ps.clone();
            other.dxil[4..20].copy_from_slice(digest);
            assert!(
                host.pipeline(&vs, &other).is_err(),
                "digest {digest:02x?} accepted"
            );
        }
    }

    #[test]
    #[cfg(windows)]
    fn constant_color_validates() {
        let mut c = constant_color();
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        let ours = c[4..20].to_vec();
        v.validate(&mut c).unwrap();
        assert_eq!(c[4..20], ours[..], "our hash differs from the validator's");
    }

    /// A pixel shader interpolating at sample positions (PERSP_SAMPLE
    /// barycentrics, `SPI_PS_INPUT_ADDR` bit 0) runs at sample frequency:
    /// DXC's PSV0 says so in PSInfo.SampleFrequency (runtime info byte 1)
    /// whenever an input is interpolated per sample.
    #[test]
    fn sample_interpolated_inputs_set_psv_sample_frequency() {
        let words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ];
        let psv_of = |input_addr: u32| {
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: false,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half: false,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let out = super::emit(&c, &native, &p, &st, &io, &res).unwrap();
            // dxil.dll, where there is one, checks PSV0 against the module.
            #[cfg(windows)]
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut out.dxil.clone()).unwrap();
            }
            let host = super::Container::parse(&out.dxil).unwrap();
            host.part(super::FourCC::PSV0).unwrap().data.to_vec()
        };
        // Runtime info starts after its size; PSInfo is {DepthOutput, SampleFrequency}.
        let center = psv_of(1 << 1);
        assert_eq!(center[4 + 1], 0, "center interpolation is per pixel");
        let sample = psv_of(1 << 0);
        assert_eq!(sample[4 + 1], 1, "PSInfo.SampleFrequency");
    }

    /// A pixel shader reading channel z of an input whose element starts at
    /// w. Looking the channel up subtracted the element's first channel
    /// before checking the channel lies in it, so in a build with overflow
    /// checks (tests, debug) any read below an element that starts at z or
    /// w panicked, whichever row it was in (found by random-input testing).
    #[test]
    fn reading_a_channel_below_an_input_element_does_not_underflow() {
        // v_interp_mov_f32 v3, p0, attr0.z; s_endpgm
        let words = [0xC80E_0202u32, 0xBF81_0000];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 0,
                    mask: 0b1000,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let w_only = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 3,
            cols: 1,
            usage: 0,
            half: false,
        };
        let isg1 = super::container::signature(&[w_only], true);
        let bytes = crate::dxbc::build(&[(super::FourCC::ISG1, &isg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        // Translated or refused with a reason; it must not panic.
        let _ = super::emit(&c, &native, &p, &st, &io, &res);
    }

    #[test]
    fn structured_buffer_unknown_stride_uses_runtime_stride_table() {
        let words: &[u32] = &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
            0xF408_0302,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
            0x7E00_0282, // v_mov_b32 v0, 2
            0x7E02_0284, // v_mov_b32 v1, 4
            0xE030_3000,
            0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
            0x7E00_0281, // v_mov_b32 v0, 1
            0x7E02_0288, // v_mov_b32 v1, 8
            0xE070_3000,
            0x8003_0200, // buffer_store_dword v2, v[0:1], s[12:15], 0 idxen offen
            0x7E00_0282, // v_mov_b32 v0, 2
            0x7E02_0280, // v_mov_b32 v1, 0
            0x7E06_028A, // v_mov_b32 v3, 10
            0xE0C8_3000,
            0x8003_0300, // buffer_atomic_add v3, v[0:1], s[12:15], 0 idxen offen
            0xE0C8_3000,
            0x8003_0300, // buffer_atomic_add v3, v[0:1], s[12:15], 0 idxen offen
            0xBF81_0000, // s_endpgm
        ];
        let native = super::NativeProgram {
            stage: super::HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![
                crate::RootParam {
                    root: crate::RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 4,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                crate::RootParam {
                    root: crate::RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 4,
                    slot: 1,
                },
            ],
            used: vec![
                crate::UsedResource {
                    kind: crate::ResourceType::Srv,
                    root: crate::RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                },
                crate::UsedResource {
                    kind: crate::ResourceType::Uav,
                    root: crate::RootType::Table,
                    table_offset_dwords: 8,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                },
            ],
            io: crate::native::StageIo::Cs(crate::native::ComputeIo {
                rsrc2: 5 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let l = super::lower::lower(&super::lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            options: super::TranslateOptions::default(),
        })
        .unwrap();
        assert!(l.stride_table);
        let mut dxil = l.dxil;
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut dxil).unwrap();
        }
        #[cfg(windows)]
        if let Ok(host) = super::host::Host::new(false) {
            let cs = super::HostShader {
                stage: super::HwStage::Cs,
                dxil,
                user_data: l.user_data,
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: l.root_cbvs,
                root_srvs: l.root_srvs,
                stride_table: l.stride_table,
                root_uavs: l.root_uavs,
            };
            let (root, pso) = host.compute_pipeline(&cs).unwrap();
            let srv_data: Vec<u32> = (100..112).collect();
            let mut ud = vec![0u32; cs.user_data_dwords().max(5) as usize];
            for u in &cs.user_data {
                if let Some(g) = super::host::guard(u.root, 2, 0) {
                    ud[u.index as usize] = g;
                }
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &super::host::Dispatch {
                        roots: vec![
                            (0, super::host::Root::Constants(ud)),
                            (
                                1,
                                super::host::Root::View {
                                    srv: true,
                                    data: vec![12, 12],
                                },
                            ),
                        ],
                        views: &[
                            super::host::View::Buffer {
                                data: &srv_data,
                                stride: super::host::Stride::Raw,
                            },
                            super::host::View::RawUav(64),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            assert_eq!(out[0][5], 107);
            assert_eq!(out[0][6], 20);
        }
    }

    /// A mesh program of `threads` lanes declaring `max` vertices and
    /// primitives; each lane exports its vertex (the position, then
    /// `vertex_rows` four-component rows) and its primitive (`prim_rows`
    /// rows), translated.
    fn mesh(
        threads: u32,
        max: u32,
        vertex_rows: u8,
        prim_rows: u8,
    ) -> Result<HostShader, ShaderError> {
        use crate::native::{MeshIo, Usage, VsOutput};
        let element = |name: &str, row: u8, kind: u8| SigElement {
            name: name.into(),
            index: row.into(),
            kind,
            comp: 9,
            interp: 0,
            row: Some(row.into()),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        };
        let mut vertex = vec![element("SV_Position", 0, 3)];
        vertex.extend((1..=vertex_rows).map(|r| element("TEXCOORD", r, 0)));
        let prims: Vec<SigElement> = (0..prim_rows).map(|r| element("PRIMDATA", r, 0)).collect();
        let mut words: Vec<u32> = vec![
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_02F2, // v_mov_b32 v1, 1.0
            0xF800_00CF, // exp pos0 v0, v0, v0, v1
            0x0100_0000,
        ];
        let mut outputs = Vec::new();
        for n in 0..vertex_rows + prim_rows {
            // exp param<n> v1, v1, v1, v1; the last ones per-primitive.
            words.extend([0xF800_000F | (32 + u32::from(n)) << 4, 0x0101_0101]);
            let primitive = n >= vertex_rows;
            outputs.push(VsOutput {
                usage: Usage::Generic,
                row: if primitive { n - vertex_rows } else { n + 1 },
                param: if primitive { 0x80 | n } else { n },
                fp16: false,
                pack2: false,
            });
        }
        words.extend([
            0xF800_0941, // exp prim v0, off, off, off done
            0x0000_0000,
            0xBF81_0000, // s_endpgm
        ]);
        let native = NativeProgram {
            stage: HwStage::Ms,
            wave64: true,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Ms(MeshIo {
                outputs,
                clip_cntl: 0,
                threads,
                max_vertices: max,
                max_primitives: max,
                topology: 2,
                lds_bytes: 0,
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let osg1 = super::container::signature(&vertex, false);
        let psg1 = super::container::signature(&prims, false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1), (FourCC::PSG1, &psg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        emit(&c, &native, &p, &st, &io, &res)
    }

    /// The vertex and primitive maxima a translated mesh shader's `PSV0`
    /// declares (bytes 12 and 14 of the runtime info, after its size).
    fn psv_maxima(dxil: &[u8]) -> (u16, u16) {
        let c = Container::parse(dxil).unwrap();
        let d = c.part(FourCC::PSV0).unwrap().data;
        let half = |at: usize| u16::from_le_bytes([d[4 + at], d[5 + at]]);
        (half(12), half(14))
    }

    /// What `dxil.dll` says of a container; `None` without it.
    #[cfg(windows)]
    fn validated(dxil: &[u8]) -> Option<Result<(), String>> {
        let path = super::validator::Validator::default_path();
        let mut v = super::validator::Validator::load(path.as_deref()).ok()?;
        Some(v.validate(&mut dxil.to_vec()))
    }

    #[cfg(not(windows))]
    fn validated(_: &[u8]) -> Option<Result<(), String>> {
        None
    }

    /// 64 lanes declaring 256 vertices and primitives, with 28 vertex and 8
    /// primitive output components: 36 KiB at those maxima, past the 32 KiB
    /// D3D12 allows (the validator refused it: "vertex plus primitive output
    /// size is greater than 32768"). The lanes write vertices and primitives
    /// 0-63 alone, so the maxima come down to 64. Outputs that fit keep the
    /// console's maxima.
    #[test]
    fn mesh_maxima_past_the_lanes_come_down_to_fit() {
        for (rows, maxima) in [((6, 2), (64, 64)), ((1, 1), (256, 256))] {
            let s = mesh(64, 256, rows.0, rows.1).unwrap();
            match validated(&s.dxil) {
                Some(r) => r.unwrap_or_else(|e| panic!("{rows:?}: {e}")),
                None => eprintln!("no dxil.dll; validation skipped"),
            }
            assert_eq!(psv_maxima(&s.dxil), maxima, "{rows:?}");
        }
    }

    /// 128 lanes, 256 vertices and primitives, 64 vertex and 8 primitive
    /// output components: past 32 KiB even at the lane count. Still
    /// translated as before, with the maxima down to the lanes.
    #[test]
    fn mesh_outputs_past_32_kib_at_the_lane_count_keep_the_lane_maxima() {
        let s = mesh(128, 256, 15, 2).unwrap();
        assert_eq!(psv_maxima(&s.dxil), (128, 128));
    }
}

#[cfg(test)]
mod atomic_tests {
    use super::*;
    use super::container::SigElement;
    use crate::native::{ComputeIo, RootParam, StageIo};

    #[test]
    fn heap_texture_atomic_widths() {
        // A synthetic compute shader loads a 2D UAV from a root table,
        // compares or adds at (0, 0), then uses the returned register(s).
        for (opcode, mask, wide) in [
            (16u32, 3, false),
            (16, 15, true),
            (17, 1, false),
            (17, 3, true),
        ] {
            let words = [
                0xBE81_0380, // s_mov_b32 s1, 0 (table address high)
                0xF40C_0200,
                0xFA00_0000, // s_load_dwordx8 s[8:15], s[0:1], 0
                0x7E00_0280,
                0x7E02_0280, // v0, v1 = coordinates
                0x7E04_0281,
                0x7E06_0282,
                0x7E08_0283,
                0x7E0A_0284, // v2..v5 = 1..4
                0xF000_2008 | opcode << 18 | mask << 8,
                0x0002_0200,
                // Use the old value in a second atomic at the same location.
                0xF044_0008 | (if wide { 3 } else { 1 }) << 8,
                0x0002_0200,
                0xBF81_0000,
            ];
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 6,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 2,
                    threads: [32, 1, 1],
                }),
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let p = ir::build(&native.code, false, native.vgprs).unwrap();
            let bytes = crate::dxbc::build(&[]);
            let shader = emit(
                &Container::parse(&bytes).unwrap(),
                &native,
                &p,
                &structure::structure(&p).unwrap(),
                &io::raise(&p, &native).unwrap(),
                &resource::track(&p, &native.layout, native.stage).unwrap(),
            )
            .unwrap();
            let container = Container::parse(&shader.dxil).unwrap();
            let features = u64::from_le_bytes(
                container
                    .part(FourCC::SFI0)
                    .unwrap()
                    .data
                    .try_into()
                    .unwrap(),
            );
            let required = 0x1040_8000; // heap atomics, typed atomics, Int64Ops
            assert_eq!(features & required, if wide { required } else { 0 });
            #[cfg(windows)]
            if let Ok(mut v) =
                validator::Validator::load(validator::Validator::default_path().as_deref())
            {
                v.validate(&mut shader.dxil.clone()).unwrap();
            }
        }
    }

    #[cfg(windows)]
    fn validate_if_available(dxil: &[u8]) {
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        let mut copy = dxil.to_vec();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy)
            .unwrap_or_else(|e| panic!("dxil.dll rejected shader:\n{e}"));
        assert_eq!(copy[4..20], ours[..], "container hash disagrees");
    }

    fn emit_synthetic(
        parts: &[(FourCC, &[u8])],
        native: &NativeProgram,
    ) -> Result<HostShader, ShaderError> {
        let bytes = crate::dxbc::build(parts);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, native).unwrap();
        let st = structure::structure(&p).unwrap();
        emit(&c, native, &p, &st, &io, &res)
    }

    fn sig_elem(
        name: &str,
        index: u32,
        kind: u8,
        comp: u8,
        row: Option<u32>,
        col: u8,
        cols: u8,
    ) -> SigElement {
        SigElement {
            name: name.into(),
            index,
            kind,
            comp,
            interp: 0,
            row,
            col,
            cols,
            usage: 0,
            half: false,
        }
    }

    #[test]
    fn conservative_depth_sets_position_centroid_and_validates() {
        for (depth_name, depth_kind) in
            [("SV_DepthLessEqual", 18u8), ("SV_DepthGreaterEqual", 19u8)]
        {
            // v_mov_b32 v0, 0.5; exp mrtz v0, off, off, off done; s_endpgm
            let words = [
                0x7E00_02F0u32, // v_mov_b32 v0, 0.5
                0xF800_0881,    // exp mrtz (target 8), en=0x1, done=1
                0x0000_0000,    // vsrc0=v0
                0xBF81_0000,    // s_endpgm
            ];
            let native = NativeProgram {
                stage: HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: StageIo::Ps {
                    inputs: vec![],
                    input_addr: 0,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let isg1 =
                container::signature(&[sig_elem("SV_Position", 0, 3, 9, Some(0), 0, 4)], true);
            let osg1 =
                container::signature(&[sig_elem(depth_name, 0, depth_kind, 9, None, 0, 1)], false);
            let host =
                emit_synthetic(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)], &native).unwrap();
            #[cfg(windows)]
            validate_if_available(&host.dxil);
        }
    }

    #[test]
    fn row_interp_harmonizes_packed_elements_and_validates() {
        // 1) Unit check on harmonize_row_interp for packed uint + float and unread siblings.
        let mut elems = vec![
            SigElement {
                interp: 1,
                ..sig_elem("COLOR", 1, 0, 5, Some(1), 0, 1)
            },
            SigElement {
                interp: 2,
                ..sig_elem("COLOR", 2, 0, 9, Some(1), 1, 1)
            },
            SigElement {
                interp: 1,
                usage: 1,
                ..sig_elem("TEXCOORD", 0, 0, 9, Some(2), 0, 2)
            },
            SigElement {
                interp: 2,
                ..sig_elem("TEXCOORD", 1, 0, 9, Some(2), 2, 2)
            },
        ];
        harmonize_row_interp(&mut elems);
        assert_eq!(elems[1].interp, 1);
        assert_eq!(elems[3].interp, 1);

        // 2) Full PS with SV_IsFrontFace + unread float in row 1 AND a read
        //    flat float + unread float in row 2 -> validates with dxil.dll.
        // v_interp_mov_f32 v0, p0, attr0.x; exp mrt0 v0, v0, v0, v0 done; s_endpgm
        let words = [
            0xC802_0002u32, // v_interp_mov_f32 v0, p0, attr0.x
            0xF800_080F,    // exp mrt0, en=0xF, done=1
            0x0000_0000,    // vsrc0..3 = v0
            0xBF81_0000,    // s_endpgm
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 2,
                    mask: 0x3,
                    attr: 0,
                    flat: true,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let isg1 = container::signature(
            &[
                sig_elem("SV_Position", 0, 3, 9, Some(0), 0, 4),
                sig_elem("TEXCOORD", 0, 0, 9, Some(1), 0, 1),
                sig_elem("SV_IsFrontFace", 0, 13, 5, Some(1), 1, 1),
                sig_elem("TEXCOORD", 1, 0, 9, Some(2), 0, 2),
                sig_elem("TEXCOORD", 2, 0, 9, Some(2), 2, 2),
            ],
            true,
        );
        let osg1 = container::signature(&[sig_elem("SV_Target", 0, 16, 9, Some(0), 0, 4)], false);
        let host =
            emit_synthetic(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)], &native).unwrap();
        #[cfg(windows)]
        validate_if_available(&host.dxil);
    }

    #[test]
    fn signature_part_preserves_element_order_within_row_and_validates() {
        // Two elements in row 1 where the earlier element has a higher column
        // (col=3, w) than the later element (col=0, xyz).
        let row1_elems = [
            sig_elem("SV_Position", 0, 3, 9, Some(0), 0, 4),
            sig_elem("TEXCOORD", 3, 0, 9, Some(1), 3, 1),
            sig_elem("NORMAL", 0, 0, 9, Some(1), 0, 3),
        ];
        let isg1 = container::signature(&row1_elems, true);
        let parsed = signature(&isg1).unwrap();
        assert_eq!(parsed[1].name, "TEXCOORD");
        assert_eq!(parsed[2].name, "NORMAL");

        // Validate through full PS emission so dxil.dll verifies ISG1 matches
        // the DXIL signature metadata byte for byte.
        // v_mov_b32 v0, 0; exp mrt0 v0, v0, v0, v0 done; s_endpgm
        let ps_words = [0x7E00_0280u32, 0xF800_080F, 0x0000_0000, 0xBF81_0000];
        let ps_native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: ps_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let osg1 = container::signature(&[sig_elem("SV_Target", 0, 16, 9, Some(0), 0, 4)], false);
        let ps =
            emit_synthetic(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)], &ps_native).unwrap();
        #[cfg(windows)]
        validate_if_available(&ps.dxil);

        // Also validate a VS whose OSG1 has the same out-of-column row order.
        // v_mov_b32 v0, 0; exp pos0 v0..v0 done; exp param0 v0..v0; s_endpgm
        let vs_words = [
            0x7E00_0280u32,
            0xF800_00CF, // exp pos0 (target 12), en=0xF
            0x0000_0000,
            0xF800_0A0F, // exp param0 (target 32), en=0xF, done=1
            0x0000_0000,
            0xBF81_0000,
        ];
        let vs_native = NativeProgram {
            stage: HwStage::Vs,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Vs {
                inputs: vec![],
                outputs: vec![crate::native::VsOutput {
                    usage: crate::native::Usage::Generic,
                    row: 1,
                    param: 0,
                    fp16: false,
                    pack2: false,
                }],
            },
            code: vs_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let vs_osg1 = container::signature(&row1_elems, false);
        let vs = emit_synthetic(&[(FourCC::OSG1, &vs_osg1)], &vs_native).unwrap();
        #[cfg(windows)]
        validate_if_available(&vs.dxil);
    }

    #[test]
    fn compute_shader_implicit_sample_uses_valver_1_8_and_validates() {
        // Synthetic CS (8x8x1): loads a 2D texture SRV from a root table (s0)
        // and samples it with an immediate sampler (s[8:11]) via `image_sample`.
        // s_load_dwordx8 s[0:7], s[12:13], 0x0 (table in s12)
        // Wait: in resource::track, a 1-SGPR table pointer uses s_load_dwordx8
        // from s[0:1] when layout has RootType::Table at user_data=0!
        // Encode:
        //   s_mov_b32 s1, 0             (high half of 64-bit table base)
        //   s_load_dwordx8 s[4:11], s[0:1], 0x0
        //   s_mov_b32 s12, 0; s_mov_b32 s13, 0; s_mov_b32 s14, 0; s_mov_b32 s15, 0
        //   v_mov_b32 v0, 0.5; v_mov_b32 v1, 0.5
        //   image_sample v0, v[0:1], s[4:11], s[12:15] dmask:0x1 dim:SQ_RSRC_IMG_2D
        //   s_endpgm
        let words = [
            0xBE81_0380u32, // s_mov_b32 s1, 0
            0xF40C_0100,    // s_load_dwordx8 s[4:11], s[0:1], 0
            0xFA00_0000,    // soffset=NULL, offset=0
            0xBE8C_0380,    // s_mov_b32 s12, 0
            0xBE8D_0380,    // s_mov_b32 s13, 0
            0xBE8E_0380,    // s_mov_b32 s14, 0
            0xBE8F_0380,    // s_mov_b32 s15, 0
            0x7E00_02F0,    // v_mov_b32 v0, 0.5
            0x7E02_02F0,    // v_mov_b32 v1, 0.5
            // MIMG image_sample (op=0x20): w0 = 0xF000_0000 | (0x20 << 18) | (1 << 8) | (1 << 3)
            0xF080_0108,
            // w1: vaddr=v0, vdata=v0, srsrc=s[4:11] (1), ssamp=s[12:15] (3)
            (3 << 21) | (1 << 16),
            0xBF81_0000, // s_endpgm
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![crate::native::RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![crate::native::UsedResource {
                kind: crate::native::ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 8,
                dwords: 8,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [8, 8, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let cs = emit_synthetic(&[], &native).unwrap();
        // PSV0 must use RuntimeInfo3 (52 bytes) for validator 1.8.
        let parsed = Container::parse(&cs.dxil).unwrap();
        let psv0 = parsed.part(FourCC::PSV0).unwrap().data;
        assert_eq!(crate::dxbc::u32_at(psv0, 0).unwrap(), 52);
        #[cfg(windows)]
        {
            validate_if_available(&cs.dxil);
            if let Ok(host) = super::host::Host::new(false) {
                host.compute_pipeline(&cs).unwrap();
            }
        }
    }

    #[test]
    fn biased_16bit_image_sample_unpacks_f32_bias_and_f16_coords_and_validates() {
        // Synthetic PS executing `image_sample_b v0, v[1:2], s[4:11], s[12:15]
        // dmask:0x1 dim:SQ_RSRC_IMG_2D a16` where v1 is a 32-bit f32 bias
        // (-1.0) and v2 holds packed f16 (u, v) = (0.5, 0.5), then exporting v0.
        let words = [
            0xBE81_0380u32, // s_mov_b32 s1, 0
            0xF40C_0100,    // s_load_dwordx8 s[4:11], s[0:1], 0
            0xFA00_0000,    // soffset=NULL, offset=0
            0xBE8C_0380,    // s_mov_b32 s12, 0
            0xBE8D_0380,    // s_mov_b32 s13, 0
            0xBE8E_0380,    // s_mov_b32 s14, 0
            0xBE8F_0380,    // s_mov_b32 s15, 0
            0x7E02_02F3,    // v_mov_b32 v1, -1.0 (32-bit f32 bias)
            0x7E04_02FF,    // v_mov_b32 v2, lit(0x3800_3800) (packed f16 0.5, 0.5)
            0x3800_3800,
            // MIMG image_sample_b (op=0x25): dmask=0x1, dim=1 (2D), a16=1 (bit 30 of w1)
            0xF000_0000 | (0x25 << 18) | (1 << 8) | (1 << 3),
            (1 << 30) | (3 << 21) | (1 << 16) | 1, // vaddr=v1 (v[1:2]), vdata=v0
            0xF800_080F,                           // exp mrt0 v0, v0, v0, v0 done
            0x0000_0000,
            0xBF81_0000, // s_endpgm
        ];
        // Sanity-check the synthetic encoding decodes as `image_sample_b ... a16`.
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let decoded = crate::gfx10::decode(&code).unwrap();
        let sample = decoded
            .iter()
            .find(|i| i.name() == Some("image_sample_b"))
            .expect("image_sample_b decoded");
        assert!(matches!(
            sample.fields(false),
            crate::gfx10::Fields::Image {
                a16: true,
                dim: 1,
                ..
            }
        ));

        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![crate::native::RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![crate::native::UsedResource {
                kind: crate::native::ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 8,
                dwords: 8,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code,
        };
        let osg1 = container::signature(&[sig_elem("SV_Target", 0, 16, 9, Some(0), 0, 4)], false);
        let ps = emit_synthetic(&[(FourCC::OSG1, &osg1)], &native).unwrap();
        #[cfg(windows)]
        validate_if_available(&ps.dxil);
    }
}

#[cfg(test)]
mod barycentric_tests {
    use super::container::SigElement;
    use super::*;
    use crate::native::{PsInput, StageIo, Usage};

    /// `SPI_PS_INPUT_ADDR` bits: PERSP_CENTER, LINEAR_CENTER.
    const PERSP_CENTER: u32 = 1 << 1;
    const LINEAR_CENTER: u32 = 1 << 5;

    /// `exp mrt0, v<r>, v<r>, v<r>, v<r> done vm`.
    fn exp_mrt0(r: u32) -> [u32; 2] {
        [0xF800_180F, r * 0x0101_0101]
    }

    /// VOP2 `v_add_f32 v<dst>, v<a>, v<b>`.
    fn v_add_f32(dst: u32, a: u32, b: u32) -> u32 {
        3 << 25 | dst << 17 | b << 9 | (256 + a)
    }

    fn element(name: &str, kind: u8, row: u32) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp: 0,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }
    }

    /// Translates a pixel program with the input VGPRs `input_addr`
    /// enables and one float4 TEXCOORD0 attribute (attr 0, row 0).
    fn pixel(words: &[u32], input_addr: u32) -> Result<HostShader, ShaderError> {
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 0,
                    mask: 0b1111,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let mut target = element("SV_Target", 16, 0);
        target.usage = 15;
        let isg1 = container::signature(&[element("TEXCOORD", 0, 0)], true);
        let osg1 = container::signature(&[target], false);
        let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
        let res = resource::track(&p, &native.layout, native.stage)?;
        let io = io::raise(&p, &native)?;
        let st = structure::structure(&p)?;
        emit(&c, &native, &p, &st, &io, &res)
    }

    /// The translation's `SFI0` features and `ISG1` rows.
    fn parts(shader: &HostShader) -> (u64, Vec<SigElement>) {
        let c = Container::parse(&shader.dxil).unwrap();
        let sfi0 = c.part(FourCC::SFI0).unwrap().data.try_into().unwrap();
        (
            u64::from_le_bytes(sfi0),
            signature(c.part(FourCC::ISG1).unwrap().data).unwrap(),
        )
    }

    fn validate(shader: &HostShader) {
        #[cfg(windows)]
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut shader.dxil.clone()).unwrap();
        }
        let _ = shader;
    }

    /// Custom interpolation reads I and J (v0, v1 with PERSP_CENTER) as
    /// plain floats: they are channels y and z of `SV_Barycentrics`, the
    /// weights of vertices 1 and 2.
    #[test]
    fn barycentrics_read_as_values_are_sv_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [v_add_f32(2, 0, 1), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        let bary: Vec<_> = inputs.iter().filter(|e| e.kind == 28).collect();
        assert_eq!(bary.len(), 1, "{inputs:?}");
        assert_eq!((bary[0].index, bary[0].col, bary[0].cols), (0, 0, 3));
        // Not packed: the validator wants packing location -1.
        assert_eq!(bary[0].row, None);
        // Only y and z (I and J) are read.
        let isg1 = Container::parse(&shader.dxil).unwrap();
        let isg1 = isg1.part(FourCC::ISG1).unwrap().data;
        let rw: Vec<u8> = (0..2).map(|k| isg1[8 + 32 * k + 25]).collect();
        assert!(rw.contains(&0b110), "{rw:?}");
        validate(&shader);
    }

    /// Two kinds of barycentrics read directly: one element per mode.
    #[test]
    fn two_barycentric_modes_get_two_elements() {
        // v0-v1 perspective i, j; v2-v3 linear i, j.
        let [e0, e1] = exp_mrt0(4);
        let words = [v_add_f32(4, 0, 3), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER | LINEAR_CENTER).unwrap();
        let (_, inputs) = parts(&shader);
        let mut indices: Vec<u32> = inputs
            .iter()
            .filter(|e| e.kind == 28)
            .map(|e| e.index)
            .collect();
        indices.sort();
        assert_eq!(indices, [0, 1], "{inputs:?}");
        validate(&shader);
    }

    /// I and J only fed to `v_interp_*` stay host interpolation: no
    /// barycentrics are declared.
    #[test]
    fn interpolation_alone_declares_no_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_eq!(features & 0x2_0000, 0);
        assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
        validate(&shader);
    }

    /// Both: the attribute is interpolated and J is used as a value.
    #[test]
    fn interpolation_and_a_barycentric_value_together() {
        let [e0, e1] = exp_mrt0(3);
        let words = [
            0xC808_0000,        // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,        // v_interp_p2_f32 v2, v1, attr0.x
            v_add_f32(3, 2, 1), // v3 = attribute + j
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        validate(&shader);
    }
    /// The `PSV0` interpolation mode of TEXCOORD0 (row 0, x-w, float).
    fn texcoord_mode(shader: &HostShader) -> u8 {
        let c = Container::parse(&shader.dxil).unwrap();
        let psv = c.part(FourCC::PSV0).unwrap().data;
        let row = [1, 0, 0x44, 0, 3];
        let at = psv.windows(5).position(|w| w == row).unwrap();
        psv[at + 5]
    }

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// An interpolation whose i is computed (`v_mul_f32 v3, 0.5, v0`) reads
    /// the attribute's vertices: TEXCOORD0 becomes constant, read with
    /// `attributeAtVertex`; i and j are `SV_Barycentrics` values.
    #[test]
    fn computed_barycentrics_read_the_vertices() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0x1006_00F0, // v_mul_f32 v3, 0.5, v0
            0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// Weights from constants alone: the vertices, no `SV_Barycentrics`.
    #[test]
    fn constant_weights_need_no_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0x7E06_02F0, // v_mov_b32 v3, 0.5
            0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
            0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// The same attribute interpolated with the loaded i and j and with a
    /// computed j: both from its vertices, the first weighed with
    /// `SV_Barycentrics`.
    #[test]
    fn computed_and_loaded_barycentrics_on_one_attribute() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC810_0100, // v_interp_p1_f32 v4, v0, attr0.y
            0xC811_0101, // v_interp_p2_f32 v4, v1, attr0.y
            0x1006_02F0, // v_mul_f32 v3, 0.5, v1
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
            v_add_f32(2, 2, 4),
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (_, inputs) = parts(&shader);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// P10 of an attribute the program interpolates is P1 - P0.
    #[test]
    fn p10_of_an_interpolated_attribute() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0xC812_0000, // v_interp_mov_f32 v4, p10, attr0.x
            v_add_f32(2, 2, 4),
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
        // Alone, what the hardware holds is not known.
        let [e0, e1] = exp_mrt0(4);
        let words = [0xC812_0000, e0, e1, S_ENDPGM];
        assert!(pixel(&words, PERSP_CENTER).is_err());
    }

    #[test]
    fn a_conditional_pixel_export_is_guarded() {
        // Only lanes for which v0 is zero export a colour. This is not the
        // final `done` export, so inactive lanes must keep their old output.
        let words: [u32; 6] = [
            0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
            0xBE80_3C6A, // s_and_saveexec_b32 s0, vcc_lo
            0xF800_000F,
            0x0302_0102, // exp mrt0, v2, v1, v2, v3
            0x887E_007E, // s_or_b32 exec_lo, exec_lo, s0
            0xBF81_0000, // s_endpgm
        ];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![],
                input_addr: 1 << 8,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let position = SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 4,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 1,
            half: false,
        };
        let target = SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        };
        let isg1 = super::container::signature(&[position], true);
        let osg1 = super::container::signature(&[target], false);
        let bytes =
            crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let emitted = super::emit(&c, &native, &p, &st, &io, &res).unwrap();

        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut emitted.dxil.clone()).unwrap();
        }
        #[cfg(not(windows))]
        drop(emitted);
    }

    /// A low-half f16 interpolation preserves the destination's old high
    /// half. A following low-half mixed-precision FMA preserves that high
    /// half again; the first interpolation's unknown low half must not make
    /// the preserved high half look unknown too.
    #[test]
    fn an_f16_interpolation_preserves_its_old_high_half() {
        let words = [
            0x7E10_0280u32, // v_mov_b32 v8, 0
            0x7E20_0280,    // v_mov_b32 v16, 0
            0xD742_000B,
            0x0002_0102, // v_interp_p1ll_f16 v11, v0, attr2.x high
            0xD75A_000F,
            0x042E_0302, // v_interp_p2_f16 v15, v1, attr2.x, v11 high
            0xCC21_000B,
            0x0A02_2108, // v_fma_mixlo_f16 v11, sel_lo(v8), v16, 0
            0xF800_1801,
            0x0000_000B, // exp mrt0, v11, off, off, off done vm
            0xBF81_0000, // s_endpgm
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 17,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 2,
                    mask: 3,
                    attr: 2,
                    flat: false,
                    fp16: true,
                    pack2: true,
                    row2: 2,
                }],
                input_addr: 2,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let input = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(2),
            col: 0,
            cols: 2,
            usage: 0,
            half: false,
        };
        let output = SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let isg1 = super::container::signature(&[input], true);
        let osg1 = super::container::signature(&[output], false);
        let bytes =
            crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let mut host = emit(&c, &native, &p, &st, &io, &res).unwrap();
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut host.dxil).unwrap();
        }
    }
}

//! IR to DXIL, one machine instruction at a time (decision 0003).
//!
//! Every register dword is an `i32` (SCC an `i1`); float operations bitcast
//! their operands. Lane masks exist only as the per-lane booleans of
//! structurisation; branches and guards are built from them. A value the
//! emitter cannot express is `Missing` with a reason, and only fails the
//! program when something with an effect (an export, a branch) needs it.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::bitcode::{Attr, Bin, Cast, Rmw, Term, V, pred};
use super::container::SigElement;
use super::shader::{Shader, flags};
use super::{TranslateOptions, UserData, Wave64Mode};
use crate::ShaderError;
use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::io::{Barycentric, Export, Input, Interp, Io, Target};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::native::{HwStage, NativeProgram, ResourceType, RootType, StageIo, Usage, UsedResource};
use crate::resource::{Address, CODE, Moved, Resources, Role, Site, Source, spilled_sgprs};
use crate::structure::{Atom, Bool, LaneExit, Structured};

type Why = &'static str;

/// Updates which have no native DXIL atomic instruction.
#[derive(Clone, Copy)]
enum AtomicUpdate {
    Inc,
    Dec,
    ReverseSub,
    MaskOr,
    Wrap,
    ClampSub,
}

/// An unformatted byte or halfword load. LLVM's DSInstructions.td and
/// BUFInstructions.td model the D16 destinations as tied to their old value.
fn subword_load(rest: &str) -> Option<(u32, bool, Option<u32>)> {
    let (rest, half) = if let Some(n) = rest.strip_suffix("_d16_hi") {
        (n, Some(16))
    } else if let Some(n) = rest.strip_suffix("_d16") {
        (n, Some(0))
    } else {
        (rest, None)
    };
    let (width, signed) = match rest {
        "i8" | "sbyte" => (8, true),
        "u8" | "ubyte" => (8, false),
        "i16" | "sshort" => (16, true),
        "u16" | "ushort" | "short" => (16, false),
        _ => return None,
    };
    Some((width, signed, half))
}

fn ds_integer_atomic(name: &str) -> Option<(&str, bool, u8)> {
    let (op, ty) = name.strip_prefix("ds_")?.rsplit_once('_')?;
    let op = op.strip_suffix("_rtn").unwrap_or(op);
    let wide = match ty {
        "b32" | "u32" | "i32" => false,
        "b64" | "u64" | "i64" => true,
        _ => return None,
    };
    match op {
        "add" | "sub" | "and" | "or" | "xor" | "min" | "max" | "wrxchg" | "wrxchg2"
        | "wrxchg2st64" | "rsub" | "inc" | "dec" | "mskor" | "wrap" | "cmpst" => {
            Some((op, wide, ty.as_bytes()[0]))
        }
        _ => None,
    }
}

/// How an interpolation evaluates an input channel.
#[derive(Debug, Clone, Copy)]
enum Eval {
    /// Loaded with this DXIL interpolation mode (1: constant, P0).
    Mode(u8),
    /// From the vertex values of a constant element: P0 when `p0`, plus
    /// i (P1 - P0), plus j (P2 - P0), weights as floats.
    Weights {
        p0: bool,
        i: Option<V>,
        j: Option<V>,
    },
}

#[derive(Debug, Clone, Copy)]
enum Val {
    I32(V),
    Bit(V),
    Missing(Why),
}

pub(crate) struct Lowered {
    pub dxil: Vec<u8>,
    /// Bit n: immediate sampler n is used for comparisons.
    pub cmp_samplers: u64,
    pub root_cbvs: Vec<u8>,
    pub root_srvs: Vec<u8>,
    pub stride_table: bool,
    pub root_uavs: Vec<u8>,
    pub user_data: Vec<UserData>,
    pub samplers: Vec<[u32; 4]>,
    /// Heap handles made with a non-uniform index.
    #[cfg(test)]
    pub nonuniform_handles: u32,
    /// Whether the program uses wave operations.
    #[cfg(test)]
    pub wave_ops: bool,
    /// Back edges that count against the spin-wait limit.
    #[cfg(test)]
    pub spin_loops: usize,
    /// The names of the emitted functions (which dx.ops ran); tests only.
    #[cfg(test)]
    pub op_names: Vec<String>,
}

pub(crate) struct Parts<'a> {
    pub native: &'a NativeProgram,
    pub p: &'a Program,
    pub st: &'a Structured,
    pub io: &'a Io,
    pub res: &'a Resources,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    pub prims: Vec<SigElement>,
    /// Mesh: what the group id arrives multiplied by.
    pub mesh_stride: u32,
    pub options: TranslateOptions,
}

enum Stop {
    /// A phi whose incoming value has no DXIL form was read: redo with it missing.
    Retry(Value, Why),
    Fail(ShaderError),
}

const UNTRACKED_SELECT: Why = "a select on an untracked mask";

pub(crate) fn lower(parts: &Parts) -> Result<Lowered, ShaderError> {
    lower_with_limit(parts, super::loop_limit())
}

pub(crate) fn lower_with_limit(parts: &Parts, loop_limit: u32) -> Result<Lowered, ShaderError> {
    match lower_program(parts, false, loop_limit) {
        Err(ShaderError::Unsupported {
            what: UNTRACKED_SELECT,
            ..
        }) if parts.native.stage == HwStage::Cs => lower_program(parts, true, loop_limit),
        result => result,
    }
}

fn lower_program(
    parts: &Parts,
    numeric_selects: bool,
    loop_limit: u32,
) -> Result<Lowered, ShaderError> {
    let mut bad = HashMap::new();
    loop {
        match Lower::new(parts, &bad, numeric_selects, loop_limit).run() {
            Ok(l) => return Ok(l),
            Err(Stop::Retry(v, why)) => {
                if bad.insert(v, why).is_some() {
                    return Err(ShaderError::Unsupported {
                        offset: 0,
                        what: "a phi that does not settle",
                    });
                }
            }
            Err(Stop::Fail(e)) => return Err(e),
        }
    }
}

/// Private memory per lane when the code does not show how far its
/// addresses reach (an index loaded or carried around a loop).
// shortcut: the console's size is set at pipeline creation (the program
// metadata read so far does not hold it); accesses past this read 0 and
// drop writes, as accesses past the console's size do.
pub(super) const SCRATCH_CAP: u32 = 4 << 10;
/// The most private memory per lane, the D3D11 limit of indexable
/// temporaries (4096 four-dword registers).
// shortcut: accesses past it read 0 and drop writes.
const SCRATCH_MAX: u32 = 64 << 10;

/// Bytes a `scratch_*` opcode moves (per lane), from the name after
/// `scratch_load_` or `scratch_store_`.
fn scratch_width(rest: &str) -> Option<u32> {
    Some(match rest.split('_').next()? {
        "dword" => 4,
        "dwordx2" => 8,
        "dwordx3" => 12,
        "dwordx4" => 16,
        "byte" | "ubyte" | "sbyte" => 1,
        "short" | "ushort" | "sshort" => 2,
        _ => return None,
    })
}

/// Private memory per lane the program needs: past the highest byte its
/// scratch accesses can reach where the code computing their addresses
/// bounds them, else [`SCRATCH_CAP`]; at most [`SCRATCH_MAX`].
fn scratch_bytes(p: &Program) -> u32 {
    let mut defs = HashMap::new();
    for i in p.blocks.iter().flat_map(|b| &b.insts) {
        for &(_, v) in &i.writes {
            defs.insert(v, i);
        }
    }
    let mut memo = HashMap::new();
    let mut end = 0u64;
    for i in p.blocks.iter().flat_map(|b| &b.insts) {
        let name = i.name();
        let Fields::Flat {
            addr,
            saddr,
            offset,
            ..
        } = &i.fields
        else {
            continue;
        };
        let rest = name
            .strip_prefix("scratch_load_")
            .or_else(|| name.strip_prefix("scratch_store_"));
        let Some(width) = rest.and_then(scratch_width) else {
            continue;
        };
        let base = match (addr, saddr) {
            (Some(o), _) | (None, Some(o)) => operand_bound(&defs, &mut memo, i, o, 0),
            (None, None) => Some(0),
        };
        // A negative offset only lowers the address (or wraps it out of
        // range, checked when the program runs).
        let off = u64::from((*offset).max(0).unsigned_abs());
        end = end.max(base.map_or(u64::from(SCRATCH_CAP), |b| b + off + u64::from(width)));
    }
    end.min(u64::from(SCRATCH_MAX)) as u32
}

/// An upper bound of operand `o` of `inst` as an unsigned number.
fn operand_bound(
    defs: &HashMap<Value, &Inst>,
    memo: &mut HashMap<Value, Option<u64>>,
    inst: &Inst,
    o: &Operand,
    depth: u8,
) -> Option<u64> {
    match *o {
        Operand::Scalar { code: NULL, .. } => Some(0),
        Operand::Scalar { code, .. } => value_bound(defs, memo, inst.read(Reg::S(code))?, depth),
        Operand::Vector { reg, .. } => value_bound(defs, memo, inst.read(Reg::V(reg))?, depth),
        Operand::Const(c) => crate::resource::inline(c).map(u64::from),
        Operand::Literal(x) => Some(u64::from(x)),
        Operand::Attr { .. } => None,
    }
}

/// An upper bound of 32-bit value `v` as an unsigned number, where the
/// instructions computing it show one (constants combined by masks, shifts,
/// sums, products, selects); `None` where it depends on anything else
/// (inputs, memory, a phi) or may wrap.
fn value_bound(
    defs: &HashMap<Value, &Inst>,
    memo: &mut HashMap<Value, Option<u64>>,
    v: Value,
    depth: u8,
) -> Option<u64> {
    if let Some(&b) = memo.get(&v) {
        return b;
    }
    let inst = *defs.get(&v)?;
    if depth > 32 {
        return None;
    }
    let (dst, src) = match &inst.fields {
        Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            src,
            imm: None,
        } => (Reg::S(*code), src),
        Fields::Valu(x)
            if x.sdwa.is_none()
                && x.dpp.is_none()
                && x.neg | x.abs | x.opsel == 0
                && !x.clamp
                && x.omod == 0 =>
        {
            match x.dst {
                Some(Operand::Vector { reg, .. }) => (Reg::V(reg), &x.src),
                _ => return None,
            }
        }
        _ => return None,
    };
    if inst.write(dst) != Some(v) {
        return None;
    }
    let mut arg =
        |k: usize| -> Option<u64> { operand_bound(defs, memo, inst, src.get(k)?, depth + 1) };
    // The smallest all-ones number at least `x` (bounds `|` and `^`).
    let ones = |x: u64| {
        if x == 0 {
            0
        } else {
            u64::MAX >> x.leading_zeros()
        }
    };
    let u24 = |x: Option<u64>| x.map(|x| x.min(0xFF_FFFF));
    let shl = |x: Option<u64>, n: Option<u64>| -> Option<u64> {
        x?.checked_shl(u32::try_from(n.filter(|&n| n < 32)?).ok()?)
    };
    let b = match inst.name() {
        "v_mov_b32" | "s_mov_b32" => arg(0),
        "v_and_b32" | "s_and_b32" => match (arg(0), arg(1)) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, y) => x.or(y),
        },
        "v_or_b32" | "s_or_b32" | "v_xor_b32" | "s_xor_b32" => Some(ones(arg(0)?.max(arg(1)?))),
        "v_lshlrev_b32" => {
            let n = arg(0);
            shl(arg(1), n)
        }
        "s_lshl_b32" => {
            let x = arg(0);
            shl(x, arg(1))
        }
        "v_lshrrev_b32" => arg(1),
        "s_lshr_b32" => arg(0),
        "v_add_nc_u32" | "v_add_u32" | "v_add_co_u32" | "s_add_u32" | "s_add_i32" => {
            Some(arg(0)? + arg(1)?)
        }
        "v_add3_u32" => Some(arg(0)? + arg(1)? + arg(2)?),
        "v_mul_u32_u24" => Some(u24(arg(0))? * u24(arg(1))?),
        "v_mul_lo_u32" | "s_mul_i32" => arg(0)?.checked_mul(arg(1)?),
        "v_mad_u32_u24" => Some(u24(arg(0))? * u24(arg(1))? + arg(2)?),
        "v_lshl_add_u32" | "v_lshl_or_b32" => {
            let x = arg(0);
            let n = arg(1);
            Some(shl(x, n)? + arg(2)?)
        }
        "v_add_lshl_u32" => {
            let x = Some(arg(0)? + arg(1)?);
            shl(x, arg(2))
        }
        "v_and_or_b32" => {
            let x = match (arg(0), arg(1)) {
                (Some(x), Some(y)) => x.min(y),
                (x, y) => x.or(y)?,
            };
            Some(x + arg(2)?)
        }
        "v_bfe_u32" => match arg(2) {
            Some(w) if w < 32 => Some((1 << w) - 1),
            _ => arg(0),
        },
        "v_cndmask_b32" | "v_max_u32" | "s_max_u32" | "s_cselect_b32" => Some(arg(0)?.max(arg(1)?)),
        "v_min_u32" | "s_min_u32" => match (arg(0), arg(1)) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, y) => x.or(y),
        },
        _ => None,
    }
    .filter(|&b| b <= u64::from(u32::MAX));
    memo.insert(v, b);
    b
}

const TWO_PI: f32 = std::f32::consts::TAU;
/// `s_sendmsg` id of a mesh or NGG stage's output allocation request.
const GS_ALLOC_REQ: u16 = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numeric {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

/// A GFX10 buffer format (the unified format of typed buffer loads): the
/// channel widths from the lowest bits up, and how the channels convert.
/// Formats are numbered in families (`8`, `16`, `8_8`, ...: AMD names
/// list channels from the highest bits), each with its numeric formats in
/// the order unorm, snorm, uscaled, sscaled, uint, sint, float.
fn buffer_format(format: u8) -> Option<(&'static [u32], Numeric)> {
    use Numeric::*;
    const NORM: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
    const ALL: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
    const WIDE: &[Numeric] = &[Uint, Sint, Float];
    // `None`: a family not decoded (packed floats, 10_10_10_2).
    const FAMILIES: [(Option<&[u32]>, &[Numeric]); 14] = [
        (Some(&[8]), NORM),
        (Some(&[16]), ALL),
        (Some(&[8, 8]), NORM),
        (Some(&[32]), WIDE),
        (Some(&[16, 16]), ALL),
        (None, ALL),                    // 10_11_11
        (None, ALL),                    // 11_11_10
        (None, NORM),                   // 10_10_10_2
        (Some(&[10, 10, 10, 2]), NORM), // 2_10_10_10
        (Some(&[8, 8, 8, 8]), NORM),
        (Some(&[32, 32]), WIDE),
        (Some(&[16, 16, 16, 16]), ALL),
        (Some(&[32, 32, 32]), WIDE),
        (Some(&[32, 32, 32, 32]), WIDE),
    ];
    let mut n = usize::from(format.checked_sub(1)?);
    for (bits, numerics) in FAMILIES {
        if n < numerics.len() {
            return Some((bits?, numerics[n]));
        }
        n -= numerics.len();
    }
    None
}

struct Lower<'a> {
    s: Shader,
    x: &'a Parts<'a>,
    bad: &'a HashMap<Value, Why>,
    lane_fail: HashMap<Value, Why>,
    vals: HashMap<Value, Val>,
    floats: HashMap<Value, V>,
    /// DXIL block per IR block (unreachable: none).
    block: Vec<Option<usize>>,
    cur: usize,
    bools: HashMap<(usize, Bool), V>,
    lane_bits: HashMap<Value, V>,
    quad_bits: HashMap<Value, V>,
    quad_visiting: HashSet<Value>,
    /// IR phi location by value.
    phi_at: HashMap<Value, (usize, usize)>,
    mask_phis: HashMap<Value, V>,
    pending_masks: Vec<(V, usize, usize)>,
    data_phis: Vec<(V, usize, usize)>,
    read_phis: HashSet<Value>,
    params: HashMap<Value, (usize, u32)>,
    user_data: Option<V>,
    user_reads: BTreeMap<u32, UserData>,
    samplers: Vec<[u32; 4]>,
    descriptors: HashMap<(Site, bool), &'a Source>,
    scalar: HashMap<Site, Address>,
    exports: HashMap<Site, &'a Export>,
    interps: HashMap<Site, &'a Interp>,
    handles: HashMap<(usize, Site, u8, bool, u32, u32), V>,
    /// Blocks splitting the edge from one IR block to another.
    edges: HashMap<(usize, usize), usize>,
    /// Per IR block: a `done` export is behind every lane entering it.
    exported: Vec<bool>,
    /// Reverse-postorder index per IR block.
    rpo: Vec<usize>,
    /// DXIL block each IR block ends in (guarded stores split blocks).
    tail: Vec<Option<usize>>,
    /// The program writes UAVs: barriers fence them too.
    writes_uavs: bool,
    wide_lds: bool,
    /// src0 of the current instruction as DPP fetched it.
    src0: Option<V>,
    /// Compute code that reads other lanes: thread ids follow the host
    /// lanes, so lane n is console lane n.
    console_lanes: bool,
    /// Retry a compute shader with numeric selects only when the normal
    /// translation needs one. Unused selects need no fixed wave size.
    numeric_selects: bool,
    /// Wave64 compute lowering mode (`Native` when not a Wave64 compute shader).
    wave64: Wave64Mode,
    /// Compute thread count per group (0 outside compute).
    total_threads: u32,
    /// LDS dwords declared by the title itself (excluding emulation scratch).
    title_lds_dwords: u32,
    /// Virtual 64-lane waves per thread group (`total_threads.div_ceil(64)`).
    virtual_waves: u32,
    /// Base dword index of the ballot / firstlane LDS scratch region.
    ballot_base: u32,
    /// Scratch dwords per virtual wave in `ballot_base` (`2` or `4`).
    ballot_stride: u32,
    /// Base dword index of the per-lane LDS exchange region (`64 * virtual_waves`).
    exchange_base: u32,
    /// Emulation scratch pushed total LDS past D3D12's 32 KiB (8192 dword) limit.
    lds_overflow: bool,
    /// Barrier-less `<= 64`-thread Wave64 compute shader sharing LDS across host waves.
    need_lds_barriers: bool,
    /// Pre-resolved constant-lane `v_readlane_b32` SGPR spills by site.
    spills: HashMap<(usize, usize), Moved>,
    /// Per IR block: whether all threads in the thread group arrive together.
    uniform: Vec<bool>,
    /// Per IR value: conservatively true only for group-uniform scalars.
    val_uniform: Vec<bool>,
    /// Entry-block `(flat_tid, console_lane, vwave)` cached when emulating Wave64.
    wave_ids: Option<(V, V, V)>,
    /// Active `begin_guarded` nesting depth (never emit barriers while > 0).
    guard_depth: u32,
    /// Masks read as data in a block other than their defining block, keyed by `def_block`.
    cross_block_masks: HashMap<Option<usize>, Vec<Value>>,
    /// Errors deferred from pre-emitted cross-block masks.
    ballot_fail: HashMap<Value, Why>,
    /// Mesh: the thread index in the group (made in the entry block).
    flat: Option<V>,
    /// Mesh: IR blocks whose `s_barrier` is left out.
    no_barrier: HashSet<usize>,
    /// Single-wave LDS lockstep synchronised by barriers instead of `WaveSize(64)`:
    /// instructions `(block, index)` that need a group-shared barrier before them,
    /// and uniform blocks that need one before branching into a divergent region.
    lds_barrier_before: HashSet<(usize, usize)>,
    lds_barrier_end: HashSet<usize>,
    /// Vertex and mesh outputs holding 16-bit values: row, whether two per
    /// dword, and the row packed after it.
    half_outputs: HashMap<u8, (bool, Option<u8>)>,
    /// Pixel inputs holding 16-bit values: row, whether two per dword, and
    /// the row packed after it.
    half_inputs: HashMap<u8, (bool, Option<u8>)>,
    /// How lane masks become data.
    mask_data: MaskData,
    /// The instruction being lowered may read lane masks as data.
    mask_ok: bool,
    /// Ballots of masks read as data, by low dword: (low, high).
    ballots: HashMap<Value, (V, V)>,
    /// Masks whose ballot is being built: a mask whose lane bit needs its
    /// own value as a number (an opaque mask keyed by a dword it shares
    /// with another) would otherwise recurse until the stack overflows.
    balloting: HashSet<Value>,
    /// Block writing each value.
    def_block: HashMap<Value, usize>,
    /// IR block being lowered.
    cur_ir: usize,
    /// Whole-wave sections by block and start index.
    sections: HashMap<usize, HashMap<usize, super::wave::Section>>,
    /// Values written in a section that code outside it reads.
    section_out: HashSet<Value>,
    /// The program's constant table (the code from the first byte read
    /// through `s_getpc_b64` on): global, first byte, dwords.
    code_table: Option<(V, u32, u32)>,
    /// Private (scratch) memory: the lane's array and its dwords.
    scratch: Option<(V, u32)>,
    /// Per-invocation back-edge step budget (see [`LOOP_LIMIT`]).
    loop_limit: u32,
    /// Per-invocation back-edge step budget for spin-wait loops (see [`super::spin_limit`]).
    spin_limit: u32,
    /// Loop headers whose natural loop executes an active `s_barrier`:
    /// these share `steps[1]` (group-uniform) instead of `steps[0]`.
    barrier_loops: HashSet<usize>,
    /// Function-wide step counters: `[lane_loops, barrier_loops, spin_loops]`.
    steps: [Option<StepTrack>; 3],
    /// The block a loop stopped by [`LOOP_LIMIT`] leaves the program from.
    abort: Option<usize>,
    /// `v_readlane_b32` sites reading an SGPR spilled via `v_writelane_b32`,
    /// resolved directly in SSA without wave operations.
    resolved_reads: HashMap<Site, Spill>,
    /// `v_writelane_b32` sites whose VGPR result is consumed only by resolved
    /// `v_readlane_b32` sites (or other pure spill writes/phis).
    pure_spill_writes: HashSet<Site>,
    /// Headers of loops that only wait for memory to change: they stop
    /// at [`super::spin_limit`] instead.
    spin: HashSet<usize>,
    /// Back edges given the spin-wait limit.
    #[cfg(test)]
    spun: usize,
}

/// A scalar value spilled into a VGPR lane by `v_writelane_b32` and read
/// back by `v_readlane_b32` at the same constant lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Spill {
    Value(Value),
    Const(u32),
}

/// SSA tracking for a function-wide loop step counter across IR blocks.
struct StepTrack {
    phis: Vec<Option<V>>,
    incoming: Vec<Vec<(V, usize)>>,
    cur_val: Option<V>,
}

/// Total back-edge steps per invocation after which loops are left, unless
/// [`super::loop_limit`] says otherwise. The console runs a loop as long as
/// its data says; data the host has not mirrored yet (a null view reads
/// zeros) can make that four billion iterations, which hangs the host GPU
/// until the driver resets it. Sharing one step budget across all loops in
/// the function prevents nested loops from multiplying the cap. A lane that
/// reaches the limit leaves the program as killed (a pixel discards).
// shortcut: a genuine shader taking more total loop steps than this is cut short too.
pub(super) const LOOP_LIMIT: u32 = 1 << 20;

/// Lane masks read as numbers (`s_bcnt1`, `s_ff1`, `v_mbcnt`, scalar
/// compares, bit tests) are ballots of the lanes' booleans, taken where the
/// mask is written: structurisation's `active` masks hold only lanes that
/// reach that block, so the ballot misses none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskData {
    /// Compute: host lane n is console lane n (the program declares the
    /// console's wave size), so ballots are the console's masks bit for bit
    /// and any use is exact.
    Console,
    /// Pixel and vertex programs: ballots over the host's wave, whatever its
    /// size. Only uses that mean the same on any wave are allowed: counts,
    /// the first lane, the count below a lane, equality (the HLSL wave
    /// operations the console compiler expands). Lane numbers stay host
    /// lanes throughout (`v_mbcnt`, `v_readlane` at a computed lane).
    Host,
    /// Mesh programs: their lanes and waves (LDS per console wave) are the
    /// console's but no host wave size can be declared.
    None,
}

/// What a value holding a lane mask read as a number stops with.
const LANE_MASK: Why = "a lane mask read as data";

/// Whether `i` is an LDS memory instruction (not a cross-lane swizzle or permute).
fn is_lds_mem(i: &Inst) -> bool {
    let n = i.name();
    matches!(i.fields, Fields::Ds { .. }) && !n.starts_with("ds_swizzle") && !n.contains("permute")
}

/// Whether `i` writes (or atomically modifies) LDS memory.
fn is_lds_write(i: &Inst) -> bool {
    is_lds_mem(i) && !i.name().starts_with("ds_read")
}

/// Whether `i` reads (or atomically reads-modifies) LDS memory.
fn is_lds_read(i: &Inst) -> bool {
    is_lds_mem(i) && !i.name().starts_with("ds_write")
}

/// Planned threadgroup-uniform LDS barriers for a single-wave Wave64 shader.
struct LdsBarriers {
    before: HashSet<(usize, usize)>,
    at_end: HashSet<usize>,
}

/// Plans thread-group LDS barriers for a single-wave Wave64 compute shader
/// (`threads <= 64`) so it does not rely on 64-lane lockstep (`WaveSize(64)`).
///
/// Returns `Some(barriers)` when every cross-lane LDS read/write hazard
/// passes through a threadgroup-uniform block where `dx.op.barrier` is safe;
/// returns `None` when a divergent region itself has both an LDS write and an
/// LDS read without reconverging.
fn plan_lds_barriers(p: &Program, st: &Structured) -> Option<LdsBarriers> {
    let n = p.blocks.len();
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut reachable = vec![false; n];
    for &b in &order {
        reachable[b] = true;
    }
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for &b in &order {
        match st.exits[b] {
            LaneExit::Jump(t) if reachable[t] => succs[b].push(t),
            LaneExit::Branch {
                taken, not_taken, ..
            } => {
                if reachable[taken] {
                    succs[b].push(taken);
                }
                if not_taken != taken && reachable[not_taken] {
                    succs[b].push(not_taken);
                }
            }
            LaneExit::End | LaneExit::Indirect => succs[b].push(n),
            _ => {}
        }
    }
    // Post-dominators with virtual exit node `n`.
    let words = (n + 1).div_ceil(64);
    let mut pdom = vec![vec![u64::MAX; words]; n + 1];
    pdom[n] = vec![0; words];
    pdom[n][n / 64] = 1u64 << (n % 64);
    loop {
        let mut changed = false;
        for &b in order.iter().rev() {
            if succs[b].is_empty() {
                continue;
            }
            let mut next = pdom[succs[b][0]].clone();
            for &s in &succs[b][1..] {
                for w in 0..words {
                    next[w] &= pdom[s][w];
                }
            }
            next[b / 64] |= 1u64 << (b % 64);
            if next != pdom[b] {
                pdom[b] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let postdom = |a: usize, b: usize| -> bool { (pdom[b][a / 64] >> (a % 64)) & 1 != 0 };

    // Values that can differ between the two 32-lane halves of a Wave64 wave:
    // vector inputs, vector loads/atomics/cross-lane ops, guarded vector writes,
    // and any value (including a scalar from `v_readfirstlane_b32`) derived from them.
    let mut varying: HashSet<Value> = p
        .inputs
        .iter()
        .filter(|&&(r, _)| matches!(r, Reg::V(_)))
        .map(|&(_, v)| v)
        .collect();
    for &b in &order {
        for (i, inst) in p.blocks[b].insts.iter().enumerate() {
            let produces_varying = match &inst.fields {
                Fields::Buffer { .. }
                | Fields::Image { .. }
                | Fields::Ds { .. }
                | Fields::Flat { .. } => true,
                Fields::Valu(v) => {
                    st.guards[b][i].is_some()
                        || v.dpp.is_some()
                        || inst.name().starts_with("v_mbcnt")
                        || matches!(inst.name(), "v_readlane_b32" | "v_writelane_b32")
                }
                _ => false,
            };
            if produces_varying {
                for &(_, w) in &inst.writes {
                    varying.insert(w);
                }
            }
        }
    }
    let inst_reads_varying = |inst: &Inst, varying: &HashSet<Value>| -> bool {
        if let Fields::Valu(v) = &inst.fields {
            // Check explicit source operands so an unguarded `v_mov_b32 v, s`
            // does not inherit `varying` from its implicit EXEC / old-dst reads.
            v.src.iter().any(|op| match *op {
                Operand::Vector { reg, .. } => inst
                    .read(Reg::V(reg))
                    .is_some_and(|val| varying.contains(&val)),
                Operand::Scalar { code, .. } if code != NULL => inst
                    .read(Reg::S(code))
                    .is_some_and(|val| varying.contains(&val)),
                _ => false,
            })
        } else {
            inst.reads.iter().any(|&(_, val)| varying.contains(&val))
        }
    };
    let mut non_uniform = vec![false; n];
    loop {
        let mut changed = false;
        // Propagate `varying` across phis and instructions.
        loop {
            let before_len = varying.len();
            for &b in &order {
                for phi in &p.blocks[b].phis {
                    let from_div = p.blocks[b]
                        .preds
                        .iter()
                        .any(|&pred| reachable[pred] && non_uniform[pred]);
                    if matches!(phi.reg, Reg::V(_))
                        || from_div
                        || phi.args.iter().any(|a| varying.contains(a))
                    {
                        varying.insert(phi.value);
                    }
                }
                for inst in &p.blocks[b].insts {
                    if inst_reads_varying(inst, &varying) {
                        for &(_, w) in &inst.writes {
                            varying.insert(w);
                        }
                    }
                }
            }
            if varying.len() == before_len {
                break;
            }
            changed = true;
        }
        // Because `threads <= 64` fits in one console wave, every branch whose
        // condition depends only on threadgroup-uniform `Atom::Uniform`s takes
        // the same edge in every thread of the group. Any block reachable from a
        // divergent branch before strictly post-dominating it is non-uniform.
        for &d in &order {
            let LaneExit::Branch { when, .. } = st.exits[d] else {
                continue;
            };
            let uniform = !non_uniform[d]
                && st
                    .bools
                    .support(when)
                    .iter()
                    .all(|a| matches!(a, Atom::Uniform(v) if !varying.contains(v)));
            if uniform {
                continue;
            }
            let mut stack = succs[d].clone();
            let mut vis = vec![false; n + 1];
            while let Some(cur) = stack.pop() {
                if cur == n || vis[cur] || (cur != d && postdom(cur, d)) {
                    continue;
                }
                vis[cur] = true;
                if !non_uniform[cur] {
                    non_uniform[cur] = true;
                    changed = true;
                }
                stack.extend_from_slice(&succs[cur]);
            }
        }
        if !changed {
            break;
        }
    }

    // Refuse if any non-uniform region has a RAW or WAR hazard without passing
    // through a uniform block.
    for &b in &order {
        if !non_uniform[b] {
            continue;
        }
        let (mut saw_w, mut saw_r) = (false, false);
        for inst in &p.blocks[b].insts {
            let (w, r) = (is_lds_write(inst), is_lds_read(inst));
            if (saw_w && r) || (saw_r && w) {
                return None;
            }
            saw_w |= w;
            saw_r |= r;
        }
        if saw_w || saw_r {
            let mut stack: Vec<usize> = succs[b]
                .iter()
                .copied()
                .filter(|&s| s < n && non_uniform[s])
                .collect();
            let mut vis = vec![false; n];
            while let Some(cur) = stack.pop() {
                if vis[cur] {
                    continue;
                }
                vis[cur] = true;
                let cw = p.blocks[cur].insts.iter().any(is_lds_write);
                let cr = p.blocks[cur].insts.iter().any(is_lds_read);
                if (saw_w && cr) || (saw_r && cw) {
                    return None;
                }
                for &s in &succs[cur] {
                    if s < n && non_uniform[s] {
                        stack.push(s);
                    }
                }
            }
        }
    }

    // Forward dataflow: bit 0 = unsynchronised LDS write, bit 1 = unsynchronised LDS read.
    const W: u8 = 1;
    const R: u8 = 2;
    let mut in_state = vec![0u8; n];
    let mut out_state = vec![0u8; n];
    let mut before = HashSet::new();
    let mut at_end = HashSet::new();
    loop {
        let mut changed = false;
        before.clear();
        at_end.clear();
        for &b in &order {
            let mut state = 0u8;
            for &pred in &p.blocks[b].preds {
                if reachable[pred] {
                    state |= out_state[pred];
                }
            }
            in_state[b] = state;
            if non_uniform[b] {
                for inst in &p.blocks[b].insts {
                    if is_lds_write(inst) {
                        state |= W;
                    }
                    if is_lds_read(inst) {
                        state |= R;
                    }
                }
            } else {
                for (i, inst) in p.blocks[b].insts.iter().enumerate() {
                    if inst.name() == "s_barrier" {
                        state = 0;
                        continue;
                    }
                    let (w, r) = (is_lds_write(inst), is_lds_read(inst));
                    if (w && state & R != 0) || (r && state & W != 0) {
                        before.insert((b, i));
                        state = 0;
                    }
                    if w {
                        state |= W;
                    }
                    if r {
                        state |= R;
                    }
                }
                if state != 0 && succs[b].iter().any(|&s| s < n && non_uniform[s]) {
                    at_end.insert(b);
                    state = 0;
                }
            }
            if state != out_state[b] {
                out_state[b] = state;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Some(LdsBarriers { before, at_end })
}

/// Instructions that read a lane mask (not EXEC implicitly, not a lane's
/// bit) as a number.
fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}

/// A select that needs the actual scalar mask words and console lane index.
fn numeric_select(st: &Structured, inst: &Inst) -> bool {
    inst.name() == "v_cndmask_b32"
        && matches!(&inst.fields, Fields::Valu(v)
            if matches!(v.src.get(2), Some(Operand::Scalar { code, .. }) if *code != NULL))
        && inst
            .writes
            .first()
            .is_some_and(|&(_, v)| !st.select_masks.contains_key(&v))
}

/// Where half `h` (dword h / 2, the high half when odd) of a parameter
/// packing 16-bit rows `a` and `b` two per dword comes from: a row and
/// channel of `elements`, none for padding. Channel c of `a` is half c;
/// `b` starts at the dword after `a`'s last channel.
fn packed_half(elements: &[SigElement], a: u8, b: Option<u8>, h: u8) -> Option<(u8, u8)> {
    let extent = |r: u8| {
        elements
            .iter()
            .filter(|e| e.row == Some(u32::from(r)))
            .map(|e| e.col + e.cols)
            .max()
    };
    let na = extent(a)?;
    let a_halves = na.div_ceil(2) * 2;
    if h < a_halves {
        return (h < na).then_some((a, h));
    }
    let b = b?;
    let c = h - a_halves;
    (c < extent(b)?).then_some((b, c))
}

/// Vertex or mesh outputs of 16-bit values (see [`Lower::half_outputs`]):
/// a packed row's partner is the next 16-bit row of the signature that no
/// output entry names, taken in row order.
fn half_outputs(io: &StageIo, outputs: &[SigElement]) -> HashMap<u8, (bool, Option<u8>)> {
    let entries = match io {
        StageIo::Vs { outputs, .. } => outputs,
        StageIo::Ms(ms) => &ms.outputs,
        _ => return HashMap::new(),
    };
    let per_vertex = || entries.iter().filter(|o| o.param & 0x80 == 0);
    let mut free: Vec<u8> = outputs
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| !per_vertex().any(|o| o.row == r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut halves: Vec<_> = per_vertex().filter(|o| o.fp16).collect();
    halves.sort_by_key(|o| o.row);
    halves
        .into_iter()
        .map(|o| {
            let b = o
                .pack2
                .then(|| free.iter().position(|&r| r > o.row))
                .flatten();
            (o.row, (o.pack2, b.map(|i| free.remove(i))))
        })
        .collect()
}

/// Pixel inputs of 16-bit values (see [`Lower::half_inputs`]): a packed row's
/// partner is `row2` when distinct from `row`, else the next unlisted 16-bit
/// row of the signature before the next `fp16` input row, taken in row order.
fn half_inputs(io: &StageIo, inputs: &[SigElement]) -> HashMap<u8, (bool, Option<u8>)> {
    let StageIo::Ps {
        inputs: entries, ..
    } = io
    else {
        return HashMap::new();
    };
    // Keep the first fp16 entry per row (matching the previous `.find()`
    // behaviour) so duplicate rows neither overwrite earlier ones nor consume
    // an extra free partner row.
    let mut seen = HashSet::new();
    let mut halves: Vec<_> = entries
        .iter()
        .filter(|i| i.fp16 && seen.insert(i.row))
        .collect();
    halves.sort_by_key(|i| i.row);
    // In the console's PsInput table, `row2` repeats `row` when no distinct
    // partner row is named; exclude both listed rows and explicitly claimed
    // partners (`row2 != row`) from the fallback pool.
    let mut free: Vec<u8> = inputs
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| {
            !entries.iter().any(|i| i.row == r)
                && !halves
                    .iter()
                    .any(|i| i.pack2 && i.row2 != i.row && i.row2 == r)
        })
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut out = HashMap::with_capacity(halves.len());
    for (idx, i) in halves.iter().enumerate() {
        let next_fp16 = halves.get(idx + 1).map_or(u8::MAX, |n| n.row);
        // Only take a fallback partner before the next fp16 input row so an
        // earlier unpaired input cannot steal a later fp16 input's partner.
        let b = if !i.pack2 {
            None
        } else if i.row2 != i.row {
            Some(i.row2)
        } else {
            free.iter()
                .position(|&r| r > i.row && r < next_fp16)
                .map(|p| free.remove(p))
        };
        out.insert(i.row, (i.pack2, b));
    }
    out
}

/// What a sample or gather opcode adds to the plain form, by its name:
/// comparison (`_c`), explicit derivatives (`_d`, coarse `_cd`, 16-bit
/// `_g16`), a level (`_l`) or level 0 (`_lz`), a bias (`_b`), a level
/// clamp (`_cl`), a texel offset (`_o`).
#[derive(Debug, Clone, Copy, Default)]
struct SampleForm {
    gather: bool,
    c: bool,
    d: bool,
    g16: bool,
    l: bool,
    lz: bool,
    b: bool,
    cl: bool,
    o: bool,
}

fn sample_form(name: &str) -> Option<SampleForm> {
    let (gather, rest) = match name.strip_prefix("image_sample") {
        Some(rest) => (false, rest),
        None => (true, name.strip_prefix("image_gather4")?),
    };
    let mut f = SampleForm {
        gather,
        ..SampleForm::default()
    };
    for token in rest.split('_').skip(1) {
        let flag = match token {
            "c" => &mut f.c,
            "d" | "cd" => &mut f.d,
            "g16" => &mut f.g16,
            "l" => &mut f.l,
            "lz" => &mut f.lz,
            "b" => &mut f.b,
            "cl" => &mut f.cl,
            "o" => &mut f.o,
            _ => return None,
        };
        *flag = true;
    }
    Some(f)
}

fn unsupported(offset: usize, what: Why) -> Stop {
    Stop::Fail(ShaderError::Unsupported { offset, what })
}

/// Constant lane index, masked to the hardware's low bits (`lane & 63` in
/// Wave64, `lane & 31` in Wave32) so aliased lanes such as 67 and 3 compare equal.
fn const_lane(op: Operand, wave64: bool) -> Option<u32> {
    let raw = match op {
        Operand::Const(c) => crate::resource::inline(c)?,
        Operand::Literal(x) => x,
        _ => return None,
    };
    Some(raw & if wave64 { 63 } else { 31 })
}

/// Resolves SGPR spills through VGPR lanes (`v_writelane_b32` ->
/// `v_readlane_b32` at a constant lane) directly in SSA, and identifies the
/// `v_writelane_b32` sites whose VGPR output never escapes to any other use.
fn resolve_spills(p: &Program, st: &Structured) -> (HashMap<Site, Spill>, HashSet<Site>) {
    let mut def_site: HashMap<Value, Site> = HashMap::new();
    let mut def_block: HashMap<Value, usize> = HashMap::new();
    let mut phi_at: HashMap<Value, (usize, usize)> = HashMap::new();
    for &(_, v) in &p.inputs {
        def_block.insert(v, 0);
    }
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, phi) in block.phis.iter().enumerate() {
            phi_at.insert(phi.value, (b, i));
            def_block.insert(phi.value, b);
        }
        for (i, inst) in block.insts.iter().enumerate() {
            for &(_, w) in &inst.writes {
                def_site.insert(w, (b, i));
                def_block.insert(w, b);
            }
        }
    }

    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut rpo = vec![usize::MAX; p.blocks.len()];
    for (n, &b) in order.iter().enumerate() {
        rpo[b] = n;
    }
    let idom = crate::ir::idoms(&p.blocks, &order, &rpo);
    let dominates = |a: usize, mut b: usize| -> bool {
        if rpo[a] == usize::MAX || rpo[b] == usize::MAX {
            return false;
        }
        while b != a {
            let d = idom[b];
            if d == b || d == usize::MAX {
                return false;
            }
            b = d;
        }
        true
    };

    fn walk(
        p: &Program,
        st: &Structured,
        def_site: &HashMap<Value, Site>,
        phi_at: &HashMap<Value, (usize, usize)>,
        v: Value,
        lane: u32,
        seen: &mut Vec<Value>,
    ) -> Result<Option<Spill>, ()> {
        if let Some(&(b, i)) = phi_at.get(&v) {
            if seen.contains(&v) {
                return Ok(None);
            }
            seen.push(v);
            let mut found = None;
            for &arg in &p.blocks[b].phis[i].args {
                if let Some(m) = walk(p, st, def_site, phi_at, arg, lane, seen)? {
                    if found.is_some_and(|f| f != m) {
                        return Err(());
                    }
                    found = Some(m);
                }
            }
            return Ok(found);
        }
        let &(b, i) = def_site.get(&v).ok_or(())?;
        let inst = &p.blocks[b].insts[i];
        let Fields::Valu(w) = &inst.fields else {
            return Err(());
        };
        if inst.name() != "v_writelane_b32"
            || w.abs != 0
            || w.neg != 0
            || w.opsel != 0
            || w.sdwa.is_some()
            || w.dpp.is_some()
        {
            return Err(());
        }
        let (Some(Operand::Vector { reg, .. }), [value, at]) = (w.dst, &w.src[..]) else {
            return Err(());
        };
        if const_lane(*at, p.wave64).ok_or(())? != lane {
            return walk(
                p,
                st,
                def_site,
                phi_at,
                inst.read(Reg::V(reg)).ok_or(())?,
                lane,
                seen,
            );
        }
        let spill = match *value {
            Operand::Scalar { code: NULL, .. } => Spill::Const(0),
            Operand::Scalar { code, .. } => {
                let sv = inst.read(Reg::S(code)).ok_or(())?;
                if st.masks.contains_key(&sv) || st.highs.contains_key(&sv) {
                    return Err(());
                }
                Spill::Value(sv)
            }
            Operand::Literal(x) => Spill::Const(x),
            Operand::Const(c) => Spill::Const(crate::resource::inline(c).ok_or(())?),
            _ => return Err(()),
        };
        Ok(Some(spill))
    }

    let mut resolved_reads = HashMap::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            if inst.name() != "v_readlane_b32" {
                continue;
            }
            let Fields::Valu(v) = &inst.fields else {
                continue;
            };
            if v.abs != 0 || v.neg != 0 || v.opsel != 0 || v.sdwa.is_some() || v.dpp.is_some() {
                continue;
            }
            let [Operand::Vector { reg, .. }, lane_op] = v.src[..] else {
                continue;
            };
            let (Some(lane), Some(vgpr_val)) =
                (const_lane(lane_op, p.wave64), inst.read(Reg::V(reg)))
            else {
                continue;
            };
            if let Ok(Some(spill)) = walk(p, st, &def_site, &phi_at, vgpr_val, lane, &mut Vec::new())
            {
                let ok = match spill {
                    Spill::Const(_) => true,
                    Spill::Value(sv) => def_block.get(&sv).is_some_and(|&db| dominates(db, b)),
                };
                if ok {
                    resolved_reads.insert((b, i), spill);
                }
            }
        }
    }

    // Track which `v_writelane_b32` VGPR values never escape outside spill ops.
    let mut candidates = HashSet::new();
    let mut writelanes: Vec<(Site, Value, Option<Value>)> = Vec::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for phi in &block.phis {
            if matches!(phi.reg, Reg::V(_)) {
                candidates.insert(phi.value);
            }
        }
        for (i, inst) in block.insts.iter().enumerate() {
            if inst.name() == "v_writelane_b32"
                && let Fields::Valu(w) = &inst.fields
                && let Some(Operand::Vector { reg, .. }) = w.dst
                && let Some(written) = inst.write(Reg::V(reg))
            {
                candidates.insert(written);
                writelanes.push(((b, i), written, inst.read(Reg::V(reg))));
            }
        }
    }

    let mut escaped = HashSet::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            if resolved_reads.contains_key(&(b, i)) {
                continue;
            }
            let in_place = if inst.name() == "v_writelane_b32"
                && let Fields::Valu(w) = &inst.fields
                && let Some(Operand::Vector { reg, .. }) = w.dst
                && !w
                    .src
                    .iter()
                    .any(|o| matches!(*o, Operand::Vector { reg: r, .. } if r == reg))
            {
                Some(Reg::V(reg))
            } else {
                None
            };
            for &(r, rv) in &inst.reads {
                if Some(r) == in_place {
                    continue;
                }
                if candidates.contains(&rv) {
                    escaped.insert(rv);
                }
            }
        }
    }

    loop {
        let before = escaped.len();
        for &(_, written, prev) in &writelanes {
            if escaped.contains(&written)
                && let Some(pv) = prev
                && candidates.contains(&pv)
            {
                escaped.insert(pv);
            }
        }
        for block in &p.blocks {
            for phi in &block.phis {
                if escaped.contains(&phi.value) {
                    for &arg in &phi.args {
                        if candidates.contains(&arg) {
                            escaped.insert(arg);
                        }
                    }
                }
            }
        }
        if escaped.len() == before {
            break;
        }
    }

    let pure_spill_writes = writelanes
        .into_iter()
        .filter(|&(_, written, _)| !escaped.contains(&written))
        .map(|(site, _, _)| site)
        .collect();
    (resolved_reads, pure_spill_writes)
}

type Sections = HashMap<usize, HashMap<usize, super::wave::Section>>;

/// Finds whole-wave sections and the values they leave behind.
fn compute_sections(p: &Program) -> (Sections, HashSet<Value>) {
    let mut sections = HashMap::new();
    let mut section_out = HashSet::new();
    let mut inside: HashMap<Value, (usize, usize, usize)> = HashMap::new();
    for b in 0..p.blocks.len() {
        let found = super::wave::sections(p, b);
        for (&start, sec) in &found {
            for i in start + 1..sec.end {
                for &(_, w) in &p.blocks[b].insts[i].writes {
                    inside.insert(w, (b, start, sec.end));
                }
            }
        }
        if !found.is_empty() {
            sections.insert(b, found);
        }
    }
    if !inside.is_empty() {
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                section_out.extend(phi.args.iter().filter(|a| inside.contains_key(a)));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, r) in &inst.reads {
                    if let Some(&(sb, start, end)) = inside.get(&r)
                        && !(sb == b && i > start && i < end)
                    {
                        section_out.insert(r);
                    }
                }
            }
        }
    }
    (sections, section_out)
}

/// Immediate post-dominator per block (or virtual exit `n`) using Cooper-Harvey-Kennedy.
/// Unreachable or never-exiting blocks have `rev_idx == usize::MAX`.
fn compute_postdoms(n: usize, succs: &[Vec<usize>], fwd: &[bool]) -> (Vec<usize>, Vec<usize>) {
    let mut rev_succs = vec![Vec::new(); n + 1];
    for b in 0..n {
        if !fwd[b] {
            continue;
        }
        for &s in &succs[b] {
            rev_succs[s].push(b);
        }
    }
    let mut post = Vec::with_capacity(n + 1);
    let mut seen = vec![false; n + 1];
    let mut stack = vec![(n, 0usize)];
    seen[n] = true;
    while let Some((node, idx)) = stack.last_mut() {
        let u = *node;
        if *idx < rev_succs[u].len() {
            let v = rev_succs[u][*idx];
            *idx += 1;
            if !seen[v] {
                seen[v] = true;
                stack.push((v, 0));
            }
        } else {
            post.push(u);
            stack.pop();
        }
    }
    post.reverse();
    let mut rev_idx = vec![usize::MAX; n + 1];
    for (idx, &b) in post.iter().enumerate() {
        rev_idx[b] = idx;
    }
    let mut ipdom = vec![usize::MAX; n + 1];
    ipdom[n] = n;
    let intersect = |mut a: usize, mut b: usize, ipdom: &[usize]| -> usize {
        while a != b {
            while rev_idx[a] > rev_idx[b] {
                a = ipdom[a];
            }
            while rev_idx[b] > rev_idx[a] {
                b = ipdom[b];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &post[1..] {
            let mut new = usize::MAX;
            for &s in &succs[b] {
                if ipdom[s] != usize::MAX {
                    new = if new == usize::MAX {
                        s
                    } else {
                        intersect(s, new, &ipdom)
                    };
                }
            }
            if new != usize::MAX && ipdom[b] != new {
                ipdom[b] = new;
                changed = true;
            }
        }
    }
    (rev_idx, ipdom)
}

/// Computes which IR blocks are thread-group uniform (all active threads arrive
/// together) and which IR values hold group-uniform scalars.
fn compute_uniformity(
    p: &Program,
    st: &Structured,
    res: &Resources,
    io_inputs: &[(Value, Input)],
) -> (Vec<bool>, Vec<bool>) {
    let n = p.blocks.len();
    let max_v = p
        .inputs
        .iter()
        .map(|i| i.1.0)
        .chain(p.blocks.iter().flat_map(|b| {
            b.phis.iter().map(|phi| phi.value.0).chain(
                b.insts
                    .iter()
                    .flat_map(|inst| inst.writes.iter().map(|w| w.1.0)),
            )
        }))
        .max()
        .map_or(0, |m| m as usize + 1);
    let mut val_uniform = vec![true; max_v];
    let io_map: HashMap<Value, Input> = io_inputs.iter().copied().collect();
    let param_set: HashSet<Value> = res.params.iter().map(|&(v, _, _)| v).collect();
    for &(_, v) in &p.inputs {
        let ok = param_set.contains(&v)
            || matches!(io_map.get(&v), Some(Input::GroupId(_) | Input::MeshGroup));
        if !ok {
            val_uniform[v.0 as usize] = false;
        }
    }
    for &k in st.masks.keys().chain(st.highs.keys()) {
        if let Some(u) = val_uniform.get_mut(k.0 as usize) {
            *u = false;
        }
    }

    if n == 0 {
        return (Vec::new(), val_uniform);
    }
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut fwd = vec![false; n];
    for &b in &order {
        fwd[b] = true;
    }
    let mut succs = vec![Vec::new(); n];
    for &b in &order {
        match st.exits[b] {
            LaneExit::Jump(t) => succs[b].push(t),
            LaneExit::Branch {
                taken, not_taken, ..
            } => {
                succs[b].push(taken);
                if not_taken != taken {
                    succs[b].push(not_taken);
                }
            }
            LaneExit::End => succs[b].push(n),
            LaneExit::Indirect => {}
        }
    }
    let (rev_idx, ipdom) = compute_postdoms(n, &succs, &fwd);
    let can_exit: Vec<bool> = (0..n).map(|b| fwd[b] && rev_idx[b] != usize::MAX).collect();
    let mut blocks = can_exit.clone();

    loop {
        let mut changed = false;
        for &b in &order {
            let blk_ok = blocks[b];
            for phi in &p.blocks[b].phis {
                let idx = phi.value.0 as usize;
                if !val_uniform[idx] {
                    continue;
                }
                let preds_ok = blk_ok
                    && p.blocks[b]
                        .preds
                        .iter()
                        .filter(|&&q| fwd[q])
                        .all(|&q| blocks[q]);
                let args_ok = phi.args.iter().all(|a| val_uniform[a.0 as usize]);
                if !(preds_ok && args_ok) {
                    val_uniform[idx] = false;
                    changed = true;
                }
            }
            for (ii, inst) in p.blocks[b].insts.iter().enumerate() {
                let iname = inst.name();
                let uniform_copy = iname == "v_mov_b32"
                    && st.guards[b][ii].is_none()
                    && matches!(
                        &inst.fields,
                        Fields::Valu(v) if v.dpp.is_none() && v.sdwa.is_none()
                    );
                let scalar_op = (matches!(inst.fields, Fields::Sop { .. } | Fields::Smem { .. })
                    || uniform_copy)
                    && !iname.starts_with("s_getreg")
                    && !iname.starts_with("s_bcnt")
                    && !iname.starts_with("s_ff1")
                    && !iname.contains("saveexec");
                let reads_ok = blk_ok
                    && scalar_op
                    && inst.reads.iter().all(|&(_, v)| val_uniform[v.0 as usize]);
                for &(_, w) in &inst.writes {
                    if val_uniform[w.0 as usize] && !reads_ok {
                        val_uniform[w.0 as usize] = false;
                        changed = true;
                    }
                }
            }
        }
        let mut next_blocks = can_exit.clone();
        for &b in &order {
            if !can_exit[b] {
                continue;
            }
            if let LaneExit::Branch {
                when,
                taken,
                not_taken,
            } = st.exits[b]
            {
                let branch_uniform = blocks[b]
                    && st.bools.support(when).iter().all(|a| match *a {
                        Atom::Uniform(v) => val_uniform[v.0 as usize],
                        _ => false,
                    });
                if !branch_uniform {
                    let stop = ipdom[b];
                    let mut stack = vec![taken, not_taken];
                    while let Some(x) = stack.pop() {
                        if x >= n || x == stop || !next_blocks[x] {
                            continue;
                        }
                        next_blocks[x] = false;
                        for &s in &succs[x] {
                            if s < n && s != stop && next_blocks[s] {
                                stack.push(s);
                            }
                        }
                    }
                }
            }
        }
        if next_blocks != blocks {
            blocks = next_blocks;
            changed = true;
        }
        if !changed {
            break;
        }
    }
    (blocks, val_uniform)
}

/// Masks read as data in a block other than their defining block, keyed by defining block (`None` = entry).
fn collect_cross_block_masks(
    p: &Program,
    st: &Structured,
    def_block: &HashMap<Value, usize>,
) -> HashMap<Option<usize>, Vec<Value>> {
    let mut out: HashMap<Option<usize>, Vec<Value>> = HashMap::new();
    let mut record = |v: Value, reader_b: usize| {
        let lo = st.highs.get(&v).copied().unwrap_or(v);
        if !st.masks.contains_key(&lo) {
            return;
        }
        let def_b = def_block.get(&lo).copied();
        if def_b != Some(reader_b) {
            let list = out.entry(def_b).or_default();
            if !list.contains(&lo) {
                list.push(lo);
            }
        }
    };
    for (b, block) in p.blocks.iter().enumerate() {
        for inst in &block.insts {
            if reads_mask_as_data(st, inst) {
                for &(_, r) in &inst.reads {
                    record(r, b);
                }
            }
        }
        let exit_when = match st.exits[b] {
            LaneExit::Branch { when, .. } => Some(when),
            _ => None,
        };
        for g in st.guards[b].iter().flatten().copied().chain(exit_when) {
            for a in st.bools.support(g) {
                if let Atom::Opaque(v) = a {
                    record(v, b);
                }
            }
        }
    }
    out
}

fn operand_const_lane(op: &Operand) -> Option<u32> {
    match *op {
        Operand::Const(c) => crate::resource::inline(c),
        Operand::Literal(x) => Some(x),
        _ => None,
    }
}

impl<'a> Lower<'a> {
    fn new(
        x: &'a Parts<'a>,
        bad: &'a HashMap<Value, Why>,
        numeric_selects: bool,
        loop_limit: u32,
    ) -> Self {
        let p = x.p;
        let mut phi_at = HashMap::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phi_at.insert(phi.value, (b, i));
            }
        }
        let def_block: HashMap<Value, usize> = p
            .blocks
            .iter()
            .enumerate()
            .flat_map(|(b, block)| {
                block
                    .insts
                    .iter()
                    .flat_map(move |i| i.writes.iter().map(move |w| (w.1, b)))
                    .chain(block.phis.iter().map(move |phi| (phi.value, b)))
            })
            .collect();
        let (sections, section_out) = compute_sections(p);
        let mut s = Shader::new(x.native.stage == HwStage::Ps);
        s.inputs = x.inputs.clone();
        s.outputs = x.outputs.clone();
        let insts = || p.blocks.iter().flat_map(|b| &b.insts);
        let writes_uavs = insts().any(|i| {
            matches!(
                i.fields,
                Fields::Image { .. } | Fields::Buffer { .. } | Fields::Flat { .. }
            ) && (i.name().contains("store") || i.name().contains("atomic"))
        });
        let (resolved_reads, pure_spill_writes) = resolve_spills(p, x.st);
        let wide_lds =
            insts().any(|i| ds_integer_atomic(i.name()).is_some_and(|(_, wide, _)| wide));
        let mut console_lanes = false;
        let mut lds_barrier_before = HashSet::new();
        let mut lds_barrier_end = HashSet::new();
        let mut wave64 = Wave64Mode::Native;
        let mut total_threads = 0u32;
        let mut title_lds_dwords = 0u32;
        let mut virtual_waves = 0u32;
        let mut ballot_base = 0u32;
        let mut ballot_stride = 0u32;
        let mut exchange_base = 0u32;
        let mut lds_overflow = false;
        let mut need_lds_barriers = false;
        let mut spills = HashMap::new();
        let mut uniform = Vec::new();
        let mut val_uniform = Vec::new();
        let mut cross_block_masks = HashMap::new();

        if let StageIo::Cs(cs) = x.native.io {
            s.threads = Some(cs.threads);
            total_threads = cs.threads.iter().product::<u32>();
            title_lds_dwords = cs.lds_bytes() / 4;
            s.lds_dwords = title_lds_dwords;
            // Code reading other lanes needs the console's wave size and
            // lane order; so do lane masks read as numbers, and LDS shared
            // within one wave, where the console compiler leaves out barriers.
            console_lanes = p.blocks.iter().enumerate().any(|(b, block)| {
                block.insts.iter().enumerate().any(|(idx, i)| {
                    let site = (b, idx);
                    let n = i.name();
                    if (n == "v_readlane_b32" && resolved_reads.contains_key(&site))
                        || (n == "v_writelane_b32" && pure_spill_writes.contains(&site))
                    {
                        return false;
                    }
                    n.starts_with("ds_swizzle")
                        || n.contains("permute")
                        || n.starts_with("v_permlane")
                        || n.starts_with("v_mbcnt")
                        || matches!(n, "v_readlane_b32" | "v_writelane_b32")
                        || matches!(&i.fields, Fields::Valu(v) if v.dpp.is_some())
                        || reads_mask_as_data(x.st, i)
                        || numeric_selects && numeric_select(x.st, i)
                })
            }) || {
                // Masks held as numbers or constant lane sets: the lane
                // tests its own bit by its console lane.
                let whens = x.st.exits.iter().filter_map(|e| match *e {
                    LaneExit::Branch { when, .. } => Some(when),
                    _ => None,
                });
                let selects = insts().filter_map(|i| {
                    let Fields::Valu(v) = &i.fields else {
                        return None;
                    };
                    if !matches!(
                        i.name(),
                        "v_cndmask_b32"
                            | "v_add_co_ci_u32"
                            | "v_sub_co_ci_u32"
                            | "v_subrev_co_ci_u32"
                    ) {
                        return None;
                    }
                    let &Operand::Scalar { code, .. } = v.src.get(2)? else {
                        return None;
                    };
                    if code == NULL {
                        return None;
                    }
                    let m = i.read(Reg::S(code))?;
                    x.st.masks.get(&m).copied()
                });
                let mut seen_bools = HashSet::new();
                let mut seen_phis = HashSet::new();
                let mut work: Vec<Bool> =
                    x.st.guards
                        .iter()
                        .flatten()
                        .flatten()
                        .copied()
                        .chain(whens)
                        .chain(selects)
                        .collect();
                let mut needs_console = false;
                while let Some(g) = work.pop() {
                    if !seen_bools.insert(g) {
                        continue;
                    }
                    for a in x.st.bools.support(g) {
                        match a {
                            Atom::Opaque(_) | Atom::Constant(_) | Atom::Quad(..) => {
                                needs_console = true;
                                break;
                            }
                            Atom::Phi(id) if seen_phis.insert(id) => {
                                if let Some(edges) = x.st.phi_masks.get(&id) {
                                    work.extend(edges.iter().copied());
                                }
                            }
                            _ => {}
                        }
                    }
                    if needs_console {
                        break;
                    }
                }
                needs_console
            };

            let req_mode = if x.native.wave64 {
                x.options.wave64
            } else {
                Wave64Mode::Native
            };
            let has_lds_mem = insts().any(is_lds_mem);
            let has_s_barrier = insts().any(|i| i.name() == "s_barrier");
            let has_swizzle_or_dpp = insts().any(|i| {
                i.name().starts_with("ds_swizzle")
                    || i.name().contains("permute")
                    || matches!(&i.fields, Fields::Valu(v) if v.dpp.is_some())
            });

            match req_mode {
                Wave64Mode::Native => {
                    let wave = if x.native.wave64 { 64 } else { 32 };
                    let one_wave = total_threads <= wave;
                    if console_lanes || one_wave && title_lds_dwords > 0 && has_lds_mem {
                        s.wave_size = Some(wave);
                    }
                }
                Wave64Mode::Wave32 | Wave64Mode::Emulate => {
                    spills = spilled_sgprs(p);
                    (uniform, val_uniform) = compute_uniformity(p, x.st, x.res, &x.io.inputs);
                    let has_non_uniform_firstlane = insts().any(|i| {
                        i.name() == "v_readfirstlane_b32"
                            && matches!(
                                &i.fields,
                                Fields::Valu(v)
                                    if matches!(v.src.first(), Some(Operand::Vector { reg, .. })
                                        if i.read(Reg::V(*reg)).is_some_and(|val| !val_uniform[val.0 as usize]))
                            )
                    });
                    let has_unresolved_readlane = p.blocks.iter().enumerate().any(|(b, blk)| {
                        blk.insts.iter().enumerate().any(|(i, inst)| {
                            inst.name() == "v_readlane_b32"
                                && !spills.contains_key(&(b, i))
                                && !resolved_reads.contains_key(&(b, i))
                        })
                    });
                    let need_emu =
                        console_lanes || !sections.is_empty() || has_non_uniform_firstlane;

                    if req_mode == Wave64Mode::Wave32 {
                        if total_threads <= 32 {
                            // Fits inside one native 32-lane wave.
                            wave64 = Wave64Mode::Wave32;
                            if need_emu || has_lds_mem {
                                s.wave_size = Some(32);
                            }
                        } else {
                            if need_emu {
                                wave64 = Wave64Mode::Wave32;
                                s.wave_size = Some(32);
                                virtual_waves = total_threads.div_ceil(64);
                                ballot_stride = if has_non_uniform_firstlane { 4 } else { 2 };
                                let ballot_dwords = ballot_stride * virtual_waves;
                                let exchange_dwords = if has_unresolved_readlane {
                                    64 * virtual_waves
                                } else {
                                    0
                                };
                                ballot_base = title_lds_dwords;
                                exchange_base = title_lds_dwords + ballot_dwords;
                                let emu_dwords = ballot_dwords + exchange_dwords;
                                s.lds_dwords = title_lds_dwords.saturating_add(emu_dwords);
                                lds_overflow = emu_dwords > 0 && s.lds_dwords > 8192;
                                cross_block_masks = collect_cross_block_masks(p, x.st, &def_block);
                            }
                            if total_threads <= 64 && has_lds_mem && !has_s_barrier {
                                if !console_lanes
                                    && let Some(barriers) = plan_lds_barriers(p, x.st)
                                {
                                    lds_barrier_before = barriers.before;
                                    lds_barrier_end = barriers.at_end;
                                } else {
                                    need_lds_barriers = true;
                                }
                            }
                        }
                    } else {
                        // Wave64Mode::Emulate (`s.wave_size = None`).
                        if need_emu {
                            wave64 = Wave64Mode::Emulate;
                            virtual_waves = total_threads.div_ceil(64);
                            ballot_stride = 2;
                            let ballot_dwords = 2 * virtual_waves;
                            let exchange_dwords = if !sections.is_empty()
                                || has_unresolved_readlane
                                || has_swizzle_or_dpp
                            {
                                64 * virtual_waves
                            } else {
                                0
                            };
                            ballot_base = title_lds_dwords;
                            exchange_base = title_lds_dwords + ballot_dwords;
                            let emu_dwords = ballot_dwords + exchange_dwords;
                            s.lds_dwords = title_lds_dwords.saturating_add(emu_dwords);
                            lds_overflow = emu_dwords > 0 && s.lds_dwords > 8192;
                            cross_block_masks = collect_cross_block_masks(p, x.st, &def_block);
                        }
                        if total_threads > 1 && total_threads <= 64 && has_lds_mem && !has_s_barrier
                        {
                            need_lds_barriers = true;
                        }
                    }
                }
            }
        }
        let mut no_barrier = HashSet::new();
        if let StageIo::Ms(ms) = &x.native.io {
            s.prims = x.prims.clone();
            s.threads = Some([ms.threads, 1, 1]);
            title_lds_dwords = ms.lds_bytes.div_ceil(4);
            s.lds_dwords = title_lds_dwords;
            s.mesh = Some(super::shader::Mesh {
                max_vertices: ms.max_vertices,
                max_primitives: ms.max_primitives,
                topology: ms.topology,
            });
            // Barriers order the lanes' LDS accesses; the console's
            // allocation request needs them too, but the host does not. One
            // lane sends that request, and what it dominates (the path for
            // counts out of range) runs in that lane alone, so its barrier
            // would hang the host.
            let order = crate::ir::reverse_postorder(&p.blocks);
            let mut index = vec![usize::MAX; p.blocks.len()];
            for (n, &b) in order.iter().enumerate() {
                index[b] = n;
            }
            let idom = crate::ir::idoms(&p.blocks, &order, &index);
            let alloc: Vec<usize> = (0..p.blocks.len())
                .filter(|&b| {
                    p.blocks[b].insts.iter().any(|i| {
                        i.name() == "s_sendmsg"
                            && matches!(i.fields, Fields::Sopp { imm } if imm & 0xF == GS_ALLOC_REQ)
                    })
                })
                .collect();
            for &b in &order {
                let mut d = b;
                let dominated = loop {
                    if alloc.contains(&d) {
                        break true;
                    }
                    if idom[d] == d || idom[d] == usize::MAX {
                        break false;
                    }
                    d = idom[d];
                };
                if ms.lds_bytes == 0 || dominated {
                    no_barrier.insert(b);
                }
            }
        }
        Lower {
            s,
            x,
            bad,
            lane_fail: HashMap::new(),
            vals: HashMap::new(),
            floats: HashMap::new(),
            block: vec![None; p.blocks.len()],
            cur: 0,
            bools: HashMap::new(),
            lane_bits: HashMap::new(),
            quad_bits: HashMap::new(),
            quad_visiting: HashSet::new(),
            phi_at,
            mask_phis: HashMap::new(),
            pending_masks: Vec::new(),
            data_phis: Vec::new(),
            read_phis: HashSet::new(),
            params: x.res.params.iter().map(|&(v, p, d)| (v, (p, d))).collect(),
            user_data: None,
            user_reads: BTreeMap::new(),
            samplers: Vec::new(),
            descriptors: x
                .res
                .descriptors
                .iter()
                .map(|d| ((d.site, d.role == Role::Sampler), &d.source))
                .collect(),
            scalar: x.res.scalar.iter().copied().collect(),
            exports: x.io.exports.iter().map(|e| (e.site, e)).collect(),
            interps: x.io.interps.iter().map(|(s, i)| (*s, i)).collect(),
            handles: HashMap::new(),
            edges: HashMap::new(),
            exported: vec![false; p.blocks.len()],
            rpo: vec![usize::MAX; p.blocks.len()],
            tail: vec![None; p.blocks.len()],
            writes_uavs,
            wide_lds,
            src0: None,
            console_lanes,
            numeric_selects,
            wave64,
            total_threads,
            title_lds_dwords,
            virtual_waves,
            ballot_base,
            ballot_stride,
            exchange_base,
            lds_overflow,
            need_lds_barriers,
            spills,
            uniform,
            val_uniform,
            wave_ids: None,
            guard_depth: 0,
            cross_block_masks,
            ballot_fail: HashMap::new(),
            flat: None,
            no_barrier,
            lds_barrier_before,
            lds_barrier_end,
            half_outputs: half_outputs(&x.native.io, &x.outputs),
            half_inputs: half_inputs(&x.native.io, &x.inputs),
            mask_data: match x.native.stage {
                HwStage::Cs => MaskData::Console,
                HwStage::Ps | HwStage::Vs => MaskData::Host,
                _ => MaskData::None,
            },
            mask_ok: false,
            ballots: HashMap::new(),
            balloting: HashSet::new(),
            def_block,
            cur_ir: 0,
            sections,
            section_out,
            code_table: None,
            scratch: None,
            loop_limit,
            spin_limit: super::spin_limit().min(loop_limit),
            barrier_loops: HashSet::new(),
            steps: [None, None, None],
            abort: None,
            resolved_reads,
            pure_spill_writes,
            spin: HashSet::new(),
            #[cfg(test)]
            spun: 0,
        }
    }

    /// The dword of the program's code at byte `byte` (a constant table
    /// the compiler appends to the code), as an internal constant array.
    // shortcut: an index outside the table reads its last dword (the
    // console would read whatever memory follows the code).
    fn code_dword(&mut self, byte: V) -> Result<V, Why> {
        let (g, base, len) = match self.code_table {
            Some(t) => t,
            None => {
                let first = self
                    .x
                    .res
                    .scalar
                    .iter()
                    .map(|s| s.1)
                    .chain(self.x.res.descriptors.iter().flat_map(|d| {
                        d.source.leaves().into_iter().filter_map(|l| match l {
                            Source::Built { address, .. } => Some(*address),
                            _ => None,
                        })
                    }))
                    .filter(|a| a.param == CODE)
                    .map(|a| a.offset & !3)
                    .min()
                    .ok_or("a code table")?;
                let code = &self.x.native.code;
                let words: Vec<u32> = code
                    .get(first as usize..)
                    .ok_or("a code table past the code")?
                    .chunks(4)
                    .map(|c| {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        u32::from_le_bytes(w)
                    })
                    .collect();
                if words.is_empty() {
                    return Err("a code table past the code");
                }
                let elems: Vec<V> = words.iter().map(|&w| self.s.m.i32(w)).collect();
                let arr = self.s.m.array(words.len() as u32, self.s.t.i32);
                let init = self.s.m.aggregate(arr, &elems);
                let g = self.s.m.constant_global("code", init);
                let t = (g, first, words.len() as u32);
                self.code_table = Some(t);
                t
            }
        };
        let c = self.cur;
        let k = self.s.m.i32(base);
        let rel = self.s.m.bin(c, Bin::Sub, byte, k);
        let two = self.s.m.i32(2);
        let index = self.s.m.bin(c, Bin::LShr, rel, two);
        let last = self.s.m.i32(len - 1);
        let over = self.s.m.cmp(c, pred::IUGT, index, last);
        let index = self.s.m.select(c, over, last, index);
        let at = self.s.m.element(c, g, index);
        Ok(self.s.m.load(c, at))
    }

    /// Private (scratch) memory, `scratch_load_*` and `scratch_store_*`.
    /// Each lane has its own: the address (VGPR, SGPR or none, plus the
    /// offset) is a byte offset into the lane's memory, and the hardware
    /// interleaves the lanes' dwords itself (RDNA 2 ISA, "Scratch
    /// (Private)"; LLVM AMDGPUUsage, "Private Segment"). On the host it is
    /// an array of dwords per lane (`alloca`, as DXC makes local arrays).
    // shortcut: a dword access is taken as aligned, a byte or short one
    // not to cross a dword (console compilers keep private data aligned).
    fn scratch(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat {
            dst,
            addr,
            data,
            saddr,
            offset,
            lds,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        if *lds {
            return Err("a scratch load into LDS");
        }
        let name = inst.name();
        let (load, rest) = match name.strip_prefix("scratch_load_") {
            Some(rest) => (true, rest),
            None => (false, name.strip_prefix("scratch_store_").ok_or(name)?),
        };
        let width = scratch_width(rest).ok_or(name)?;
        let (signed, hi) = (rest.starts_with('s'), rest.ends_with("_hi"));
        let guard = self.guard(b, i)?;
        let base = match (addr, saddr) {
            (Some(o), _) => self.vgprs(inst, o, 1)?[0],
            (None, Some(o)) => self.operand(inst, o)?,
            (None, None) => self.s.m.i32(0),
        };
        let (arr, dwords) = match self.scratch {
            Some(a) => a,
            None => {
                let dwords = scratch_bytes(self.x.p).div_ceil(4);
                let ty = self.s.m.array(dwords, self.s.t.i32);
                let a = (self.s.m.alloca(self.s.entry, ty), dwords);
                self.scratch = Some(a);
                a
            }
        };
        let c = self.cur;
        let off = self.s.m.i32(i32::from(*offset) as u32);
        let at = self.s.m.bin(c, Bin::Add, base, off);
        let two = self.s.m.i32(2);
        let first = self.s.m.bin(c, Bin::LShr, at, two);
        // Dword `k` of the access: its element (the first when out of
        // range) and whether it is in range.
        let element = |l: &mut Self, k: u32| -> (V, V) {
            let c = l.cur;
            let k = l.s.m.i32(k);
            let index = l.s.m.bin(c, Bin::Add, first, k);
            let n = l.s.m.i32(dwords);
            let inside = l.s.m.cmp(c, pred::IULT, index, n);
            let zero = l.s.m.i32(0);
            let index = l.s.m.select(c, inside, index, zero);
            (l.s.m.element(c, arr, index), inside)
        };
        let three = self.s.m.i32(3);
        let byte = self.s.m.bin(c, Bin::And, at, three);
        let shift = self.s.m.bin(c, Bin::Shl, byte, three);
        if load {
            let Some(Operand::Vector { reg, .. }) = *dst else {
                return Err("a scratch load result");
            };
            let zero = self.s.m.i32(0);
            if width >= 4 {
                for k in 0..width / 4 {
                    let (p, inside) = element(self, k);
                    let x = self.s.m.load(c, p);
                    let x = self.s.m.select(c, inside, x, zero);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let (p, inside) = element(self, 0);
            let w = self.s.m.load(c, p);
            let w = self.s.m.select(c, inside, w, zero);
            let x = self.s.m.bin(c, Bin::LShr, w, shift);
            let left = self.s.m.i32(32 - 8 * width);
            let x = self.s.m.bin(c, Bin::Shl, x, left);
            let mut x = self
                .s
                .m
                .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left);
            if rest.contains("d16") {
                // Into one half of the VGPR (extended to 16 bits); the
                // other half keeps its bits.
                let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
                    Some(Ok(o)) => o,
                    _ => self.s.m.i32(0),
                };
                let low = self.s.m.i32(0xFFFF);
                let h = self.s.m.bin(c, Bin::And, x, low);
                let (h, keep) = if hi {
                    let sixteen = self.s.m.i32(16);
                    (self.s.m.bin(c, Bin::Shl, h, sixteen), low)
                } else {
                    (h, self.s.m.i32(0xFFFF_0000))
                };
                let o = self.s.m.bin(c, Bin::And, old, keep);
                x = self.s.m.bin(c, Bin::Or, o, h);
            }
            self.write(inst, Reg::V(reg), x, guard);
            return Ok(());
        }
        let data = data.as_ref().ok_or("scratch store data")?;
        let vals = self.vgprs(inst, data, width.div_ceil(4) as u8)?;
        for (k, mut x) in vals.into_iter().enumerate() {
            let (p, inside) = element(self, k as u32);
            let g = match guard {
                Some(g) => self.s.m.bin(self.cur, Bin::And, g, inside),
                None => inside,
            };
            let at = self.begin_guarded(Some(g));
            let c = self.cur;
            if width < 4 {
                // The byte or short within its dword (`_d16_hi`: from the
                // high half of the VGPR).
                if hi {
                    let sixteen = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, sixteen);
                }
                let mask = self.s.m.i32((1 << (8 * width)) - 1);
                let mask = self.s.m.bin(c, Bin::Shl, mask, shift);
                let x = self.s.m.bin(c, Bin::Shl, x, shift);
                let x = self.s.m.bin(c, Bin::And, x, mask);
                let ones = self.s.m.i32(u32::MAX);
                let keep = self.s.m.bin(c, Bin::Xor, mask, ones);
                let old = self.s.m.load(c, p);
                let old = self.s.m.bin(c, Bin::And, old, keep);
                let new = self.s.m.bin(c, Bin::Or, old, x);
                self.s.m.store(c, p, new);
            } else {
                self.s.m.store(c, p, x);
            }
            self.end_guarded(at, &[]);
        }
        Ok(())
    }

    /// A global memory access (`global_load_*`, `global_store_*`): from the
    /// program's constant table, or from a root SRV/UAV at the address the
    /// title sets.
    fn flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, data, lds, .. } = &inst.fields else {
            unreachable!()
        };
        if *lds {
            return Err("a global load directly into LDS");
        }
        let name = inst.name();
        if let Some(rest) = name.strip_prefix("global_store_") {
            let n = match rest {
                "dword" => 1u32,
                "dwordx2" => 2,
                "dwordx3" => 3,
                "dwordx4" => 4,
                _ => return Err("a global memory write"),
            };
            let a = *self
                .scalar
                .get(&(b, i))
                .ok_or("an untracked global address")?;
            if a.param == CODE {
                return Err("a global memory write");
            }
            let guard = self.guard(b, i)?;
            let vals = self.vgprs(inst, data.as_ref().ok_or("a global store value")?, n as u8)?;
            let c = self.cur;
            let mut off = self.s.m.i32(a.offset);
            if let Some(d) = a.dynamic {
                let d = self.i32_of(d)?;
                off = self.s.m.bin(c, Bin::Add, off, d);
            }
            let src = Source::Built {
                address: a,
                bits: 0,
                words: [0, 0],
            };
            let h = self.handle((b, i), &src, false, (11 | 0x1000, 0))?;
            let iu = self.s.m.undef(self.s.t.i32);
            let mut args = vec![h, off, iu];
            args.extend(&vals);
            args.resize(3 + 4, iu);
            args.push(self.s.m.i8((1u8 << n) - 1));
            args.push(self.s.m.i32(4));
            let at = self.begin_guarded(guard);
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                "rawBufferStore.i32",
                Attr::NoUnwind,
                void,
                140,
                &args,
            );
            self.end_guarded(at, &[]);
            return Ok(());
        }
        let rest = name
            .strip_prefix("global_load_")
            .ok_or("a global store or atomic without root UAV address support")?;
        let sub = subword_load(rest);
        let n = match rest {
            "dword" => 1,
            "dwordx2" => 2,
            "dwordx3" => 3,
            "dwordx4" => 4,
            "dword_addtid" => return Err("a global load with a console thread ID offset"),
            _ if sub.is_some() => 1,
            _ => return Err(name),
        };
        let a = *self
            .scalar
            .get(&(b, i))
            .ok_or("an untracked global address")?;
        let guard = self.guard(b, i)?;
        let Some(Operand::Vector { reg, .. }) = *dst else {
            return Err("a global load result");
        };
        let c = self.cur;
        let mut off = self.s.m.i32(a.offset);
        if let Some(d) = a.dynamic {
            let d = self.i32_of(d)?;
            off = self.s.m.bin(c, Bin::Add, off, d);
        }
        let h = if a.param == CODE {
            None
        } else {
            let src = Source::Built {
                address: a,
                bits: 0,
                words: [0, 0],
            };
            let u = if self
                .used_kind(&src)
                .is_some_and(|u| u.kind == ResourceType::Uav)
            {
                0x1000
            } else {
                0
            };
            Some(self.handle((b, i), &src, false, (11 | u, 0))?)
        };
        let dword = |l: &mut Self, byte: V| -> Result<V, Why> {
            match h {
                None => l.code_dword(byte),
                Some(h) => {
                    let u = l.s.m.undef(l.s.t.i32);
                    Ok(l.raw_dword(h, [byte, u]))
                }
            }
        };
        if let Some((width, signed, half)) = sub {
            let x = self.read_subword(off, width, signed, dword)?;
            self.write_subword_result(inst, reg, x, half, guard)?;
        } else {
            for k in 0..n {
                let four = self.s.m.i32(4 * k);
                let at = self.s.m.bin(c, Bin::Add, off, four);
                let x = dword(self, at)?;
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
        }
        Ok(())
    }

    /// Whether this shader emulates 64-lane waves across multiple host waves or arbitrary width.
    fn emulating_wave64(&self) -> bool {
        self.wave64 == Wave64Mode::Emulate
            || (self.wave64 == Wave64Mode::Wave32 && self.total_threads > 32)
    }

    /// Thread-group barrier fencing shared memory only (`barrier(1 | 8)`).
    fn lds_barrier(&mut self, block: usize) {
        debug_assert_eq!(self.guard_depth, 0);
        let mode = self.s.m.i32(1 | 8);
        let void = self.s.t.void;
        self.s
            .op(block, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    /// Ensures `(flat_tid, console_lane, vwave)` exist in `self.s.entry` and returns them.
    fn ensure_wave_ids(&mut self) -> (V, V, V) {
        if let Some(ids) = self.wave_ids {
            return ids;
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat_in = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let flat = if self.wave64 == Wave64Mode::Emulate {
            flat_in
        } else {
            let first = self.s.op(
                e,
                "waveReadLaneFirst.i32",
                Attr::NoUnwind,
                i32t,
                118,
                &[flat_in],
            );
            let wave = self.s.wave_size.expect("set with console lanes");
            let mask = self.s.m.i32(!(wave - 1));
            let base = self.s.m.bin(e, Bin::And, first, mask);
            let lane = self
                .s
                .op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
            self.s.flags |= flags::WAVE_OPS;
            self.s.m.bin(e, Bin::Add, base, lane)
        };
        let k63 = self.s.m.i32(63);
        let k6 = self.s.m.i32(6);
        let console_lane = self.s.m.bin(e, Bin::And, flat, k63);
        let vwave = self.s.m.bin(e, Bin::LShr, flat, k6);
        let ids = (flat, console_lane, vwave);
        self.wave_ids = Some(ids);
        ids
    }

    /// Byte offset of LDS scratch dword `base + vwave * stride + offset`.
    fn scratch_byte(&mut self, block: usize, base: u32, stride: u32, vwave: V, offset: u32) -> V {
        let base_v = self.s.m.i32(base);
        let stride_v = self.s.m.i32(stride);
        let wave_off = self.s.m.bin(block, Bin::Mul, vwave, stride_v);
        let mut idx = self.s.m.bin(block, Bin::Add, base_v, wave_off);
        if offset != 0 {
            let off_v = self.s.m.i32(offset);
            idx = self.s.m.bin(block, Bin::Add, idx, off_v);
        }
        let two = self.s.m.i32(2);
        self.s.m.bin(block, Bin::Shl, idx, two)
    }

    /// Byte offset of LDS exchange dword `exchange_base + vwave * 64 + lane_offset`.
    fn exchange_byte(&mut self, block: usize, vwave: V, lane_offset: V) -> V {
        let base_v = self.s.m.i32(self.exchange_base);
        let k64 = self.s.m.i32(64);
        let wave_off = self.s.m.bin(block, Bin::Mul, vwave, k64);
        let row = self.s.m.bin(block, Bin::Add, base_v, wave_off);
        let idx = self.s.m.bin(block, Bin::Add, row, lane_offset);
        let two = self.s.m.i32(2);
        self.s.m.bin(block, Bin::Shl, idx, two)
    }

    /// Applies associative `op` to two `i32` bit-patterns in the current block.
    fn apply_wave_op(&mut self, op: super::wave::Op, a: V, b: V) -> V {
        use super::wave::Op;
        let c = self.cur;
        match op {
            Op::AddU => self.s.m.bin(c, Bin::Add, a, b),
            Op::Or => self.s.m.bin(c, Bin::Or, a, b),
            Op::And => self.s.m.bin(c, Bin::And, a, b),
            Op::Xor => self.s.m.bin(c, Bin::Xor, a, b),
            Op::MinU | Op::MaxU | Op::MinI | Op::MaxI => {
                let p = match op {
                    Op::MinU => pred::IULT,
                    Op::MaxU => pred::IUGT,
                    Op::MinI => pred::SLT,
                    _ => pred::SGT,
                };
                let cmp = self.s.m.cmp(c, p, a, b);
                self.s.m.select(c, cmp, a, b)
            }
            Op::AddF => {
                let fa = self.as_float(a);
                let fb = self.as_float(b);
                let r = self.s.m.bin(c, Bin::Add, fa, fb);
                self.as_int(r)
            }
            Op::MaxF => {
                let fa = self.as_float(a);
                let fb = self.as_float(b);
                let r = self.binary(35, fa, fb);
                self.as_int(r)
            }
            Op::MinF => {
                let fa = self.as_float(a);
                let fb = self.as_float(b);
                let r = self.binary(36, fa, fb);
                self.as_int(r)
            }
        }
    }

    /// Exchanges one `i32` per 32-lane half across each virtual 64-lane wave using `ballot_base`.
    /// Must run in `self.cur` in convergent control flow (`uniform_here = true`) outside any guard.
    fn exchange_two_halves(
        &mut self,
        val: V,
        partial_h1_default: V,
        uniform_here: bool,
    ) -> Result<(V, V), Why> {
        if self.guard_depth > 0 || !uniform_here {
            return Err("a cross-lane wave operation in divergent control flow");
        }
        let (_, console_lane, vwave) = self.ensure_wave_ids();
        let stride = self.ballot_stride;
        let base = self.ballot_base;
        self.lds_barrier(self.cur);
        let k0 = self.s.m.i32(0);
        let k32 = self.s.m.i32(32);
        let is_h0 = self.s.m.cmp(self.cur, pred::EQ, console_lane, k0);
        let at0 = self.begin_guarded(Some(is_h0));
        let p0 = self.scratch_byte(self.cur, base, stride, vwave, 0);
        self.lds_store_dword(p0, val);
        let tail = self.total_threads & 63;
        if tail != 0 && tail <= 32 {
            let last_w = self.s.m.i32(self.virtual_waves - 1);
            let is_last = self.s.m.cmp(self.cur, pred::EQ, vwave, last_w);
            let at_last = self.begin_guarded(Some(is_last));
            let p1 = self.scratch_byte(self.cur, base, stride, vwave, 1);
            self.lds_store_dword(p1, partial_h1_default);
            self.end_guarded(at_last, &[]);
        }
        self.end_guarded(at0, &[]);

        let is_h1 = self.s.m.cmp(self.cur, pred::EQ, console_lane, k32);
        let at1 = self.begin_guarded(Some(is_h1));
        let p1 = self.scratch_byte(self.cur, base, stride, vwave, 1);
        self.lds_store_dword(p1, val);
        self.end_guarded(at1, &[]);

        self.lds_barrier(self.cur);
        let c = self.cur;
        let r0_b = self.scratch_byte(c, base, stride, vwave, 0);
        let r1_b = self.scratch_byte(c, base, stride, vwave, 1);
        Ok((self.lds_load_dword(r0_b), self.lds_load_dword(r1_b)))
    }

    /// Exchanges `val` from every console lane across the virtual 64-lane wave via LDS
    /// and returns the value held by `lane`. Requires a uniform block.
    fn exchange_64(&mut self, val: V, lane: V) -> Result<V, Why> {
        if self.guard_depth > 0 || !self.uniform.get(self.cur_ir).copied().unwrap_or(false) {
            return Err("a cross-lane read in divergent control flow");
        }
        let (_, console_lane, vwave) = self.ensure_wave_ids();
        self.lds_barrier(self.cur);
        let wb = self.exchange_byte(self.cur, vwave, console_lane);
        self.lds_store_dword(wb, val);
        self.lds_barrier(self.cur);
        let k63 = self.s.m.i32(63);
        let idx = self.s.m.bin(self.cur, Bin::And, lane, k63);
        let rb = self.exchange_byte(self.cur, vwave, idx);
        Ok(self.lds_load_dword(rb))
    }

    /// Emulates `v_readfirstlane_b32(val)` across all 64 lanes of a virtual wave.
    fn emulated_readfirstlane(&mut self, val: V, guard: Option<V>) -> Result<V, Why> {
        if self.guard_depth > 0 || !self.uniform.get(self.cur_ir).copied().unwrap_or(false) {
            return Err("v_readfirstlane in divergent control flow");
        }
        let (_, console_lane, vwave) = self.ensure_wave_ids();
        let stride = self.ballot_stride;
        let base = self.ballot_base;
        let act = guard.unwrap_or_else(|| self.s.m.bool(true));
        if self.wave64 == Wave64Mode::Wave32 {
            let c = self.cur;
            let i32t = self.s.t.i32;
            let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
            let r = self
                .s
                .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[act]);
            self.s.flags |= flags::WAVE_OPS;
            let mask = self.s.m.extract(c, r, 0);
            let k0 = self.s.m.i32(0);
            let any = self.s.m.cmp(c, pred::NE, mask, k0);
            let bit = self
                .s
                .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[mask]);
            let safe_idx = self.s.m.select(c, any, bit, k0);
            let picked = self.read_lane(val, safe_idx);

            self.lds_barrier(self.cur);
            let k0 = self.s.m.i32(0);
            let k32 = self.s.m.i32(32);
            let is_h0 = self.s.m.cmp(self.cur, pred::EQ, console_lane, k0);
            let at0 = self.begin_guarded(Some(is_h0));
            let pm0 = self.scratch_byte(self.cur, base, stride, vwave, 0);
            let pv0 = self.scratch_byte(self.cur, base, stride, vwave, 1);
            self.lds_store_dword(pm0, mask);
            self.lds_store_dword(pv0, picked);
            let tail = self.total_threads & 63;
            if tail != 0 && tail <= 32 {
                let last_w = self.s.m.i32(self.virtual_waves - 1);
                let is_last = self.s.m.cmp(self.cur, pred::EQ, vwave, last_w);
                let at_last = self.begin_guarded(Some(is_last));
                let pm1 = self.scratch_byte(self.cur, base, stride, vwave, 2);
                self.lds_store_dword(pm1, k0);
                self.end_guarded(at_last, &[]);
            }
            self.end_guarded(at0, &[]);

            let is_h1 = self.s.m.cmp(self.cur, pred::EQ, console_lane, k32);
            let at1 = self.begin_guarded(Some(is_h1));
            let pm1 = self.scratch_byte(self.cur, base, stride, vwave, 2);
            let pv1 = self.scratch_byte(self.cur, base, stride, vwave, 3);
            self.lds_store_dword(pm1, mask);
            self.lds_store_dword(pv1, picked);
            self.end_guarded(at1, &[]);

            self.lds_barrier(self.cur);
            let c = self.cur;
            let pm0 = self.scratch_byte(c, base, stride, vwave, 0);
            let pv0 = self.scratch_byte(c, base, stride, vwave, 1);
            let pv1 = self.scratch_byte(c, base, stride, vwave, 3);
            let m0 = self.lds_load_dword(pm0);
            let v0 = self.lds_load_dword(pv0);
            let v1 = self.lds_load_dword(pv1);
            let use0 = self.s.m.cmp(c, pred::NE, m0, k0);
            return Ok(self.s.m.select(c, use0, v0, v1));
        }

        // Emulate mode: find lowest active console lane via atomicMin.
        self.lds_barrier(self.cur);
        let k0 = self.s.m.i32(0);
        let kmax = self.s.m.i32(u32::MAX);
        let is_init = self.s.m.cmp(self.cur, pred::EQ, console_lane, k0);
        let at0 = self.begin_guarded(Some(is_init));
        let pw = self.scratch_byte(self.cur, base, stride, vwave, 0);
        let pv = self.scratch_byte(self.cur, base, stride, vwave, 1);
        self.lds_store_dword(pw, kmax);
        self.lds_store_dword(pv, k0);
        self.end_guarded(at0, &[]);

        self.lds_barrier(self.cur);
        let at_act = self.begin_guarded(guard);
        let pw_b = self.scratch_byte(self.cur, base, stride, vwave, 0);
        if self.wide_lds {
            self.lds_update_dword(pw_b, |l, old| {
                let lt = l.s.m.cmp(l.cur, pred::IULT, console_lane, old);
                l.s.m.select(l.cur, lt, console_lane, old)
            });
        } else {
            let pw = self.lds_ptr(pw_b);
            let _ = self
                .s
                .m
                .atomic(self.cur, super::bitcode::Rmw::UMin, pw, console_lane);
        }
        self.end_guarded(at_act, &[]);

        self.lds_barrier(self.cur);
        let pw = self.scratch_byte(self.cur, base, stride, vwave, 0);
        let winner = self.lds_load_dword(pw);
        let is_win = self.s.m.cmp(self.cur, pred::EQ, winner, console_lane);
        let at_win = self.begin_guarded(Some(is_win));
        let pv = self.scratch_byte(self.cur, base, stride, vwave, 1);
        self.lds_store_dword(pv, val);
        self.end_guarded(at_win, &[]);

        self.lds_barrier(self.cur);
        let pv = self.scratch_byte(self.cur, base, stride, vwave, 1);
        Ok(self.lds_load_dword(pv))
    }

    /// Pre-emits cross-block wave ballots while `self.cur` is at the end of `def_b` (`None` = entry).
    fn preemit_masks(&mut self, def_b: Option<usize>) {
        let Some(los) = self.cross_block_masks.get(&def_b).cloned() else {
            return;
        };
        let uniform_here = match def_b {
            None => true,
            Some(b) => self.uniform.get(b).copied().unwrap_or(false),
        };
        for lo in los {
            if self.ballots.contains_key(&lo) || self.ballot_fail.contains_key(&lo) {
                continue;
            }
            if let Err(why) = self.compute_ballot_in_cur(lo, uniform_here) {
                self.ballot_fail.insert(lo, why);
            }
        }
    }

    /// Materializes 64-lane ballots needed as data or opaque guards by `(b, i)` before `inst` runs.
    fn materialize_masks(&mut self, b: usize, i: usize, inst: &Inst) {
        let mut todo = Vec::new();
        let mut add = |v: Value| {
            let lo = self.x.st.highs.get(&v).copied().unwrap_or(v);
            if self.x.st.masks.contains_key(&lo)
                && !self.ballots.contains_key(&lo)
                && !self.ballot_fail.contains_key(&lo)
                && !todo.contains(&lo)
            {
                todo.push(lo);
            }
        };
        if self.mask_ok && reads_mask_as_data(self.x.st, inst) {
            for &(_, r) in &inst.reads {
                add(r);
            }
        }
        if let Some(g) = self.x.st.guards[b][i] {
            for a in self.x.st.bools.support(g) {
                if let Atom::Opaque(v) = a {
                    add(v);
                }
            }
        }
        let uniform_here = self.uniform.get(b).copied().unwrap_or(false);
        for lo in todo {
            if let Err(why) = self.compute_ballot_in_cur(lo, uniform_here) {
                self.ballot_fail.insert(lo, why);
            }
        }
    }

    /// Emits a 64-lane ballot for active mask `lo` into `self.cur`.
    fn compute_ballot_in_cur(&mut self, lo: Value, uniform_here: bool) -> Result<(V, V), Why> {
        let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
        if !self.x.st.active.contains(&lo) {
            return Err("a lane mask holding lanes elsewhere read as data");
        }
        let c = self.cur;
        if !self.balloting.insert(lo) {
            return Err("a lane mask whose lanes depend on its own value");
        }
        let bit = self.boolean(m, c);
        self.balloting.remove(&lo);
        let bit = bit?;
        let xy = if self.wave64 == Wave64Mode::Emulate {
            if self.guard_depth > 0 || !uniform_here {
                return Err("a 64-lane wave ballot in divergent control flow");
            }
            let (_, console_lane, vwave) = self.ensure_wave_ids();
            let base = self.ballot_base;
            let stride = self.ballot_stride;
            self.lds_barrier(self.cur);
            let k0 = self.s.m.i32(0);
            let is0 = self.s.m.cmp(self.cur, pred::EQ, console_lane, k0);
            let at0 = self.begin_guarded(Some(is0));
            let p0 = self.scratch_byte(self.cur, base, stride, vwave, 0);
            let p1 = self.scratch_byte(self.cur, base, stride, vwave, 1);
            self.lds_store_dword(p0, k0);
            self.lds_store_dword(p1, k0);
            self.end_guarded(at0, &[]);

            self.lds_barrier(self.cur);
            let at_bit = self.begin_guarded(Some(bit));
            let bcur = self.cur;
            let k32 = self.s.m.i32(32);
            let k31 = self.s.m.i32(31);
            let k1 = self.s.m.i32(1);
            let half = self.s.m.cmp(bcur, pred::IUGE, console_lane, k32);
            let half_off = self.s.m.cast(bcur, Cast::ZExt, half, self.s.t.i32);
            let base_v = self.s.m.i32(base);
            let stride_v = self.s.m.i32(stride);
            let wave_off = self.s.m.bin(bcur, Bin::Mul, vwave, stride_v);
            let slot_base = self.s.m.bin(bcur, Bin::Add, base_v, wave_off);
            let slot_idx = self.s.m.bin(bcur, Bin::Add, slot_base, half_off);
            let two = self.s.m.i32(2);
            let slot_byte = self.s.m.bin(bcur, Bin::Shl, slot_idx, two);
            let sh = self.s.m.bin(bcur, Bin::And, console_lane, k31);
            let mask_bit = self.s.m.bin(bcur, Bin::Shl, k1, sh);
            if self.wide_lds {
                self.lds_update_dword(slot_byte, |l, old| l.s.m.bin(l.cur, Bin::Or, old, mask_bit));
            } else {
                let ptr = self.lds_ptr(slot_byte);
                let _ = self
                    .s
                    .m
                    .atomic(bcur, super::bitcode::Rmw::Or, ptr, mask_bit);
            }
            self.end_guarded(at_bit, &[]);

            self.lds_barrier(self.cur);
            let c = self.cur;
            let p0 = self.scratch_byte(c, base, stride, vwave, 0);
            let p1 = self.scratch_byte(c, base, stride, vwave, 1);
            (self.lds_load_dword(p0), self.lds_load_dword(p1))
        } else {
            let i32t = self.s.t.i32;
            let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
            let r = self
                .s
                .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
            self.s.flags |= flags::WAVE_OPS;
            let half = self.s.m.extract(c, r, 0);
            let zero = self.s.m.i32(0);
            self.exchange_two_halves(half, zero, uniform_here)?
        };
        self.ballots.insert(lo, xy);
        Ok(xy)
    }

    /// An instruction inside a whole-wave section: the values that leave
    /// the section are made from what the simulation found them to be.
    fn section_inst(&mut self, b: usize, i: usize, start: usize) {
        const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";
        let inst = &self.x.p.blocks[b].insts[i];
        for &(_, w) in &inst.writes {
            let val = if self.section_out.contains(&w) {
                let shape = self.sections[&b][&start].shapes.get(&w).copied();
                match shape.ok_or(UNKNOWN).and_then(|sh| self.shape(sh)) {
                    Ok(x) => Val::I32(x),
                    Err(why) => Val::Missing(why),
                }
            } else {
                Val::Missing(UNKNOWN)
            };
            self.vals.insert(w, val);
        }
    }

    /// A value a whole-wave section leaves, as DXIL wave operations over the
    /// host's active lanes: the console's result with console lanes; on
    /// another wave size, what the HLSL wave operation the console compiler
    /// expanded gives on that wave.
    fn shape(&mut self, shape: super::wave::Shape) -> Result<V, Why> {
        use super::wave::{Op, Shape};
        let (op, src, mask, ident, prefix) = match shape {
            Shape::K(k) => return Ok(self.s.m.i32(k)),
            Shape::Value(v) => return self.i32_of(v),
            Shape::Active(op, src, mask, ident) => (op, src, mask, ident, None),
            Shape::Masked(src, mask, ident) => {
                let c = self.cur;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let on = self.boolean(m, c)?;
                let x = self.i32_of(src)?;
                let id = self.s.m.i32(ident);
                return Ok(self.s.m.select(c, on, x, id));
            }
            Shape::Prefix(op, src, mask, ident, inclusive) => {
                (op, src, mask, ident, Some(inclusive))
            }
        };
        if self.mask_data == MaskData::None {
            return Err("a wave reduction in a mesh shader");
        }
        let c = self.cur;
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        let on = self.boolean(m, c)?;
        let x = self.i32_of(src)?;
        let id = self.s.m.i32(ident);
        let x = self.s.m.select(c, on, x, id);

        if self.wave64 == Wave64Mode::Emulate {
            if self.guard_depth > 0 || !self.uniform.get(self.cur_ir).copied().unwrap_or(false) {
                return Err("a wave reduction in divergent control flow");
            }
            let (_, console_lane, vwave) = self.ensure_wave_ids();
            let max_in_vwave = self.total_threads.min(64);
            if let Some(inclusive) = prefix {
                if !matches!(op, Op::AddU | Op::AddF) {
                    return Err("a wave prefix of an operation other than a sum");
                }
                let mut cur_val = x;
                for shift in 0..6 {
                    let stride = 1u32 << shift;
                    if stride >= max_in_vwave {
                        break;
                    }
                    self.lds_barrier(self.cur);
                    let wp = self.exchange_byte(self.cur, vwave, console_lane);
                    self.lds_store_dword(wp, cur_val);
                    self.lds_barrier(self.cur);
                    let ks = self.s.m.i32(stride);
                    let active = self.s.m.cmp(self.cur, pred::IUGE, console_lane, ks);
                    let at = self.begin_guarded(Some(active));
                    let partner_lane = self.s.m.bin(self.cur, Bin::Sub, console_lane, ks);
                    let rp = self.exchange_byte(self.cur, vwave, partner_lane);
                    let partner = self.lds_load_dword(rp);
                    let sum = self.apply_wave_op(op, partner, cur_val);
                    cur_val = self.end_guarded(at, &[(sum, cur_val)])[0];
                }
                if !inclusive {
                    self.lds_barrier(self.cur);
                    let wp = self.exchange_byte(self.cur, vwave, console_lane);
                    self.lds_store_dword(wp, cur_val);
                    self.lds_barrier(self.cur);
                    let k0 = self.s.m.i32(0);
                    let k1 = self.s.m.i32(1);
                    let not0 = self.s.m.cmp(self.cur, pred::NE, console_lane, k0);
                    let at = self.begin_guarded(Some(not0));
                    let prev_lane = self.s.m.bin(self.cur, Bin::Sub, console_lane, k1);
                    let rp = self.exchange_byte(self.cur, vwave, prev_lane);
                    let prev = self.lds_load_dword(rp);
                    cur_val = self.end_guarded(at, &[(prev, id)])[0];
                }
                return Ok(cur_val);
            } else {
                self.lds_barrier(self.cur);
                let wp = self.exchange_byte(self.cur, vwave, console_lane);
                self.lds_store_dword(wp, x);
                let partial = !self.total_threads.is_multiple_of(64);
                let (flat, _, _) = self.ensure_wave_ids();
                let total_v = self.s.m.i32(self.total_threads);
                for shift in (0..6).rev() {
                    let stride = 1u32 << shift;
                    if stride >= max_in_vwave {
                        continue;
                    }
                    self.lds_barrier(self.cur);
                    let b = self.cur;
                    let ks = self.s.m.i32(stride);
                    let mut active = self.s.m.cmp(b, pred::IULT, console_lane, ks);
                    if partial {
                        let partner_flat = self.s.m.bin(b, Bin::Add, flat, ks);
                        let in_bounds = self.s.m.cmp(b, pred::IULT, partner_flat, total_v);
                        active = self.s.m.bin(b, Bin::And, active, in_bounds);
                    }
                    let at = self.begin_guarded(Some(active));
                    let cb = self.cur;
                    let partner_lane = self.s.m.bin(cb, Bin::Add, console_lane, ks);
                    let pa = self.exchange_byte(cb, vwave, console_lane);
                    let pb = self.exchange_byte(cb, vwave, partner_lane);
                    let va = self.lds_load_dword(pa);
                    let vb = self.lds_load_dword(pb);
                    let comb = self.apply_wave_op(op, va, vb);
                    let pw = self.exchange_byte(cb, vwave, console_lane);
                    self.lds_store_dword(pw, comb);
                    self.end_guarded(at, &[]);
                }
                self.lds_barrier(self.cur);
                let k0 = self.s.m.i32(0);
                let rp = self.exchange_byte(self.cur, vwave, k0);
                return Ok(self.lds_load_dword(rp));
            }
        }

        self.s.flags |= flags::WAVE_OPS;
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let arg = if float { self.as_float(x) } else { x };
        let ty = if float { f32t } else { i32t };
        let sfx = if float { "f32" } else { "i32" };
        let r = match (op, prefix) {
            (Op::AddU | Op::AddF, Some(inclusive)) => {
                let (k0, k1) = (self.s.m.i8(0), self.s.m.i8(1));
                let r = self.s.op(
                    c,
                    &format!("wavePrefixOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    121,
                    &[arg, k0, k1],
                );
                if self.emulating_wave64() {
                    if self.guard_depth > 0
                        || !self.uniform.get(self.cur_ir).copied().unwrap_or(false)
                    {
                        return Err("a wave prefix in divergent control flow");
                    }
                    let half_total = self.s.m.bin(c, Bin::Add, r, arg);
                    let half_total_bits = if float {
                        self.as_int(half_total)
                    } else {
                        half_total
                    };
                    let local = if inclusive { half_total } else { r };
                    let local_bits = if float { self.as_int(local) } else { local };
                    let (_, console_lane, vwave) = self.ensure_wave_ids();
                    let stride = self.ballot_stride;
                    let base = self.ballot_base;
                    self.lds_barrier(self.cur);
                    let k31 = self.s.m.i32(31);
                    let is31 = self.s.m.cmp(self.cur, pred::EQ, console_lane, k31);
                    let at = self.begin_guarded(Some(is31));
                    let ptr = self.scratch_byte(self.cur, base, stride, vwave, 0);
                    self.lds_store_dword(ptr, half_total_bits);
                    self.end_guarded(at, &[]);
                    self.lds_barrier(self.cur);
                    let ptr = self.scratch_byte(self.cur, base, stride, vwave, 0);
                    let h0_bits = self.lds_load_dword(ptr);
                    let combined = self.apply_wave_op(op, h0_bits, local_bits);
                    let k32 = self.s.m.i32(32);
                    let in_h0 = self.s.m.cmp(self.cur, pred::IULT, console_lane, k32);
                    return Ok(self.s.m.select(self.cur, in_h0, local_bits, combined));
                }
                if inclusive {
                    self.s.m.bin(c, Bin::Add, r, arg)
                } else {
                    r
                }
            }
            (_, Some(_)) => return Err("a wave prefix of an operation other than a sum"),
            (Op::Or | Op::And | Op::Xor, None) => {
                let k = self.s.m.i8(match op {
                    Op::And => 0,
                    Op::Or => 1,
                    _ => 2,
                });
                self.s
                    .op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[arg, k])
            }
            (_, None) => {
                let (code, unsigned) = match op {
                    Op::AddU => (0, 1),
                    Op::AddF => (0, 0),
                    Op::MinU => (2, 1),
                    Op::MaxU => (3, 1),
                    Op::MinI | Op::MinF => (2, 0),
                    _ => (3, 0),
                };
                let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
                self.s.op(
                    c,
                    &format!("waveActiveOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    119,
                    &[arg, k, u],
                )
            }
        };
        let local_bits = if float { self.as_int(r) } else { r };
        if self.emulating_wave64() {
            let uniform_here = self.uniform.get(self.cur_ir).copied().unwrap_or(false);
            let (h0, h1) = self.exchange_two_halves(local_bits, id, uniform_here)?;
            return Ok(self.apply_wave_op(op, h0, h1));
        }
        Ok(local_bits)
    }

    fn run(mut self) -> Result<Lowered, Stop> {
        if self.lds_overflow {
            return Err(unsupported(0, "LDS exceeding 32 KiB with wave64 emulation"));
        }
        let p = self.x.p;
        let order = crate::ir::reverse_postorder(&p.blocks);
        for (n, &b) in order.iter().enumerate() {
            self.rpo[b] = n;
        }
        for (n, &b) in order.iter().enumerate() {
            self.block[b] = Some(if n == 0 {
                self.s.entry
            } else {
                self.s.m.new_block()
            });
        }
        // Function-wide step budgets (`LOOP_LIMIT`; spin waits use `spin_limit`):
        // non-barrier, non-spin loops share `steps[0]`, loops executing an active
        // `s_barrier` share `steps[1]` so divergent earlier loops never desynchronise
        // a barrier, and spin-wait loops share `steps[2]`.
        self.spin = super::spin::spin_loops(p);
        let mut has_lane_loops = false;
        let mut has_spin_loops = false;
        for &h in &order {
            let back: Vec<usize> = p.blocks[h]
                .preds
                .iter()
                .copied()
                .filter(|&q| self.block[q].is_some() && self.rpo[h] <= self.rpo[q])
                .collect();
            if back.is_empty() {
                continue;
            }
            let mut in_loop = HashSet::from([h]);
            let mut stack = back;
            while let Some(u) = stack.pop() {
                if in_loop.insert(u) {
                    stack.extend(
                        p.blocks[u]
                            .preds
                            .iter()
                            .copied()
                            .filter(|&q| self.block[q].is_some()),
                    );
                }
            }
            if in_loop.into_iter().any(|u| self.has_active_barrier(u)) {
                self.barrier_loops.insert(h);
            } else if self.spin.contains(&h) {
                has_spin_loops = true;
            } else {
                has_lane_loops = true;
            }
        }
        for (kind, needed) in [
            (0, has_lane_loops),
            (1, !self.barrier_loops.is_empty()),
            (2, has_spin_loops),
        ] {
            if !needed {
                continue;
            }
            let mut phis = vec![None; p.blocks.len()];
            for &b in &order[1..] {
                let incoming_count = p.blocks[b]
                    .preds
                    .iter()
                    .filter(|&&q| self.block[q].is_some())
                    .count();
                if incoming_count >= 2 {
                    let blk = self.block[b].expect("reachable");
                    phis[b] = Some(self.s.m.phi(blk, self.s.t.i32));
                }
            }
            self.steps[kind] = Some(StepTrack {
                phis,
                incoming: (0..p.blocks.len()).map(|_| Vec::new()).collect(),
                cur_val: None,
            });
        }
        self.cur = self.s.entry;
        self.inputs()?;
        if self.emulating_wave64() {
            self.ensure_wave_ids();
            self.preemit_masks(None);
        }
        if let Some(mesh) = self.s.mesh {
            self.mesh_prologue(mesh).map_err(|w| unsupported(0, w))?;
        }
        self.block[order[0]] = Some(self.cur);
        for &b in &order {
            self.cur = self.block[b].expect("reachable");
            self.cur_ir = b;
            let zero_step = (b == order[0]).then(|| self.s.m.i32(0));
            for track in self.steps.iter_mut().flatten() {
                track.cur_val = Some(if let Some(z) = zero_step {
                    z
                } else if let Some(phi) = track.phis[b] {
                    phi
                } else {
                    track.incoming[b][0].0
                });
            }
            let preds: Vec<usize> = p.blocks[b]
                .preds
                .iter()
                .copied()
                .filter(|&q| self.block[q].is_some())
                .collect();
            self.exported[b] = !preds.is_empty() && preds.iter().all(|&q| self.exported_out(q));
            for (i, phi) in p.blocks[b].phis.iter().enumerate() {
                let val = match phi.reg {
                    Reg::S(c) if c == EXEC || c == EXEC + 1 => Val::Missing("a lane mask"),
                    _ if self.bad.contains_key(&phi.value) => Val::Missing(self.bad[&phi.value]),
                    Reg::Scc => {
                        let v = self.s.m.phi(self.cur, self.s.t.i1);
                        self.data_phis.push((v, b, i));
                        Val::Bit(v)
                    }
                    _ => {
                        let v = self.s.m.phi(self.cur, self.s.t.i32);
                        self.data_phis.push((v, b, i));
                        Val::I32(v)
                    }
                };
                self.vals.insert(phi.value, val);
            }
            let mut i = 0;
            while i < p.blocks[b].insts.len() {
                if self.lds_barrier_before.contains(&(b, i)) {
                    self.emit_lds_barrier();
                }
                self.inst(b, i)?;
                if let Some(end) = self.sections.get(&b).and_then(|s| s.get(&i)).map(|s| s.end) {
                    for j in i + 1..end {
                        self.section_inst(b, j, i);
                    }
                    i = end;
                    continue;
                }
                i += 1;
            }
            if self.lds_barrier_end.contains(&b) {
                self.emit_lds_barrier();
            }
            self.mask_ok = self.mask_data == MaskData::Console;
            if self.emulating_wave64() {
                self.preemit_masks(Some(b));
            }
            self.tail[b] = Some(self.cur);
            self.exit(b)?;
        }
        self.fill_phis()?;
        if self.s.pixel {
            super::harmonize_row_interp(&mut self.s.inputs);
        }
        let user_data = self.user_reads.values().copied().collect();
        self.s.samplers = self.samplers.len() as u32;
        let root_cbvs = self.s.root_cbvs.iter().map(|r| r.0).collect();
        let root_srvs = self.s.root_srvs.iter().map(|r| r.0).collect();
        let stride_table = self.s.stride_table.is_some();
        let root_uavs = self.s.root_uavs.iter().map(|r| r.0).collect();
        let cmp_samplers = self.s.cmp_samplers;
        #[cfg(test)]
        let op_names = self.s.m.func_names().map(str::to_owned).collect();
        Ok(Lowered {
            cmp_samplers,
            root_cbvs,
            root_srvs,
            stride_table,
            root_uavs,
            #[cfg(test)]
            nonuniform_handles: self.s.nonuniform_handles,
            #[cfg(test)]
            wave_ops: self.s.flags & flags::WAVE_OPS != 0,
            #[cfg(test)]
            spin_loops: self.spun,
            dxil: self.s.finish(),
            user_data,
            samplers: self.samplers,
            #[cfg(test)]
            op_names,
        })
    }

    fn exported_out(&self, b: usize) -> bool {
        self.exported[b] || self.has_done_export(b)
    }

    fn has_done_export(&self, b: usize) -> bool {
        (0..self.x.p.blocks[b].insts.len())
            .any(|i| self.exports.get(&(b, i)).is_some_and(|e| e.done))
    }

    /// Values of the program's input registers.
    fn inputs(&mut self) -> Result<(), Stop> {
        let io_inputs: HashMap<Value, Input> = self.x.io.inputs.iter().copied().collect();
        for &(reg, v) in &self.x.p.inputs {
            let val = match reg {
                Reg::S(_) if self.params.contains_key(&v) => continue, // loaded on first use
                // Most programs only interpolate with them: loaded on first
                // use as a value (`get`).
                Reg::V(_) if matches!(io_inputs.get(&v), Some(Input::Barycentric { .. })) => {
                    continue;
                }
                Reg::S(n) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Above the SGPRs the hardware initializes, a register
                    // holds what the last wave left, so no program depends
                    // on it: a read the paths only seem to allow (a branch
                    // past the write, taken only with one past the read)
                    // reads zero, as VGPRs do.
                    None if n <= 105 && self.x.io.initialized_sgprs.is_some_and(|k| n >= k) => {
                        Val::I32(self.s.m.i32(0))
                    }
                    None => Val::Missing("a hardware SGPR input"),
                },
                Reg::Scc => Val::Missing("SCC on entry"),
                Reg::V(_) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Registers no one set read as zero (the validator
                    // refuses reads of undef).
                    None => Val::I32(self.s.m.i32(0)),
                },
            };
            self.vals.insert(v, val);
        }
        Ok(())
    }

    fn input(&mut self, input: Input) -> Val {
        let (kind, col) = match input {
            Input::VertexId => (1, 0),
            Input::InstanceId => (2, 0),
            Input::Position(c) => (3, c),
            Input::Attribute { row, channel } => {
                return match self.find_input(row, channel) {
                    Some((id, col)) => self.load_input(id, col),
                    None => Val::Missing("a vertex input the signature lacks"),
                };
            }
            Input::Barycentric { kind, index } => return self.barycentric(kind, index),
            Input::PrimitiveId => return Val::Missing("the primitive id"),
            // The console code tests it against 0 (`v_cmp_ne_u32`), as the
            // host's value can be.
            Input::FrontFace => (13, 0),
            Input::ThreadIdInGroup(c) => return self.thread_id(c),
            Input::GroupId(c) => return self.compute_id("groupId.i32", 94, c),
            Input::GroupInfo => return Val::Missing("the thread-group info SGPR"),
            Input::MeshGroup => {
                let Val::I32(g) = self.compute_id("groupId.i32", 94, 0) else {
                    unreachable!()
                };
                let n = self.s.m.i32(self.x.mesh_stride);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Mul, g, n));
            }
            Input::MergedWaveInfo => {
                // shortcut: only the wave index (bits 24-27) is set; the
                // console's other bits (lane counts) are read by no program.
                let f = self.mesh_thread();
                let shift = self.s.m.i32(self.wave().trailing_zeros());
                let w = self.s.m.bin(self.s.entry, Bin::LShr, f, shift);
                let k24 = self.s.m.i32(24);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Shl, w, k24));
            }
            _ => return Val::Missing("a pixel system input"),
        };
        let free = self
            .s
            .inputs
            .iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0);
        let id = match self.s.inputs.iter().position(|e| e.kind == kind) {
            Some(id) => {
                if kind != 3 {
                    self.s.inputs[id].row.get_or_insert(free);
                }
                id
            }
            None if kind >= 3 => return Val::Missing("a pixel system input the signature lacks"),
            None => {
                // The console signature leaves out vertex and instance ids; PC
                // gives them a row after the others.
                let row = self
                    .s
                    .inputs
                    .iter()
                    .filter_map(|e| e.row)
                    .map(|r| r + 1)
                    .max()
                    .unwrap_or(0);
                self.s.inputs.push(SigElement {
                    name: ["", "SV_VertexID", "SV_InstanceID"][kind as usize].into(),
                    index: 0,
                    kind,
                    comp: 5,
                    interp: 0,
                    row: Some(row),
                    col: 0,
                    cols: 1,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        self.load_input(id, col)
    }

    /// A component of a compute thread or group id, read in the entry block.
    fn compute_id(&mut self, name: &str, opcode: u32, c: u8) -> Val {
        let (entry, c) = (self.s.entry, self.s.m.i32(u32::from(c)));
        Val::I32(
            self.s
                .op(entry, name, Attr::ReadNone, self.s.t.i32, opcode, &[c]),
        )
    }

    fn find_input(&self, row: u8, channel: u8) -> Option<(usize, u8)> {
        self.s.inputs.iter().enumerate().find_map(|(id, e)| {
            let r = e.row?;
            // `then`, not `then_some`: the channel is only past `col` when
            // the condition holds.
            (r == u32::from(row) && channel >= e.col && channel < e.col + e.cols)
                .then(|| (id, channel - e.col))
        })
    }

    fn load_input(&mut self, id: usize, col: u8) -> Val {
        if self.s.inputs[id].half {
            // shortcut: 16-bit vertex inputs (the fetch routine's) are not
            // unpacked; pixel inputs are (`interp`).
            return Val::Missing("a 16-bit input");
        }
        self.load_any_input(id, col)
    }

    /// An input element's channel as declared (16-bit ones as 32-bit floats).
    fn load_any_input(&mut self, id: usize, col: u8) -> Val {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let (int, bit) = (e.comp != 9, e.comp == 1);
        let ty = if bit {
            self.s.t.i1
        } else if int {
            self.s.t.i32
        } else {
            self.s.t.f32
        };
        let suffix = if bit {
            ".i1"
        } else if int {
            ".i32"
        } else {
            ".f32"
        };
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.undef(self.s.t.i32),
        ];
        let v = self.s.op(
            entry,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        );
        if bit {
            let z = self.s.m.cast(entry, Cast::ZExt, v, self.s.t.i32);
            Val::I32(z)
        } else if int {
            Val::I32(v)
        } else {
            Val::I32(self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.i32))
        }
    }

    /// The DXIL form of an IR value.
    fn get(&mut self, v: Value) -> Result<Val, Why> {
        if self.phi_at.contains_key(&v) {
            self.read_phis.insert(v);
        }
        let is_mask = self.x.st.masks.contains_key(&v) || self.x.st.highs.contains_key(&v);
        match self.vals.get(&v) {
            Some(&Val::Missing(_)) | None if self.mask_ok && is_mask => {
                return self.mask_dword(v).map(Val::I32);
            }
            Some(&Val::Missing(w)) => return Err(w),
            Some(&val) => return Ok(val),
            None => {}
        }
        if let Some(&(param, dword)) = self.params.get(&v) {
            let val = self.user(param, dword);
            self.vals.insert(v, val);
            return Ok(val);
        }
        if let Some(loaded) = self.lazy_barycentric(v) {
            loaded?;
            return Ok(self.vals[&v]);
        }
        Err("a value defined after its use")
    }

    /// Loads `v` if it is a barycentric input VGPR not loaded yet (see
    /// `inputs`): `None` if it is not one.
    fn lazy_barycentric(&mut self, v: Value) -> Option<Result<(), Why>> {
        let (kind, index) = self.x.io.inputs.iter().find_map(|&(w, i)| match i {
            Input::Barycentric { kind, index } if w == v => Some((kind, index)),
            _ => None,
        })?;
        let val = self.barycentric(kind, index);
        self.vals.insert(v, val);
        Some(match val {
            Val::Missing(w) => Err(w),
            _ => Ok(()),
        })
    }

    /// Barycentric `index` (0 i, 1 j) of `kind` as a value: a channel of
    /// `SV_Barycentrics` declared with the same interpolation mode.
    ///
    /// GFX10 interpolates P0 + i (P1 - P0) + j (P2 - P0), so i and j are
    /// the weights of vertices 1 and 2. The host's weights (x, y, z) are
    /// those of vertices 0, 1 and 2, vertex 0 being the provoking vertex,
    /// which is the console's P0 (`v_interp_mov` P0 is lowered to a flat
    /// input): i is y and j is z.
    /// shortcut: triangles only; on lines and points the console's i and j
    /// are not checked against the host's weights.
    fn barycentric(&mut self, kind: Barycentric, index: u8) -> Val {
        let Some(mode) = interp_mode(kind) else {
            return Val::Missing("pull-model barycentrics");
        };
        if index > 1 {
            return Val::Missing("pull-model barycentrics");
        }
        // One element per interpolation mode (the console's own, if its
        // signature declares one): DXIL allows two.
        let bary = |e: &SigElement| e.kind == 28;
        let id = match self
            .s
            .inputs
            .iter()
            .position(|e| bary(e) && (e.interp == mode || e.usage == 0))
        {
            Some(id) => id,
            None => {
                let index = self.s.inputs.iter().filter(|e| bary(e)).count() as u32;
                if index >= 2 {
                    return Val::Missing("barycentrics of a third interpolation mode");
                }
                // A system value the validator wants unpacked (packing
                // location -1), like other non-packed system values.
                self.s.inputs.push(SigElement {
                    name: "SV_Barycentrics".into(),
                    index,
                    kind: 28,
                    comp: 9,
                    interp: mode,
                    row: None,
                    col: 0,
                    cols: 3,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        let e = &mut self.s.inputs[id];
        if e.col != 0 || e.cols < 3 {
            return Val::Missing("a console barycentrics element without i and j");
        }
        e.row = None;
        e.interp = mode;
        self.s.flags |= flags::BARYCENTRICS;
        self.load_any_input(id, index + 1)
    }

    fn i32_of(&mut self, v: Value) -> Result<V, Why> {
        match self.get(v)? {
            Val::I32(x) => Ok(x),
            Val::Bit(x) => Ok(self.s.m.cast(self.cur, Cast::ZExt, x, self.s.t.i32)),
            Val::Missing(w) => Err(w),
        }
    }

    /// A user-data dword: a root parameter's value.
    fn user(&mut self, param: usize, dword: u32) -> Val {
        let rp = self.x.native.layout[param];
        match rp.root {
            RootType::ResourceHeap | RootType::SamplerHeap => return Val::I32(self.s.m.i32(0)),
            RootType::Constants | RootType::Table => {}
            _ => return Val::Missing("a root descriptor"),
        }
        let index = u32::from(rp.user_data) + dword;
        self.user_reads.insert(
            index,
            UserData {
                index,
                slot: rp.slot,
                root: rp.root,
                dword,
            },
        );
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        Val::I32(self.s.m.extract(entry, r, index % 4))
    }

    /// A dword of a lane mask as a number: the ballot of the lanes' bits,
    /// taken in the block that writes the mask (see [`MaskData`]).
    fn mask_dword(&mut self, v: Value) -> Result<V, Why> {
        let (lo, high) = match self.x.st.highs.get(&v) {
            Some(&lo) => (lo, true),
            None => (v, false),
        };
        if let Some(&why) = self.ballot_fail.get(&lo) {
            return Err(why);
        }
        let (x, y) = match self.ballots.get(&lo) {
            Some(&xy) => xy,
            None => {
                if self.emulating_wave64() {
                    match self.def_block.get(&lo).copied() {
                        Some(b) if b == self.cur_ir => {
                            let uniform_here = self.uniform.get(b).copied().unwrap_or(false);
                            self.compute_ballot_in_cur(lo, uniform_here)?
                        }
                        Some(b) if !self.uniform.get(b).copied().unwrap_or(false) => {
                            return Err("a 64-lane wave ballot in divergent control flow");
                        }
                        None => return Err("a 64-lane wave ballot outside uniform entry"),
                        _ => return Err("a lane mask written after its use"),
                    }
                } else {
                    let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
                    if !self.x.st.active.contains(&lo) {
                        return Err("a lane mask holding lanes elsewhere read as data");
                    }
                    let at = match self.def_block.get(&lo) {
                        // Program inputs (EXEC on entry).
                        None => self.s.entry,
                        Some(&b) if b == self.cur_ir => self.cur,
                        Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
                    };
                    if !self.balloting.insert(lo) {
                        return Err("a lane mask whose lanes depend on its own value");
                    }
                    let bit = self.boolean(m, at);
                    self.balloting.remove(&lo);
                    let bit = bit?;
                    let i32t = self.s.t.i32;
                    let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                    let r = self
                        .s
                        .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
                    self.s.flags |= flags::WAVE_OPS;
                    let hi = if self.wave64 == Wave64Mode::Wave32 && self.total_threads <= 32 {
                        self.s.m.i32(0)
                    } else {
                        self.s.m.extract(at, r, 1)
                    };
                    let xy = (self.s.m.extract(at, r, 0), hi);
                    self.ballots.insert(lo, xy);
                    xy
                }
            }
        };
        Ok(if high {
            y
        } else if self.mask_data == MaskData::Host && !self.x.native.wave64 {
            // A wave32 program's host wave may still have 64 lanes, whose
            // upper half is a second console wave: its bits are the ballot's
            // second dword.
            let lane = self.lane_index();
            let c = self.cur;
            let k32 = self.s.m.i32(32);
            let upper = self.s.m.cmp(c, pred::IUGE, lane, k32);
            self.s.flags |= flags::WAVE_OPS;
            self.s.m.select(c, upper, y, x)
        } else {
            x
        })
    }

    // ---- operands ----

    fn operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Ok(self.s.m.i32(0)),
            Operand::Scalar { code, .. } => {
                let v = inst.read(Reg::S(code)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Vector { reg, .. } => {
                let v = inst.read(Reg::V(reg)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Const(c) => crate::resource::inline(c)
                .map(|x| self.s.m.i32(x))
                .ok_or("a special operand (vccz, execz, scc, lds_direct)"),
            Operand::Literal(x) => Ok(self.s.m.i32(x)),
            Operand::Attr { .. } => Err("an attribute operand"),
        }
    }

    fn as_float(&mut self, x: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, x, self.s.t.f32)
    }

    fn as_int(&mut self, f: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, f, self.s.t.i32)
    }

    /// The SDWA operand selection of source `k`: a byte or word of the
    /// dword, zero- or sign-extended (none: the whole dword).
    fn sdwa_src(inst: &Inst, k: usize) -> Option<(u8, bool)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        let sel = *s.src_sel.get(k)?;
        let sext = s.sext >> k & 1 != 0;
        (sel < 6).then_some((sel, sext))
    }

    /// Bits `sel` (0-3 a byte, 4-5 a word) of `x`, extended.
    fn select_bits(&mut self, x: V, sel: u8, sext: bool) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8, 8 * sel)
        } else {
            (16, 16 * (sel - 4))
        };
        let left = self.s.m.i32(u32::from(32 - width - at));
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - u32::from(width));
        let op = if sext { Bin::AShr } else { Bin::LShr };
        self.s.m.bin(c, op, t, right)
    }

    /// A float source with its abs/neg modifiers.
    fn fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let selected = Self::sdwa_src(inst, k);
        let mut f = match op {
            _ if selected.is_some() => {
                let (sel, sext) = selected.expect("checked");
                let x = self.operand(inst, &op)?;
                let x = self.select_bits(x, sel, sext);
                self.as_float(x)
            }
            _ if k == 0 && self.src0.is_some() => {
                let x = self.src0.expect("checked");
                self.as_float(x)
            }
            Operand::Vector { reg, .. } | Operand::Scalar { code: reg, .. }
                if !matches!(op, Operand::Scalar { code: NULL, .. }) =>
            {
                let r = if matches!(op, Operand::Vector { .. }) {
                    Reg::V(reg)
                } else {
                    Reg::S(reg)
                };
                let val = inst.read(r).ok_or("an unread operand")?;
                match self.floats.get(&val) {
                    Some(&f) => f,
                    None => {
                        let x = self.i32_of(val)?;
                        self.as_float(x)
                    }
                }
            }
            _ => {
                let x = self.operand(inst, &op)?;
                self.as_float(x)
            }
        };
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    fn src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 {
            return Err("integer operand modifiers");
        }
        match self.src0 {
            Some(x) if k == 0 => Ok(x),
            _ => {
                let x = self.operand(inst, &op)?;
                Ok(match Self::sdwa_src(inst, k) {
                    Some((sel, sext)) => self.select_bits(x, sel, sext),
                    None => x,
                })
            }
        }
    }

    fn unary(&mut self, op: u32, f: V) -> V {
        self.s.op(
            self.cur,
            "unary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[f],
        )
    }

    fn binary(&mut self, op: u32, a: V, b: V) -> V {
        self.s.op(
            self.cur,
            "binary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[a, b],
        )
    }

    /// The hardware's maximum (`max`) or minimum of two floats: as DXIL's
    /// FMax/FMin, but -0 orders below +0 (RDNA2 ISA, `v_max_f32`), where
    /// the compare and select DXIL specifies gives either zero.
    fn max_min(&mut self, max: bool, a: V, b: V) -> V {
        let c = self.cur;
        let r = self.binary(if max { 35 } else { 36 }, a, b);
        let (ia, ib) = (self.as_int(a), self.as_int(b));
        let either = self.s.m.bin(c, Bin::Or, ia, ib);
        let magnitude = self.s.m.i32(0x7FFF_FFFF);
        let either = self.s.m.bin(c, Bin::And, either, magnitude);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, pred::EQ, either, zero);
        // Of two zeros, +0 unless both are -0 (max), -0 unless neither (min).
        let op = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, op, ia, ib);
        let signed = self.as_float(sign);
        self.s.m.select(c, zeros, signed, r)
    }

    fn fmad(&mut self, a: V, b: V, c: V) -> V {
        self.s.op(
            self.cur,
            "tertiary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            46,
            &[a, b, c],
        )
    }

    /// `f` as an f16 in the low bits, rounded toward zero: the nearest-even
    /// conversion, one step back toward zero where it rounded away (which
    /// also turns an overflow to infinity into the largest finite half).
    fn f16_toward_zero(&mut self, f: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let h = self
            .s
            .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
        let back = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
        let (fa, ba) = (self.unary(6, f), self.unary(6, back));
        // Ordered: never for NaN.
        let away = self.s.m.cmp(c, pred::OGT, ba, fa);
        let one = self.s.m.i32(1);
        let stepped = self.s.m.bin(c, Bin::Sub, h, one);
        self.s.m.select(c, away, stepped, h)
    }

    /// `(a >> off[4:0]) & ((1 << width) - 1)`, sign-extended from the field's
    /// top bit when `signed`; widths from 32 on take every bit. A signed
    /// field that runs past bit 31 extends from bit 31, as the hardware
    /// does: the shift is arithmetic then.
    fn bfe(&mut self, a: V, off: V, width: V, signed: bool) -> V {
        let c = self.cur;
        let (k31, k32, zero, one, ones) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(0),
            self.s.m.i32(1),
            self.s.m.i32(u32::MAX),
        );
        let off = self.s.m.bin(c, Bin::And, off, k31);
        let shift = if signed { Bin::AShr } else { Bin::LShr };
        let shifted = self.s.m.bin(c, shift, a, off);
        let full = self.s.m.cmp(c, pred::IUGE, width, k32);
        let w = self.s.m.bin(c, Bin::And, width, k31);
        let mask = self.s.m.bin(c, Bin::Shl, one, w);
        let mask = self.s.m.bin(c, Bin::Sub, mask, one);
        let mask = self.s.m.select(c, full, ones, mask);
        let r = self.s.m.bin(c, Bin::And, shifted, mask);
        if !signed {
            return r;
        }
        // Sign-extend a field narrower than 32 bits (none for width 0).
        let left = self.s.m.bin(c, Bin::Sub, k32, w);
        let left = self.s.m.bin(c, Bin::And, left, k31);
        let t = self.s.m.bin(c, Bin::Shl, r, left);
        let t = self.s.m.bin(c, Bin::AShr, t, left);
        let is0 = self.s.m.cmp(c, pred::EQ, w, zero);
        let keep = self.s.m.bin(c, Bin::Or, full, is0);
        self.s.m.select(c, keep, r, t)
    }

    /// The median of three. With a NaN among them the hardware gives the
    /// minimum of the three instead (the host's minimum skips NaNs, so that
    /// is the smallest number among them).
    fn med3(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let lo = self.binary(36, a, b);
        let hi = self.binary(35, a, b);
        let t = self.binary(36, hi, third);
        let median = self.binary(35, lo, t);
        let least = self.binary(36, lo, third);
        let ab = self.s.m.cmp(c, pred::UNO, a, b);
        let nan3 = self.s.m.cmp(c, pred::UNO, third, third);
        let nan = self.s.m.bin(c, Bin::Or, ab, nan3);
        self.s.m.select(c, nan, least, median)
    }

    /// `a - floor(a)`, which on the hardware never reaches 1.0: where it
    /// would round to it (a tiny negative `a`), the result is `below_one`,
    /// the largest value under 1.0 in the operation's width. Hosts differ
    /// here (some give 1.0). Comparing against a `threshold` distinct from
    /// `below_one` prevents drivers (observed on NVIDIA) from folding
    /// `r > k ? k : r` into `minNum` (`FMNMX`) and swallowing NaN.
    fn fract(&mut self, a: V, threshold: f32, below_one: f32) -> V {
        let c = self.cur;
        let r = self.unary(22, a);
        let limit = self.s.m.f32(threshold.to_bits());
        let k = self.s.m.f32(below_one.to_bits());
        let over = self.s.m.cmp(c, pred::OGE, r, limit);
        self.s.m.select(c, over, k, r)
    }

    /// Sine, or cosine, of an angle in turns, as the hardware takes it. The
    /// whole turns are taken off first (`x - round_even(x)`, exact, within
    /// half a turn), so a large angle keeps its fraction; scaling to radians
    /// first rounds it away. A zero passes unreduced (the sine of -0 is -0).
    /// The sine of a half turn and the cosine of a quarter turn are exactly
    /// +0 on the hardware, where the host's function of the rounded angle
    /// gives about 1e-7. Infinities and NaN give NaN.
    /// shortcut: below 2^-12 turns the hardware scales by a slightly
    /// different constant than 2π (a last-place difference).
    fn trig_turns(&mut self, x: V, cosine: bool) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let whole = self.unary(26, x);
        let part = self.s.m.bin(c, Bin::Sub, x, whole);
        let is_zero = self.s.m.cmp(c, pred::OEQ, x, zero);
        let part = self.s.m.select(c, is_zero, x, part);
        let k = self.s.m.f32(TWO_PI.to_bits());
        let angle = self.s.m.bin(c, Bin::Mul, part, k);
        let r = self.unary(if cosine { 12 } else { 13 }, angle);
        let at = if cosine { 0.25f32 } else { 0.5 };
        let at = self.s.m.f32(at.to_bits());
        let size = self.unary(6, part);
        let exact = self.s.m.cmp(c, pred::OEQ, size, at);
        self.s.m.select(c, exact, zero, r)
    }

    /// The high dword of a 32 × 32-bit product.
    fn mul_hi(&mut self, a: V, b: V, signed: bool) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let ext = if signed { Cast::SExt } else { Cast::ZExt };
        let a = self.s.m.cast(c, ext, a, i64t);
        let b = self.s.m.cast(c, ext, b, i64t);
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::LShr, m, k32);
        self.s.flags |= flags::INT64;
        self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32)
    }

    /// `0 × anything = 0`, as in D3D9.
    fn legacy_mul(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let zero = self.s.m.f32(0);
        let az = self.s.m.cmp(c, pred::OEQ, a, zero);
        let bz = self.s.m.cmp(c, pred::OEQ, b, zero);
        let z = self.s.m.bin(c, Bin::Or, az, bz);
        self.s.m.select(c, z, zero, m)
    }

    // ---- lane booleans ----

    fn boolean(&mut self, b: Bool, at: usize) -> Result<V, Why> {
        if b == Bool::TRUE || b == Bool::FALSE {
            return Ok(self.s.m.bool(b == Bool::TRUE));
        }
        if let Some(&v) = self.bools.get(&(at, b)) {
            return Ok(v);
        }
        let (atom, lo, hi) = self.x.st.bools.node(b).expect("not a constant");
        let a = self.atom(atom, at)?;
        let m = &mut self.s.m;
        let v = match (lo, hi) {
            (Bool::FALSE, Bool::TRUE) => a,
            (Bool::TRUE, Bool::FALSE) => {
                let t = m.bool(true);
                m.bin(at, Bin::Xor, a, t)
            }
            (Bool::FALSE, h) => {
                let h = self.boolean(h, at)?;
                self.s.m.bin(at, Bin::And, a, h)
            }
            (l, Bool::TRUE) => {
                let l = self.boolean(l, at)?;
                self.s.m.bin(at, Bin::Or, a, l)
            }
            (l, h) => {
                let l = self.boolean(l, at)?;
                let h = self.boolean(h, at)?;
                self.s.m.select(at, a, h, l)
            }
        };
        self.bools.insert((at, b), v);
        Ok(v)
    }

    fn atom(&mut self, atom: Atom, at: usize) -> Result<V, Why> {
        match atom {
            Atom::Lane(v) => self.lane_bits.get(&v).copied().ok_or_else(|| {
                self.lane_fail
                    .get(&v)
                    .copied()
                    .unwrap_or("a lane mask from a carry")
            }),
            Atom::Uniform(v) => match self.get(v)? {
                Val::Bit(x) => Ok(x),
                Val::I32(x) => {
                    let z = self.s.m.i32(0);
                    Ok(self.s.m.cmp(at, pred::NE, x, z))
                }
                Val::Missing(w) => Err(w),
            },
            Atom::Phi(id) => {
                if let Some(&v) = self.mask_phis.get(&id) {
                    return Ok(v);
                }
                let &(b, i) = self
                    .phi_at
                    .get(&id)
                    .ok_or("a lane-mask join without a phi")?;
                let blk = self.block[b].ok_or("an unreachable join")?;
                let v = self.s.m.phi(blk, self.s.t.i1);
                self.mask_phis.insert(id, v);
                self.pending_masks.push((v, b, i));
                Ok(v)
            }
            Atom::Quad(lo, a) => self.quad_any(lo, a),
            // With console lanes, a mask held as a number is the console's:
            // the lane tests its own bit.
            Atom::Opaque(v) if self.console_lanes => {
                let hi = match self.pair_high(v) {
                    Some(h) => Some(self.get(h)?),
                    None => None,
                };
                let lo = self.get(v)?;
                let word = |w: Val| match w {
                    Val::I32(x) => Ok(x),
                    _ => Err("an opaque lane mask"),
                };
                let lo = word(lo)?;
                let hi = hi.map(word).transpose()?;
                Ok(self.lane_bit(at, lo, hi))
            }
            Atom::Constant(k) if self.console_lanes => {
                let lo = self.s.m.i32(k as u32);
                let hi = self.s.m.i32((k >> 32) as u32);
                Ok(self.lane_bit(at, lo, Some(hi)))
            }
            Atom::Opaque(_) => Err("an opaque lane mask"),
            Atom::Constant(_) => Err("a constant lane mask"),
        }
    }

    /// `s_wqm` ("some lane of the quad holds `a`"), evaluated in the block
    /// that defines `lo` so whole quads participate before any later branch.
    fn quad_any(&mut self, lo: Value, a: Bool) -> Result<V, Why> {
        if let Some(&v) = self.quad_bits.get(&lo) {
            return Ok(v);
        }
        let at = match self.def_block.get(&lo) {
            None => self.s.entry,
            Some(&b) => match self.tail[b] {
                Some(tb) => tb,
                None if b == self.cur_ir => self.cur,
                None => return Err("a quad mask written after its use"),
            },
        };
        if !self.quad_visiting.insert(lo) {
            return Err("a quad mask whose lanes depend on its own value");
        }
        let bit = self.boolean(a, at);
        self.quad_visiting.remove(&lo);
        let bit = bit?;
        let i32t = self.s.t.i32;
        let z = self.s.m.i32(0);
        let q = if self.s.pixel
            || self
                .s
                .mesh
                .is_some_and(|_| self.s.threads.is_some_and(|t| t[0].is_multiple_of(4)))
        {
            self.s.flags |= flags::WAVE_OPS;
            let x = self.s.m.cast(at, Cast::ZExt, bit, i32t);
            let mut any = x;
            for dir in 0..3u8 {
                let op = self.s.m.i8(dir);
                let other = self
                    .s
                    .op(at, "quadOp.i32", Attr::NoUnwind, i32t, 123, &[x, op]);
                any = self.s.m.bin(at, Bin::Or, any, other);
            }
            self.s.m.cmp(at, pred::NE, any, z)
        } else if self.s.mesh.is_none() && self.s.threads.is_some() && self.console_lanes {
            self.s.flags |= flags::WAVE_OPS;
            let x = self.s.m.cast(at, Cast::ZExt, bit, i32t);
            let lane = self
                .s
                .op(at, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
            let not3 = self.s.m.i32(!3);
            let base = self.s.m.bin(at, Bin::And, lane, not3);
            let mut any = z;
            for k in 0..4u32 {
                let kk = self.s.m.i32(k);
                let src = self.s.m.bin(at, Bin::Or, base, kk);
                let other = self.s.op(
                    at,
                    "waveReadLaneAt.i32",
                    Attr::NoUnwind,
                    i32t,
                    117,
                    &[x, src],
                );
                any = self.s.m.bin(at, Bin::Or, any, other);
            }
            self.s.m.cmp(at, pred::NE, any, z)
        } else if self.s.mesh.is_some() {
            return Err("a cross-lane operation in a mesh shader");
        } else {
            return Err("a cross-lane operation outside compute");
        };
        self.quad_bits.insert(lo, q);
        Ok(q)
    }

    /// This lane's bit of the mask `lo`, `hi` (wave64).
    fn lane_bit(&mut self, at: usize, lo: V, hi: Option<V>) -> V {
        let lane = if self.s.threads.is_some() && self.emulating_wave64() {
            self.ensure_wave_ids().1
        } else {
            self.s.flags |= flags::WAVE_OPS;
            self.s.op(
                at,
                "waveGetLaneIndex",
                Attr::ReadOnly,
                self.s.t.i32,
                111,
                &[],
            )
        };
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let word = match hi {
            Some(hi) => {
                let low = self.s.m.cmp(at, pred::IULT, lane, k32);
                self.s.m.select(at, low, lo, hi)
            }
            None => lo,
        };
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    fn select_condition(&mut self, inst: &Inst, mask: Option<&Operand>) -> Result<V, Why> {
        if let Some(&m) = inst
            .writes
            .first()
            .and_then(|&(_, v)| self.x.st.select_masks.get(&v))
        {
            return self.boolean(m, self.cur);
        }
        let Some(&Operand::Scalar { code, .. }) = mask else {
            return Err(UNTRACKED_SELECT);
        };
        if code == NULL {
            return Ok(self.s.m.bool(false));
        }
        if !self.numeric_selects || !self.console_lanes {
            return Err(UNTRACKED_SELECT);
        }
        // A scalar operation can define either dword independently. The SSA
        // operands at this use give the actual pair, even across a join.
        let lo = self.i32_of(inst.read(Reg::S(code)).ok_or(UNTRACKED_SELECT)?)?;
        let hi = if self.x.p.wave64 {
            Some(self.i32_of(inst.read(Reg::S(code + 1)).ok_or(UNTRACKED_SELECT)?)?)
        } else {
            None
        };
        self.s.flags |= flags::WAVE_OPS;
        Ok(self.lane_bit(self.cur, lo, hi))
    }

    /// The value of the high dword of the wave64 register pair whose low
    /// dword holds `lo` (none in wave32).
    fn pair_high(&self, lo: Value) -> Option<Value> {
        if !self.x.native.wave64 {
            return None;
        }
        let p = self.x.p;
        let pick = |reg: Reg| match reg {
            Reg::S(c) => Some(Reg::S(c + 1)),
            _ => None,
        };
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == lo) {
            let h = pick(reg)?;
            return p.inputs.iter().find(|i| i.0 == h).map(|i| i.1);
        }
        if let Some(&(b, i)) = self.phi_at.get(&lo) {
            let h = pick(p.blocks[b].phis[i].reg)?;
            return p.blocks[b]
                .phis
                .iter()
                .find(|q| q.reg == h)
                .map(|q| q.value);
        }
        let b = *self.def_block.get(&lo)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == lo))?;
        let &(reg, _) = inst.writes.iter().find(|w| w.1 == lo)?;
        inst.write(pick(reg)?)
    }

    fn guard(&mut self, b: usize, i: usize) -> Result<Option<V>, Why> {
        match self.x.st.guards[b][i] {
            None => Ok(None),
            Some(g) => Ok(Some(self.boolean(g, self.cur)?)),
        }
    }

    // ---- results ----

    fn set(&mut self, v: Option<Value>, val: Val) {
        if let Some(v) = v {
            self.vals.insert(v, val);
        }
    }

    /// Writes VGPR `reg` (or the SGPR of a lane read), under `guard`.
    fn write(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let x = match (Self::sdwa_dst(inst), reg) {
            (Some((sel, unused)), Reg::V(_)) => self.place_bits(inst, reg, x, sel, unused),
            _ => x,
        };
        self.write_placed(inst, reg, x, guard);
    }

    /// [`Self::write`] of a dword already placed (SDWA applied).
    fn write_placed(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let Some(new) = inst.write(reg) else { return };
        let x = match (guard, reg) {
            (Some(g), Reg::V(_)) => {
                let old = match inst.read(reg).map(|o| self.i32_of(o)) {
                    Some(Ok(o)) => o,
                    // The validator refuses reads of undef.
                    _ => self.s.m.i32(0),
                };
                self.s.m.select(self.cur, g, x, old)
            }
            _ => x,
        };
        self.vals.insert(new, Val::I32(x));
    }

    /// The SDWA destination selection: a byte or word, and what the other
    /// bits get (0 zeros, 1 the sign above the field, 2 the old bits).
    fn sdwa_dst(inst: &Inst) -> Option<(u8, u8)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        (s.dst_sel < 6).then_some((s.dst_sel, s.dst_unused))
    }

    /// The low byte or word of `x` placed at `sel` in VGPR `reg`.
    fn place_bits(&mut self, inst: &Inst, reg: Reg, x: V, sel: u8, unused: u8) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8u32, 8 * u32::from(sel))
        } else {
            (16, 16 * u32::from(sel - 4))
        };
        let left = self.s.m.i32(32 - width);
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - width - at);
        if unused == 1 {
            // Bits below the field shift in as zeros.
            return self.s.m.bin(c, Bin::AShr, t, right);
        }
        let field = self.s.m.bin(c, Bin::LShr, t, right);
        if unused != 2 {
            return field;
        }
        let old = match inst.read(reg).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let keep = self.s.m.i32(!((((1u64 << width) - 1) as u32) << at));
        let old = self.s.m.bin(c, Bin::And, old, keep);
        self.s.m.bin(c, Bin::Or, old, field)
    }

    fn write_f(&mut self, inst: &Inst, f: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let mut f = f;
        match v.omod {
            0 => {}
            k => {
                let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(k)];
                let c = self.s.m.f32(factor.to_bits());
                f = self.s.m.bin(self.cur, Bin::Mul, f, c);
            }
        }
        if v.clamp {
            f = self.unary(7, f);
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a float result outside a VGPR");
        };
        let x = self.as_int(f);
        self.write(inst, Reg::V(reg), x, guard);
        if guard.is_none()
            && Self::sdwa_dst(inst).is_none()
            && let Some(new) = inst.write(Reg::V(reg))
        {
            self.floats.insert(new, f);
        }
        Ok(())
    }

    fn write_i(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 {
            return Err("integer output modifiers");
        }
        match v.dst {
            Some(Operand::Vector { reg, .. }) => self.write(inst, Reg::V(reg), x, guard),
            Some(Operand::Scalar { code, .. }) => self.write(inst, Reg::S(code), x, None),
            _ => return Err("a result without a register"),
        }
        Ok(())
    }

    /// Starts code only the lanes where `guard` holds run (a store under a
    /// lane guard): returns where it started and the join block.
    fn begin_guarded(&mut self, guard: Option<V>) -> Option<(usize, usize)> {
        let g = guard?;
        self.guard_depth += 1;
        let (then, join) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(self.cur, Term::CondBr(g, then, join));
        let head = self.cur;
        self.cur = then;
        Some((head, join))
    }

    /// Ends guarded code; each (value inside, value outside) becomes a phi.
    fn end_guarded(&mut self, at: Option<(usize, usize)>, values: &[(V, V)]) -> Vec<V> {
        let Some((head, join)) = at else {
            return values.iter().map(|v| v.0).collect();
        };
        self.guard_depth -= 1;
        let then = self.cur;
        self.s.m.terminate(then, Term::Br(join));
        self.cur = join;
        values
            .iter()
            .map(|&(inside, outside)| {
                let ty = self.s.m.type_of(inside);
                let phi = self.s.m.phi(join, ty);
                self.s.m.set_phi(phi, vec![(inside, then), (outside, head)]);
                phi
            })
            .collect()
    }

    /// Atomically updates one location. The first compare also obtains the
    /// initial value, so every observation is atomic, even when it is i64.
    fn atomic_update(&mut self, zero: V, exchange: impl FnOnce(&mut Self, V) -> V) -> V {
        let head = self.cur;
        let retry = self.s.m.new_block();
        let done = self.s.m.new_block();
        self.s.m.terminate(head, Term::Br(retry));
        self.cur = retry;
        let ty = self.s.m.type_of(zero);
        let expected = self.s.m.phi(retry, ty);
        let old = exchange(self, expected);
        let success = self.s.m.cmp(retry, pred::EQ, expected, old);
        self.s.m.set_phi(expected, vec![(zero, head), (old, retry)]);
        self.s.m.terminate(retry, Term::CondBr(success, done, retry));
        self.cur = done;
        old
    }

    /// AMD RDNA 2 ISA sections 12.13–12.14 (including DS_WRAP).
    fn atomic_new_value(&mut self, op: AtomicUpdate, old: V, data: V, data2: V) -> V {
        use AtomicUpdate::*;
        let c = self.cur;
        let ty = self.s.m.type_of(old);
        let zero = self.s.m.int_const(ty, 0);
        let one = self.s.m.int_const(ty, 1);
        match op {
            Inc => {
                let wrap = self.s.m.cmp(c, pred::IUGE, old, data);
                let next = self.s.m.bin(c, Bin::Add, old, one);
                self.s.m.select(c, wrap, zero, next)
            }
            Dec => {
                let at_zero = self.s.m.cmp(c, pred::EQ, old, zero);
                let above = self.s.m.cmp(c, pred::IUGT, old, data);
                let wrap = self.s.m.bin(c, Bin::Or, at_zero, above);
                let next = self.s.m.bin(c, Bin::Sub, old, one);
                self.s.m.select(c, wrap, data, next)
            }
            ReverseSub => self.s.m.bin(c, Bin::Sub, data, old),
            MaskOr => {
                let ones = self.s.m.int_const(ty, -1);
                let keep = self.s.m.bin(c, Bin::Xor, data, ones);
                let kept = self.s.m.bin(c, Bin::And, old, keep);
                self.s.m.bin(c, Bin::Or, kept, data2)
            }
            Wrap => {
                let wrap = self.s.m.cmp(c, pred::IUGE, old, data);
                let low = self.s.m.bin(c, Bin::Sub, old, data);
                let high = self.s.m.bin(c, Bin::Add, old, data2);
                self.s.m.select(c, wrap, low, high)
            }
            ClampSub => {
                let underflow = self.s.m.cmp(c, pred::IULT, old, data);
                let diff = self.s.m.bin(c, Bin::Sub, old, data);
                self.s.m.select(c, underflow, zero, diff)
            }
        }
    }

    fn byte_shift(&mut self, address: V) -> V {
        let three = self.s.m.i32(3);
        let byte = self.s.m.bin(self.cur, Bin::And, address, three);
        self.s.m.bin(self.cur, Bin::Shl, byte, three)
    }

    fn write_subword_result(
        &mut self,
        inst: &Inst,
        reg: u8,
        x: V,
        half: Option<u32>,
        guard: Option<V>,
    ) -> Result<(), Why> {
        let reg = Reg::V(reg);
        let x = if let Some(shift) = half {
            let old = self.i32_of(inst.read(reg).ok_or("a D16 load's previous register value")?)?;
            let c = self.cur;
            let (mask, keep, at) = (
                self.s.m.i32(0xFFFF),
                self.s.m.i32(!(0xFFFF << shift)),
                self.s.m.i32(shift),
            );
            let field = self.s.m.bin(c, Bin::And, x, mask);
            let field = self.s.m.bin(c, Bin::Shl, field, at);
            let old = self.s.m.bin(c, Bin::And, old, keep);
            self.s.m.bin(c, Bin::Or, old, field)
        } else {
            x
        };
        self.write(inst, reg, x, guard);
        Ok(())
    }

    /// Loads a byte field via aligned dword accesses. A half starting at
    /// byte 3 reads the first byte of the next dword.
    fn read_subword(
        &mut self,
        address: V,
        width: u32,
        signed: bool,
        dword: impl Fn(&mut Self, V) -> Result<V, Why>,
    ) -> Result<V, Why> {
        let c = self.cur;
        let shift = self.byte_shift(address);
        let not3 = self.s.m.i32(!3);
        let aligned = self.s.m.bin(c, Bin::And, address, not3);
        let w = dword(self, aligned)?;
        let low = self.s.m.bin(c, Bin::LShr, w, shift);
        let x = if width == 16 {
            let k24 = self.s.m.i32(24);
            let split = self.s.m.cmp(c, pred::EQ, shift, k24);
            let at = self.begin_guarded(Some(split));
            let four = self.s.m.i32(4);
            let next = self.s.m.bin(self.cur, Bin::Add, aligned, four);
            let high = dword(self, next)?;
            let eight = self.s.m.i32(8);
            let high = self.s.m.bin(self.cur, Bin::Shl, high, eight);
            let both = self.s.m.bin(self.cur, Bin::Or, low, high);
            self.end_guarded(at, &[(both, low)])[0]
        } else {
            low
        };
        let left = self.s.m.i32(32 - width);
        let x = self.s.m.bin(self.cur, Bin::Shl, x, left);
        Ok(self.s.m.bin(self.cur, if signed { Bin::AShr } else { Bin::LShr }, x, left))
    }

    /// Masks the affected bytes in each dword. A CAS keeps concurrent
    /// writes to other bytes in the same dword from being lost.
    fn write_subword(
        &mut self,
        address: V,
        width: u32,
        data: V,
        store: impl Fn(&mut Self, V, V, V),
    ) {
        let c = self.cur;
        let shift = self.byte_shift(address);
        let not3 = self.s.m.i32(!3);
        let aligned = self.s.m.bin(c, Bin::And, address, not3);
        let mask = self.s.m.i32((1 << width) - 1);
        let data = self.s.m.bin(c, Bin::And, data, mask);
        let bits = self.s.m.bin(c, Bin::Shl, data, shift);
        let mask = self.s.m.bin(c, Bin::Shl, mask, shift);
        store(self, aligned, mask, bits);
        if width == 16 {
            let c = self.cur;
            let k24 = self.s.m.i32(24);
            let split = self.s.m.cmp(c, pred::EQ, shift, k24);
            let at = self.begin_guarded(Some(split));
            let four = self.s.m.i32(4);
            let next = self.s.m.bin(self.cur, Bin::Add, aligned, four);
            let (eight, byte_mask) = (self.s.m.i32(8), self.s.m.i32(255));
            let bits = self.s.m.bin(self.cur, Bin::LShr, data, eight);
            store(self, next, byte_mask, bits);
            self.end_guarded(at, &[]);
        }
    }

    fn lds_ptr(&mut self, byte: V) -> V {
        let shift = self.s.m.i32(if self.wide_lds { 3 } else { 2 });
        let index = self.s.m.bin(self.cur, Bin::LShr, byte, shift);
        let lds = self.s.lds(self.wide_lds);
        self.s.m.element(self.cur, lds, index)
    }

    // DXIL forbids reinterpreting TGSM through a differently sized pointer.
    // When a shader needs i64 atomics, every access uses the same i64 array.
    fn lds_half_shift(&mut self, byte: V) -> V {
        let (four, eight) = (self.s.m.i32(4), self.s.m.i32(8));
        let half = self.s.m.bin(self.cur, Bin::And, byte, four);
        let shift = self.s.m.bin(self.cur, Bin::Mul, half, eight);
        let ty = self.s.m.int(64);
        self.s.m.cast(self.cur, Cast::ZExt, shift, ty)
    }

    fn lds_load_dword(&mut self, byte: V) -> V {
        if self.wide_lds {
            let end = self.s.m.i32(self.s.lds_dwords * 4);
            let in_bounds = self.s.m.cmp(self.cur, pred::IULT, byte, end);
            let at = self.begin_guarded(Some(in_bounds));
            let p = self.lds_ptr(byte);
            let ty = self.s.m.int(64);
            let zero = self.s.m.int_const(ty, 0);
            // Atomic access avoids racing with a lane using the other half.
            let pair = self.s.m.atomic(self.cur, Rmw::Or, p, zero);
            let shift = self.lds_half_shift(byte);
            let x = self.s.m.bin(self.cur, Bin::LShr, pair, shift);
            let x = self.s.m.cast(self.cur, Cast::Trunc, x, self.s.t.i32);
            let zero = self.s.m.i32(0);
            return self.end_guarded(at, &[(x, zero)])[0];
        }
        let p = self.lds_ptr(byte);
        self.s.m.load(self.cur, p)
    }

    fn lds_update_dword(&mut self, byte: V, update: impl FnOnce(&mut Self, V) -> V) -> V {
        let p = self.lds_ptr(byte);
        let zero = if self.wide_lds {
            let ty = self.s.m.int(64);
            self.s.m.int_const(ty, 0)
        } else {
            self.s.m.i32(0)
        };
        let shift = self.wide_lds.then(|| self.lds_half_shift(byte));
        let mut result = None;
        self.atomic_update(zero, |l, old| {
            let c = l.cur;
            let field = if let Some(shift) = shift {
                let field = l.s.m.bin(c, Bin::LShr, old, shift);
                l.s.m.cast(c, Cast::Trunc, field, l.s.t.i32)
            } else {
                old
            };
            result = Some(field);
            let next = update(l, field);
            let next = if let Some(shift) = shift {
                let ty = l.s.m.int(64);
                let next = l.s.m.cast(c, Cast::ZExt, next, ty);
                let next = l.s.m.bin(c, Bin::Shl, next, shift);
                let mask = l.s.m.int_const(ty, i64::from(u32::MAX));
                let mask = l.s.m.bin(c, Bin::Shl, mask, shift);
                l.atomic_new_value(AtomicUpdate::MaskOr, old, mask, next)
            } else {
                next
            };
            l.s.m.cmpxchg(c, p, old, next)
        });
        result.expect("the compare-exchange body was emitted")
    }

    fn lds_store_dword(&mut self, byte: V, data: V) {
        if self.wide_lds {
            self.lds_update_dword(byte, |_, _| data);
        } else {
            let p = self.lds_ptr(byte);
            self.s.m.store(self.cur, p, data);
        }
    }

    fn join64(&mut self, lo: V, hi: V) -> V {
        let c = self.cur;
        let ty = self.s.m.int(64);
        let lo = self.s.m.cast(c, Cast::ZExt, lo, ty);
        let hi = self.s.m.cast(c, Cast::ZExt, hi, ty);
        let k32 = self.s.m.int_const(ty, 32);
        let hi = self.s.m.bin(c, Bin::Shl, hi, k32);
        self.s.m.bin(c, Bin::Or, lo, hi)
    }

    fn atomic_data(&mut self, inst: &Inst, data: &Operand, wide: bool) -> Result<V, Why> {
        let words = self.vgprs(inst, data, if wide { 2 } else { 1 })?;
        Ok(if wide { self.join64(words[0], words[1]) } else { words[0] })
    }

    /// Code that depends on the lane layout: wave operations, and in compute
    /// the console's wave size (lanes are thread ids in group order there).
    /// Pixel lanes match only within a quad.
    fn cross_lane(&mut self, quad: bool) -> Result<(), Why> {
        if self.s.mesh.is_some() {
            // shortcut: no wave size for mesh shaders in SM 6.6, so host
            // lanes cannot follow the console's; emulate through group-shared
            // memory if mesh programs turn out to need these.
            return Err("a cross-lane operation in a mesh shader");
        }
        if self.s.threads.is_none() && !(quad && self.s.pixel) {
            return Err("a cross-lane operation outside compute");
        }
        debug_assert!(self.s.threads.is_none() || self.console_lanes || self.emulating_wave64());
        if self.wave64 != Wave64Mode::Emulate {
            self.s.flags |= flags::WAVE_OPS;
        }
        Ok(())
    }

    /// Group-shared memory barrier + thread-group sync (`GroupMemoryBarrierWithGroupSync`).
    fn emit_lds_barrier(&mut self) {
        let mode = self.s.m.i32(1 | 8);
        let void = self.s.t.void;
        self.s
            .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    /// Component `c` of the thread id in the group. With console lanes, the
    /// thread is relabelled so that its host lane is its console lane: the
    /// flat id is the wave's first lane's, rounded down to the wave size,
    /// plus the lane index. (A bijection while each host wave holds a
    /// wave-aligned run of flat ids, as the console's waves do.)
    fn thread_id(&mut self, c: u8) -> Val {
        if !self.console_lanes && !self.emulating_wave64() {
            return self.compute_id("threadIdInGroup.i32", 95, c);
        }
        let e = self.s.entry;
        let [x, y, _] = self.s.threads.expect("compute");
        let f = self.ensure_wave_ids().0;
        let (xk, yk, xy) = (self.s.m.i32(x), self.s.m.i32(y), self.s.m.i32(x * y));
        Val::I32(match c {
            0 => self.s.m.bin(e, Bin::URem, f, xk),
            1 => {
                let t = self.s.m.bin(e, Bin::UDiv, f, xk);
                self.s.m.bin(e, Bin::URem, t, yk)
            }
            _ => self.s.m.bin(e, Bin::UDiv, f, xy),
        })
    }

    /// Mesh: the counts and the primitives no lane exports. The console's
    /// counts come from one lane's allocation request, which DXIL cannot
    /// express (one call, dominating every output write).
    // shortcut: the counts are the declared maxima and each lane first culls
    // its primitive (all indices 0, zero area), so primitives past the
    // program's count are culled and vertices past it unused; this costs
    // the host the unused vertex and primitive slots. Pass the program's own
    // counts if that shows in frame time.
    fn mesh_prologue(&mut self, mesh: super::shader::Mesh) -> Result<(), Why> {
        let threads = self.s.threads.expect("mesh")[0];
        if mesh.topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads > 128 || mesh.max_vertices > 256 || mesh.max_primitives > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        let e = self.s.entry;
        let void = self.s.t.void;
        let (v, p) = (
            self.s.m.i32(mesh.max_vertices),
            self.s.m.i32(mesh.max_primitives),
        );
        self.s
            .op(e, "setMeshOutputCounts", Attr::NoUnwind, void, 168, &[v, p]);
        let flat = self.mesh_thread();
        let guard = (threads > mesh.max_primitives).then(|| self.s.m.cmp(e, pred::IULT, flat, p));
        let at = self.begin_guarded(guard);
        let z = self.s.m.i32(0);
        self.s.op(
            self.cur,
            "emitIndices",
            Attr::NoUnwind,
            void,
            169,
            &[flat, z, z, z],
        );
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: the thread's index in the group, which is its vertex and
    /// primitive index.
    fn mesh_thread(&mut self) -> V {
        if let Some(f) = self.flat {
            return f;
        }
        let e = self.s.entry;
        let f = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            self.s.t.i32,
            96,
            &[],
        );
        self.flat = Some(f);
        f
    }

    /// The console wave size.
    fn wave(&self) -> u32 {
        if self.x.native.wave64 { 64 } else { 32 }
    }

    fn host_lane_index(&mut self) -> V {
        self.s.op(
            self.cur,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        )
    }

    fn lane_index(&mut self) -> V {
        if self.s.mesh.is_some() {
            // The console's lane: mesh code reads no other lanes, so host
            // waves need not match the console's.
            let f = self.mesh_thread();
            let m = self.s.m.i32(self.wave() - 1);
            return self.s.m.bin(self.s.entry, Bin::And, f, m);
        }
        if self.emulating_wave64() {
            return self.ensure_wave_ids().1;
        }
        self.host_lane_index()
    }

    /// `x` of lane `src` (in this lane's quad) through quad reads, whose
    /// lane must be a constant.
    fn quad_pick(&mut self, x: V, src: V) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let three = self.s.m.i32(3);
        let q = self.s.m.bin(c, Bin::And, src, three);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self.s.op(
                c,
                "quadReadLaneAt.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                122,
                &[x, kn],
            );
            let is = self.s.m.cmp(c, pred::EQ, q, kn);
            r = self.s.m.select(c, is, read, r);
        }
        r
    }

    fn read_lane(&mut self, x: V, lane: V) -> V {
        self.s.op(
            self.cur,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            117,
            &[x, lane],
        )
    }

    /// The float value of a register value.
    fn float_of(&mut self, v: Value) -> Result<V, Why> {
        if let Some(&f) = self.floats.get(&v) {
            return Ok(f);
        }
        let x = self.i32_of(v)?;
        Ok(self.as_float(x))
    }

    /// The values of VGPRs `reg..reg + n` as `inst` reads them.
    fn vgprs(&mut self, inst: &Inst, op: &Operand, n: u8) -> Result<Vec<V>, Why> {
        let Operand::Vector { reg, .. } = *op else {
            return Err("a vector operand");
        };
        (0..n)
            .map(|k| {
                let v = inst.read(Reg::V(reg + k)).ok_or("an unread operand")?;
                self.i32_of(v)
            })
            .collect()
    }

    /// LDS (thread-group shared memory) and `ds_swizzle`.
    fn ds(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Ds {
            dst,
            addr,
            data,
            offset0,
            offset1,
            gds,
        } = &inst.fields
        else {
            unreachable!()
        };
        if *gds {
            return Err("GDS");
        }
        let name = inst.name();
        if name == "ds_nop" {
            return Ok(());
        }
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
        let dst_reg = match dst {
            Some(Operand::Vector { reg, .. }) => Some(*reg),
            _ => None,
        };
        if name == "ds_swizzle_b32" {
            // Bit 15 set: each lane of a quad picks a lane of it; else lanes
            // of each group of 32 combine their index with and/or/xor masks.
            let quad = offset & 0x8000 != 0
                || offset & 0x1C == 0x1C && offset & 0x380 == 0 && offset >> 10 & 0x1F < 4;
            // Mesh: host quads are threads 4n..4n+3 of a 1D group, as the
            // console's, so quad-local swizzles read through quad operations.
            let mesh_quad = self.s.mesh.is_some() && quad;
            if !mesh_quad {
                self.cross_lane(quad)?;
            }
            // The source sits in the address field.
            let x = self.vgprs(inst, addr.as_ref().ok_or("a swizzle source")?, 1)?[0];
            let lane = if self.wave64 == Wave64Mode::Emulate {
                self.lane_index()
            } else {
                self.host_lane_index()
            };
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let src = if offset & 0x8000 != 0 {
                let (three, one) = (k(self, 3), k(self, 1));
                let sel = self.s.m.bin(c, Bin::And, lane, three);
                let sh = self.s.m.bin(c, Bin::Shl, sel, one);
                let pattern = k(self, offset & 0xFF);
                let pick = self.s.m.bin(c, Bin::LShr, pattern, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, three);
                let not3 = k(self, !3);
                let base = self.s.m.bin(c, Bin::And, lane, not3);
                self.s.m.bin(c, Bin::Or, base, pick)
            } else {
                let and = k(self, offset & 0x1F | !0x1F);
                let or = k(self, offset >> 5 & 0x1F);
                let xor = k(self, offset >> 10 & 0x1F);
                let t = self.s.m.bin(c, Bin::And, lane, and);
                let t = self.s.m.bin(c, Bin::Or, t, or);
                self.s.m.bin(c, Bin::Xor, t, xor)
            };
            let r = if mesh_quad {
                self.quad_pick(x, src)
            } else if self.wave64 == Wave64Mode::Emulate {
                self.exchange_64(x, src)?
            } else {
                self.read_lane(x, src)
            };
            let reg = dst_reg.ok_or("a swizzle result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if self.title_lds_dwords == 0 {
            return Err("LDS the program does not allocate");
        }
        if name.contains("addtid") {
            return Err("an LDS address based on the console lane ID and M0");
        }
        if name.contains("_src2_") {
            return Err("an LDS operation with a second memory source");
        }
        if name.starts_with("ds_gws") || matches!(name, "ds_append" | "ds_consume" | "ds_ordered_count") {
            return Err("an LDS or GDS queue or global wave synchronization operation");
        }
        if self.need_lds_barriers
            && self.guard_depth == 0
            && self.uniform.get(b).copied().unwrap_or(false)
        {
            self.lds_barrier(self.cur);
        }
        let a = self.vgprs(inst, addr.as_ref().ok_or("an LDS address")?, 1)?[0];
        // Out-of-bounds console LDS accesses read 0 and drop writes; clamping
        // read addresses also keeps inactive lanes with garbage VGPRs in bounds.
        let byte_and_bounds = |l: &mut Self, bytes: u32| -> (V, V) {
            let c = l.cur;
            let at = if bytes == 0 {
                a
            } else {
                let o = l.s.m.i32(bytes);
                l.s.m.bin(c, Bin::Add, a, o)
            };
            let two = l.s.m.i32(2);
            let index = l.s.m.bin(c, Bin::LShr, at, two);
            let limit = l.s.m.i32(l.s.lds_dwords);
            let in_bounds = l.s.m.cmp(c, pred::IULT, index, limit);
            (at, in_bounds)
        };
        let width = |n: &str| match n.rsplit('_').next() {
            Some("b64") => 2,
            Some("b96") => 3,
            Some("b128") => 4,
            _ => 1,
        };
        // (byte offset, dwords) per element; read2/write2 scale their two
        // offsets by the element size (×64 for st64).
        let pair = |n: &str| -> Vec<(u32, u32)> {
            let w = width(n);
            let scale = 4 * w * if n.contains("st64") { 64 } else { 1 };
            vec![
                (u32::from(*offset0) * scale, w),
                (u32::from(*offset1) * scale, w),
            ]
        };
        // Bytes and halves: (bits, signed) of `ds_read_{u,i}{8,16}` and
        // `ds_write_b{8,16}`, read and written within their dword.
        // shortcut: a half at byte 3 of a dword (crossing into the next) is
        // taken as within it.
        // `_d16`: the value is a 16-bit half of the VGPR (`_hi`: the high
        // one); a read keeps the other half.
        let base = name.trim_end_matches("_hi").trim_end_matches("_d16");
        let (d16, hi) = (name.contains("_d16"), name.ends_with("_hi"));
        let narrow = match base {
            "ds_read_u8" | "ds_write_b8" => Some((8u32, false)),
            "ds_read_i8" => Some((8, true)),
            "ds_read_u16" | "ds_write_b16" => Some((16, false)),
            "ds_read_i16" => Some((16, true)),
            _ => None,
        };
        if let Some((bits, signed)) = narrow {
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let (address, in_bounds) = byte_and_bounds(self, offset);
            let three = k(self, 3);
            let byte = self.s.m.bin(c, Bin::And, address, three);
            let shift = self.s.m.bin(c, Bin::Shl, byte, three);
            let mask = k(self, (1u32 << bits) - 1);
            if name.starts_with("ds_read") {
                let reg = dst_reg.ok_or("an LDS read result")?;
                let zero = k(self, 0);
                let safe = self.s.m.select(c, in_bounds, address, zero);
                let loaded = self.lds_load_dword(safe);
                let c = self.cur;
                let w = self.s.m.select(c, in_bounds, loaded, zero);
                let x = self.s.m.bin(c, Bin::LShr, w, shift);
                let x = if signed {
                    // Up to the top, then arithmetically back down.
                    let up = k(self, 32 - bits);
                    let t = self.s.m.bin(c, Bin::Shl, x, up);
                    self.s.m.bin(c, Bin::AShr, t, up)
                } else {
                    self.s.m.bin(c, Bin::And, x, mask)
                };
                let x = if !d16 {
                    x
                } else if !hi {
                    self.low_half(inst, reg, x)
                } else {
                    let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
                        Some(Ok(o)) => o,
                        _ => k(self, 0),
                    };
                    let (low, sixteen) = (k(self, 0xFFFF), k(self, 16));
                    let old = self.s.m.bin(c, Bin::And, old, low);
                    let x = self.s.m.bin(c, Bin::Shl, x, sixteen);
                    self.s.m.bin(c, Bin::Or, old, x)
                };
                self.write(inst, Reg::V(reg), x, guard);
            } else {
                let mut v = self.vgprs(inst, data.first().ok_or("an LDS write")?, 1)?[0];
                if hi {
                    let sixteen = k(self, 16);
                    v = self.s.m.bin(c, Bin::LShr, v, sixteen);
                }
                let v = self.s.m.bin(c, Bin::And, v, mask);
                let v = self.s.m.bin(c, Bin::Shl, v, shift);
                let hole = self.s.m.bin(c, Bin::Shl, mask, shift);
                let all = k(self, u32::MAX);
                let keep = self.s.m.bin(c, Bin::Xor, hole, all);
                let cond = match guard {
                    Some(g) => self.s.m.bin(c, Bin::And, g, in_bounds),
                    None => in_bounds,
                };
                let at = self.begin_guarded(Some(cond));
                if self.wide_lds {
                    self.s.flags |= flags::INT64 | flags::ATOMIC_INT64_GROUP_SHARED;
                }
                self.lds_update_dword(address, |l, old| {
                    let c = l.cur;
                    let kept = l.s.m.bin(c, Bin::And, old, keep);
                    l.s.m.bin(c, Bin::Or, kept, v)
                });
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_read") {
            let reg = dst_reg.ok_or("an LDS read result")?;
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                return Err("an LDS read of bytes or halves");
            };
            let mut n = 0;
            for (off, w) in elements {
                for k in 0..w {
                    let (address, in_bounds) = byte_and_bounds(self, off + 4 * k);
                    let c = self.cur;
                    let zero = self.s.m.i32(0);
                    let safe = self.s.m.select(c, in_bounds, address, zero);
                    let loaded = self.lds_load_dword(safe);
                    let x = self.s.m.select(self.cur, in_bounds, loaded, zero);
                    self.write(inst, Reg::V(reg + n), x, guard);
                    n += 1;
                }
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_write") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                return Err("an LDS write of bytes or halves");
            };
            let mut stores = Vec::new();
            for (e, (off, w)) in elements.into_iter().enumerate() {
                let vals = self.vgprs(inst, data.get(e).ok_or("an LDS write")?, w as u8)?;
                for (k, x) in vals.into_iter().enumerate() {
                    stores.push((off + 4 * k as u32, x));
                }
            }
            if let [(off, x)] = stores[..] {
                let (address, in_bounds) = byte_and_bounds(self, off);
                let cond = match guard {
                    Some(g) => self.s.m.bin(self.cur, Bin::And, g, in_bounds),
                    None => in_bounds,
                };
                let at = self.begin_guarded(Some(cond));
                self.lds_store_dword(address, x);
                self.end_guarded(at, &[]);
            } else {
                let outer = self.begin_guarded(guard);
                for (off, x) in stores {
                    let (address, in_bounds) = byte_and_bounds(self, off);
                    let inner = self.begin_guarded(Some(in_bounds));
                    self.lds_store_dword(address, x);
                    self.end_guarded(inner, &[]);
                }
                self.end_guarded(outer, &[]);
            }
            return Ok(());
        }
        // Atomics: the old value returns with `_rtn`.
        if name.ends_with("_f32") || name.ends_with("_f64") {
            // Mesa's aco_insert_fp_mode.cpp tracks the applicable MODE
            // denormal bits for these. NativeProgram does not carry them.
            return Err("an LDS floating-point atomic requiring the console's denormal mode");
        }
        if name.starts_with("ds_condxchg32") {
            return Err("an LDS conditional exchange with undocumented compare operands");
        }
        let (op, wide, kind) = ds_integer_atomic(name).ok_or("an LDS operation")?;
        if matches!(op, "wrxchg2" | "wrxchg2st64") {
            // AMD RDNA 2 ISA §12.13 specifies separate exchanges at the
            // two addresses. It does not specify their order if they alias.
            if offset0 == offset1 {
                return Err("an LDS paired exchange with coincident addresses");
            }
            let reg = dst_reg.ok_or("an LDS exchange result")?;
            let d0 = self.atomic_data(inst, data.first().ok_or("LDS exchange data")?, wide)?;
            let d1 = self.atomic_data(inst, data.get(1).ok_or("LDS exchange data")?, wide)?;
            let ty = self.s.m.type_of(d0);
            let zero = self.s.m.int_const(ty, 0);
            let at = self.begin_guarded(guard);
            let mut old = Vec::new();
            for ((off, _), data) in pair(name).into_iter().zip([d0, d1]) {
                let address = byte_and_bounds(self, off).0;
                let x = if !wide && self.wide_lds {
                    self.lds_update_dword(address, |_, _| data)
                } else {
                    let p = self.lds_ptr(address);
                    self.s.m.atomic(self.cur, Rmw::Xchg, p, data)
                };
                old.push((x, zero));
            }
            if self.wide_lds {
                self.s.flags |= flags::INT64 | flags::ATOMIC_INT64_GROUP_SHARED;
            }
            let words = if wide { 2 } else { 1 };
            for (k, x) in self.end_guarded(at, &old).into_iter().enumerate() {
                self.write_atomic_result(inst, reg + k as u8 * words, x, wide, guard);
            }
            return Ok(());
        }
        let rmw = match (op, kind) {
            ("add" | "sub", _) => Some(Rmw::Add),
            ("and", _) => Some(Rmw::And),
            ("or", _) => Some(Rmw::Or),
            ("xor", _) => Some(Rmw::Xor),
            ("min", b'i') => Some(Rmw::Min),
            ("max", b'i') => Some(Rmw::Max),
            ("min", _) => Some(Rmw::UMin),
            ("max", _) => Some(Rmw::UMax),
            ("wrxchg", _) => Some(Rmw::Xchg),
            _ => None,
        };
        let update = match op {
            "rsub" => Some(AtomicUpdate::ReverseSub),
            "inc" => Some(AtomicUpdate::Inc),
            "dec" => Some(AtomicUpdate::Dec),
            "mskor" => Some(AtomicUpdate::MaskOr),
            "wrap" => Some(AtomicUpdate::Wrap),
            _ if rmw.is_some() || op == "cmpst" => None,
            _ => return Err("an LDS operation"),
        };
        let v0 = self.atomic_data(inst, data.first().ok_or("LDS atomic data")?, wide)?;
        let ty = self.s.m.type_of(v0);
        let zero = self.s.m.int_const(ty, 0);
        let v1 = if matches!(op, "mskor" | "wrap" | "cmpst") {
            self.atomic_data(inst, data.get(1).ok_or("a second LDS atomic operand")?, wide)?
        } else {
            zero
        };
        let (address, in_bounds) = byte_and_bounds(self, offset);
        let cond = match guard {
            Some(g) => self.s.m.bin(self.cur, Bin::And, g, in_bounds),
            None => in_bounds,
        };
        let at = self.begin_guarded(Some(cond));
        if wide || self.wide_lds {
            self.s.flags |= flags::INT64 | flags::ATOMIC_INT64_GROUP_SHARED;
        }
        let r = if !wide && self.wide_lds {
            self.lds_update_dword(address, |l, old| {
                if let Some(update) = update {
                    return l.atomic_new_value(update, old, v0, v1);
                }
                let c = l.cur;
                match op {
                    "add" => l.s.m.bin(c, Bin::Add, old, v0),
                    "sub" => l.s.m.bin(c, Bin::Sub, old, v0),
                    "and" => l.s.m.bin(c, Bin::And, old, v0),
                    "or" => l.s.m.bin(c, Bin::Or, old, v0),
                    "xor" => l.s.m.bin(c, Bin::Xor, old, v0),
                    "wrxchg" => v0,
                    "cmpst" => {
                        let same = l.s.m.cmp(c, pred::EQ, old, v0);
                        l.s.m.select(c, same, v1, old)
                    }
                    _ => {
                        let pred = match (op, kind) {
                            ("min", b'i') => pred::SLT,
                            ("max", b'i') => pred::SGT,
                            ("min", _) => pred::IULT,
                            ("max", _) => pred::IUGT,
                            _ => unreachable!(),
                        };
                        let take = l.s.m.cmp(c, pred, v0, old);
                        l.s.m.select(c, take, v0, old)
                    }
                }
            })
        } else if let Some(rmw) = rmw {
            let p = self.lds_ptr(address);
            // DXIL supports atomic add, with subtraction encoded as -DATA.
            let data = if op == "sub" { self.s.m.bin(self.cur, Bin::Sub, zero, v0) } else { v0 };
            self.s.m.atomic(self.cur, rmw, p, data)
        } else if let Some(update) = update {
            let p = self.lds_ptr(address);
            self.atomic_update(zero, |l, old| {
                let x = l.atomic_new_value(update, old, v0, v1);
                l.s.m.cmpxchg(l.cur, p, old, x)
            })
        } else {
            // Compare with data0, store data1.
            let p = self.lds_ptr(address);
            self.s.m.cmpxchg(self.cur, p, v0, v1)
        };
        let r = self.end_guarded(at, &[(r, zero)])[0];
        if let Some(reg) = dst_reg {
            self.write_atomic_result(inst, reg, r, wide, guard);
        }
        Ok(())
    }

    fn write_atomic_result(&mut self, inst: &Inst, reg: u8, old: V, wide: bool, guard: Option<V>) {
        if wide {
            let (lo, hi) = self.split64(old);
            self.write(inst, Reg::V(reg), lo, guard);
            self.write(inst, Reg::V(reg + 1), hi, guard);
        } else {
            self.write(inst, Reg::V(reg), old, guard);
        }
    }

    // ---- instructions ----

    fn inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        let offset = inst.machine.offset;
        let name = inst.name();
        self.mask_ok = match self.mask_data {
            MaskData::Console => true,
            MaskData::Host => {
                name.starts_with("s_bcnt1_i32_")
                    || name.starts_with("s_ff1_i32_")
                    || name.starts_with("v_mbcnt_")
                    || matches!(
                        name,
                        "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_u64" | "s_cmp_lg_u64"
                    )
            }
            MaskData::None => false,
        };
        if self.emulating_wave64() {
            self.materialize_masks(b, i, inst);
        }
        let effect = matches!(inst.fields, Fields::Exp { .. })
            || matches!(inst.fields, Fields::Flat { .. })
                && !(name.starts_with("global_load_") || name.starts_with("scratch_load_"))
            || name.contains("store")
            || name.contains("atomic")
            || name == "s_barrier"
            || matches!(inst.fields, Fields::Ds { .. })
                && !(name.starts_with("ds_read")
                    || name.starts_with("ds_swizzle")
                    || name.contains("permute"));
        let r = match &inst.fields {
            Fields::Sop { .. } => self.sop(inst),
            Fields::Sopp { .. } if name == "s_barrier" && self.no_barrier.contains(&b) => Ok(()),
            Fields::Sopp { .. } if name == "s_barrier" => {
                // Thread-group sync with a shared-memory fence, and a UAV
                // fence when the program writes UAVs (the waits before an
                // s_barrier make those writes visible on the console).
                let mode = if self.writes_uavs { 1 | 2 | 8 } else { 1 | 8 };
                let mode = self.s.m.i32(mode);
                let void = self.s.t.void;
                self.s
                    .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
                Ok(())
            }
            Fields::Sopp { .. } => Ok(()),
            Fields::Smem { .. } => self.smem(inst, (b, i)),
            Fields::Valu(_) if name.starts_with("v_interp_") => self.interp(inst, (b, i)),
            Fields::Valu(_) => self.valu(inst, b, i),
            Fields::Vintrp { .. } => self.interp(inst, (b, i)),
            Fields::Image { .. } => self.image(inst, b, i),
            Fields::Buffer { .. } => self.buffer(inst, b, i),
            Fields::Exp { .. } => self.export(b, i),
            Fields::Ds { .. } => self.ds(inst, b, i),
            Fields::Flat { .. } if name.starts_with("global_") => self.flat(inst, b, i),
            Fields::Flat { .. } if name.starts_with("scratch_") => self.scratch(inst, b, i),
            // Generic addresses: private, shared or global memory by aperture.
            Fields::Flat { .. } => Err("flat memory"),
        };
        match r {
            Ok(()) => Ok(()),
            Err(why) if effect => Err(unsupported(offset, why)),
            Err(why) => {
                for &(_, v) in &inst.writes {
                    self.vals.insert(v, Val::Missing(why));
                }
                Ok(())
            }
        }
    }

    fn is_mask(&self, v: Option<Value>) -> bool {
        v.is_some_and(|v| self.x.st.masks.contains_key(&v))
    }

    fn sop(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Sop { dst, src, imm } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let scc = inst.write(Reg::Scc);
        let dst_code = match dst {
            Some(Operand::Scalar { code, .. }) => Some(*code),
            _ => None,
        };
        let lo = dst_code.and_then(|c| inst.write(Reg::S(c)));
        if inst.write(Reg::S(EXEC)).is_some() || self.is_mask(lo) {
            return Err(LANE_MASK);
        }
        if matches!(
            name,
            "s_setreg_imm32_b32" | "s_setreg_b32" | "s_swappc_b64" | "s_version"
        ) || name.starts_with("s_waitcnt")
        {
            // shortcut: the float MODE register is not modelled; s_swappc_b64 is the
            // vertex fetch call, whose results arrive as inputs.
            return Err("a register the emitter does not model");
        }
        let c = self.cur;
        let src32 = |l: &mut Self, k: usize| -> Result<V, Why> {
            // SOPK arithmetic reads its destination as the first source.
            if k == 0 && matches!(name, "s_addk_i32" | "s_mulk_i32") {
                let d = dst.as_ref().ok_or("a missing operand")?;
                return l.operand(inst, d);
            }
            let op = src.get(k).ok_or("a missing operand")?;
            l.operand(inst, op)
        };
        let sext16 = |l: &mut Self| l.s.m.i32(i32::from(imm.unwrap_or(0) as i16) as u32);
        // Dword `h` of 64-bit source `k` (inline constants sign-extend).
        let half = |l: &mut Self, k: usize, h: u8| -> Result<V, Why> {
            match src.get(k) {
                Some(&Operand::Scalar { code, .. }) if code != NULL => {
                    let v = inst.read(Reg::S(code + h)).ok_or("an unread operand")?;
                    l.i32_of(v)
                }
                Some(op) if h == 0 => l.operand(inst, op),
                Some(Operand::Const(k)) => {
                    let x = crate::resource::inline(*k).ok_or("a special operand")?;
                    Ok(l.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                // Of a signed operand, a literal sign-extends too.
                Some(&Operand::Literal(x)) if name.ends_with("_i64") => {
                    Ok(l.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                _ => Ok(l.s.m.i32(0)),
            }
        };
        if let Some(op) = name.strip_prefix("s_cmp_").filter(|o| o.ends_with("_u64")) {
            // eq and lg only: both dwords.
            let mut r = None;
            for h in 0..2 {
                let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                let e = self.s.m.cmp(c, pred::EQ, a, b);
                r = Some(match r {
                    Some(x) => self.s.m.bin(c, Bin::And, x, e),
                    None => e,
                });
            }
            let mut r = r.expect("two dwords");
            if op.starts_with("lg") {
                let t = self.s.m.bool(true);
                r = self.s.m.bin(c, Bin::Xor, r, t);
            }
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").or(name.strip_prefix("s_cmpk_")) {
            let (a, b) = if name.starts_with("s_cmpk_") {
                let b = if op.ends_with("_i32") {
                    sext16(self)
                } else {
                    self.s.m.i32(u32::from(imm.unwrap_or(0)))
                };
                (src32(self, 0)?, b)
            } else {
                if op.ends_with("64") {
                    return Err("a 64-bit scalar compare");
                }
                (src32(self, 0)?, src32(self, 1)?)
            };
            let signed = op.ends_with("_i32");
            let p = match &op[..2] {
                "eq" => pred::EQ,
                "lg" => pred::NE,
                "gt" if signed => pred::SGT,
                "ge" if signed => pred::SGE,
                "lt" if signed => pred::SLT,
                "le" if signed => pred::SLE,
                "gt" => pred::IUGT,
                "ge" => pred::IUGE,
                "lt" => pred::IULT,
                "le" => pred::IULE,
                _ => return Err("a scalar compare"),
            };
            let r = self.s.m.cmp(c, p, a, b);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
        if wide {
            let hi = dst_code.and_then(|c| inst.write(Reg::S(c + 1)));
            let nonzero = |l: &mut Self, a: V, b: V| {
                let o = l.s.m.bin(c, Bin::Or, a, b);
                let z = l.s.m.i32(0);
                l.s.m.cmp(c, pred::NE, o, z)
            };
            return match name {
                "s_bcnt1_i32_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let na = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[a]);
                    let nb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[b]);
                    let r = self.s.m.bin(c, Bin::Add, na, nb);
                    self.set(lo, Val::I32(r));
                    let z = self.s.m.i32(0);
                    let s = self.s.m.cmp(c, pred::NE, r, z);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_ff1_i32_b64" => {
                    // The lowest set bit; -1 for none (as DXIL's first bit low).
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let fa = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[a]);
                    let fb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[b]);
                    let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
                    let fb32 = self.s.m.bin(c, Bin::Add, fb, k32);
                    let none = self.s.m.cmp(c, pred::EQ, b, z);
                    let fb = self.s.m.select(c, none, fb, fb32);
                    let low = self.s.m.cmp(c, pred::NE, a, z);
                    let r = self.s.m.select(c, low, fa, fb);
                    self.set(lo, Val::I32(r));
                    Ok(())
                }
                "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64"
                | "s_nand_b64" | "s_nor_b64" | "s_xnor_b64" | "s_not_b64" => {
                    let ones = self.s.m.i32(u32::MAX);
                    let mut r = Vec::new();
                    for h in 0..2u8 {
                        let a = half(self, 0, h)?;
                        let x = if name == "s_not_b64" {
                            self.s.m.bin(c, Bin::Xor, a, ones)
                        } else {
                            let mut b = half(self, 1, h)?;
                            if name.contains("n2") {
                                b = self.s.m.bin(c, Bin::Xor, b, ones);
                            }
                            let op = match &name[2..5] {
                                "and" | "nan" => Bin::And,
                                "or_" | "orn" | "nor" => Bin::Or,
                                _ => Bin::Xor,
                            };
                            let x = self.s.m.bin(c, op, a, b);
                            if matches!(name, "s_nand_b64" | "s_nor_b64" | "s_xnor_b64") {
                                self.s.m.bin(c, Bin::Xor, x, ones)
                            } else {
                                x
                            }
                        };
                        r.push(x);
                    }
                    self.set(lo, Val::I32(r[0]));
                    self.set(hi, Val::I32(r[1]));
                    let s = nonzero(self, r[0], r[1]);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let n = src32(self, 1)?;
                    let i64t = self.s.m.int(64);
                    self.s.flags |= flags::INT64;
                    let a = self.s.m.cast(c, Cast::ZExt, a, i64t);
                    let b = self.s.m.cast(c, Cast::ZExt, b, i64t);
                    let k32 = self.s.m.int_const(i64t, 32);
                    let b = self.s.m.bin(c, Bin::Shl, b, k32);
                    let x = self.s.m.bin(c, Bin::Or, a, b);
                    let k63 = self.s.m.i32(63);
                    let n = self.s.m.bin(c, Bin::And, n, k63);
                    let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                    let op = match name {
                        "s_lshl_b64" => Bin::Shl,
                        "s_lshr_b64" => Bin::LShr,
                        _ => Bin::AShr,
                    };
                    let r = self.s.m.bin(c, op, x, n);
                    let rl = self.s.m.cast(c, Cast::Trunc, r, self.s.t.i32);
                    let rh = self.s.m.bin(c, Bin::LShr, r, k32);
                    let rh = self.s.m.cast(c, Cast::Trunc, rh, self.s.t.i32);
                    self.set(lo, Val::I32(rl));
                    self.set(hi, Val::I32(rh));
                    let s = nonzero(self, rl, rh);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_mov_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    self.set(lo, Val::I32(a));
                    self.set(hi, Val::I32(b));
                    Ok(())
                }
                "s_cselect_b64" => {
                    let cond = self.scc_in(inst)?;
                    for (h, v) in [(0, lo), (1, hi)] {
                        let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                        let r = self.s.m.select(c, cond, a, b);
                        self.set(v, Val::I32(r));
                    }
                    Ok(())
                }
                "s_wqm_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let (rl, rh) = (self.scalar_wqm32(c, a), self.scalar_wqm32(c, b));
                    self.set(lo, Val::I32(rl));
                    self.set(hi, Val::I32(rh));
                    let s = nonzero(self, rl, rh);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                _ => Err(name),
            };
        }
        let m = &mut self.s.m;
        let _ = m;
        let (r, carry): (V, Option<V>) = match name {
            "s_mov_b32" => (src32(self, 0)?, None),
            "s_movk_i32" => (sext16(self), None),
            "s_cselect_b32" => {
                let cond = self.scc_in(inst)?;
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.s.m.select(c, cond, a, b), None)
            }
            "s_brev_b32" => {
                let a = src32(self, 0)?;
                let r = self
                    .s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a]);
                (r, None)
            }
            "s_not_b32" => {
                let a = src32(self, 0)?;
                let ones = self.s.m.i32(u32::MAX);
                (self.s.m.bin(c, Bin::Xor, a, ones), None)
            }
            "s_add_u32" | "s_add_i32" | "s_addk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_addk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                let r = self.s.m.bin(c, Bin::Add, a, b);
                let carry = if name == "s_add_u32" {
                    Some(self.s.m.cmp(c, pred::IULT, r, a))
                } else {
                    None
                };
                (r, carry)
            }
            "s_sub_u32" | "s_sub_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let r = self.s.m.bin(c, Bin::Sub, a, b);
                let carry = (name == "s_sub_u32").then(|| self.s.m.cmp(c, pred::IULT, a, b));
                (r, carry)
            }
            "s_addc_u32" | "s_subb_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let cin = self.scc_in(inst)?;
                let (r, carry) = self.add_sub_carry(c, name == "s_addc_u32", a, b, Some(cin));
                (r, Some(carry))
            }
            "s_mul_i32" | "s_mulk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_mulk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                (self.s.m.bin(c, Bin::Mul, a, b), None)
            }
            "s_mul_hi_u32" | "s_mul_hi_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.mul_hi(a, b, name.ends_with("i32")), None)
            }
            "s_flbit_i32_b32" | "s_flbit_i32" => {
                // The first bit from the top (differing from the sign for
                // the signed form), -1 for none: DXIL's first-bit-high.
                let a = src32(self, 0)?;
                let op = if name == "s_flbit_i32_b32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_ff1_i32_b32" | "s_bcnt1_i32_b32" => {
                // First set bit (-1 for zero, as DXIL's), or the bit count.
                let a = src32(self, 0)?;
                let op = if name == "s_ff1_i32_b32" { 32 } else { 31 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_bfe_u32" | "s_bfe_i32" => {
                // src1: offset in bits 0-4, width in bits 16-22.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let (k16, k127) = (self.s.m.i32(16), self.s.m.i32(0x7F));
                let w = self.s.m.bin(c, Bin::LShr, b, k16);
                let w = self.s.m.bin(c, Bin::And, w, k127);
                (self.bfe(a, b, w, name == "s_bfe_i32"), None)
            }
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32"
            | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32" => {
                let (a, mut b) = (src32(self, 0)?, src32(self, 1)?);
                let ones = self.s.m.i32(u32::MAX);
                if name.contains("n2") {
                    b = self.s.m.bin(c, Bin::Xor, b, ones);
                }
                let op = match &name[2..5] {
                    "and" | "nan" => Bin::And,
                    "or_" | "orn" | "nor" => Bin::Or,
                    _ => Bin::Xor,
                };
                let mut r = self.s.m.bin(c, op, a, b);
                if matches!(name, "s_nand_b32" | "s_nor_b32" | "s_xnor_b32") {
                    r = self.s.m.bin(c, Bin::Xor, r, ones);
                }
                (r, None)
            }
            "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let op = match name {
                    "s_lshl_b32" => Bin::Shl,
                    "s_lshr_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                (self.s.m.bin(c, op, a, b), None)
            }
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                // SCC: the sum overflows 32 bits, through bits shifted out
                // of src0 or the add's carry.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let n = u32::from(name.as_bytes()[6] - b'0');
                let k = self.s.m.i32(n);
                let t = self.s.m.bin(c, Bin::Shl, a, k);
                let r = self.s.m.bin(c, Bin::Add, t, b);
                let (rest, z) = (self.s.m.i32(32 - n), self.s.m.i32(0));
                let out = self.s.m.bin(c, Bin::LShr, a, rest);
                let shifted_out = self.s.m.cmp(c, pred::NE, out, z);
                let carry = self.s.m.cmp(c, pred::IULT, r, t);
                (r, Some(self.s.m.bin(c, Bin::Or, shifted_out, carry)))
            }
            "s_min_u32" | "s_max_u32" | "s_min_i32" | "s_max_i32" => {
                // SCC: src0 is the one chosen.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let p = match name {
                    "s_min_u32" => pred::IULT,
                    "s_max_u32" => pred::IUGT,
                    "s_min_i32" => pred::SLT,
                    _ => pred::SGT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                (self.s.m.select(c, t, a, b), Some(t))
            }
            "s_abs_i32" => {
                let a = src32(self, 0)?;
                let z = self.s.m.i32(0);
                let neg = self.s.m.cmp(c, pred::SLT, a, z);
                let n = self.s.m.bin(c, Bin::Sub, z, a);
                (self.s.m.select(c, neg, n, a), None)
            }
            "s_bitset0_b32" | "s_bitset1_b32" => {
                // Sets (clears) bit src[4:0] of the destination it reads.
                let old = self.operand(inst, dst.as_ref().ok_or("a missing operand")?)?;
                let a = src32(self, 0)?;
                let (k31, one) = (self.s.m.i32(31), self.s.m.i32(1));
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let bit = self.s.m.bin(c, Bin::Shl, one, a);
                if name == "s_bitset1_b32" {
                    (self.s.m.bin(c, Bin::Or, old, bit), None)
                } else {
                    let ones = self.s.m.i32(u32::MAX);
                    let nb = self.s.m.bin(c, Bin::Xor, bit, ones);
                    (self.s.m.bin(c, Bin::And, old, nb), None)
                }
            }
            "s_wqm_b32" => {
                let a = src32(self, 0)?;
                (self.scalar_wqm32(c, a), None)
            }
            "s_bfm_b32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let one = self.s.m.i32(1);
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                (self.s.m.bin(c, Bin::Shl, m, b), None)
            }
            _ => return Err(name),
        };
        self.set(lo, Val::I32(r));
        if let Some(scc) = scc {
            let val = match carry {
                Some(k) => Val::Bit(k),
                None if matches!(
                    name,
                    "s_mov_b32"
                        | "s_movk_i32"
                        | "s_cselect_b32"
                        | "s_brev_b32"
                        | "s_mul_i32"
                        | "s_mulk_i32"
                ) =>
                {
                    return Ok(());
                }
                None if name.starts_with("s_add") || name.starts_with("s_sub") => {
                    Val::Missing("the carry of a scalar operation")
                }
                None => {
                    let z = self.s.m.i32(0);
                    Val::Bit(self.s.m.cmp(c, pred::NE, r, z))
                }
            };
            self.vals.insert(scc, val);
        }
        Ok(())
    }

    /// `a + b (+ cin)` or `a - b (- cin)` with the carry (borrow) out, as the
    /// scalar and vector add-with-carry instructions set it.
    fn add_sub_carry(&mut self, c: usize, add: bool, a: V, b: V, cin: Option<V>) -> (V, V) {
        let op = if add { Bin::Add } else { Bin::Sub };
        let mut r = self.s.m.bin(c, op, a, b);
        // Without a carry in: a + b wraps below a; a - b borrows when a < b.
        let mut carry = if add {
            self.s.m.cmp(c, pred::IULT, r, a)
        } else {
            self.s.m.cmp(c, pred::IULT, a, b)
        };
        if let Some(cin) = cin {
            let one = self.s.m.cast(c, Cast::ZExt, cin, self.s.t.i32);
            let t = r;
            r = self.s.m.bin(c, op, t, one);
            // The carry in carries on when the partial result is all ones
            // (add) or zero (sub).
            let edge = self.s.m.i32(if add { u32::MAX } else { 0 });
            let at_edge = self.s.m.cmp(c, pred::EQ, t, edge);
            let more = self.s.m.bin(c, Bin::And, cin, at_edge);
            carry = self.s.m.bin(c, Bin::Or, carry, more);
        }
        (r, carry)
    }

    fn scalar_wqm32(&mut self, c: usize, x: V) -> V {
        let (one, two, nibs, fifteen) = (
            self.s.m.i32(1),
            self.s.m.i32(2),
            self.s.m.i32(0x1111_1111),
            self.s.m.i32(0xF),
        );
        let s1 = self.s.m.bin(c, Bin::LShr, x, one);
        let t = self.s.m.bin(c, Bin::Or, x, s1);
        let s2 = self.s.m.bin(c, Bin::LShr, t, two);
        let t = self.s.m.bin(c, Bin::Or, t, s2);
        let t = self.s.m.bin(c, Bin::And, t, nibs);
        self.s.m.bin(c, Bin::Mul, t, fifteen)
    }

    fn scc_in(&mut self, inst: &Inst) -> Result<V, Why> {
        let v = inst.read(Reg::Scc).ok_or("an unread SCC")?;
        match self.get(v)? {
            Val::Bit(x) => Ok(x),
            _ => Err("SCC without a value"),
        }
    }

    fn smem(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Fields::Smem {
            data,
            offset,
            soffset,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if name.starts_with("s_scratch_") {
            return Err("scalar access to the console scratch aperture");
        }
        if name.contains("_store_") {
            return Err("a scalar store without root UAV address support");
        }
        if let Some(a) = self.scalar.get(&site).copied().filter(|a| a.param == CODE) {
            return self.code_smem(inst, a);
        }
        let root_desc = self.scalar.get(&site).copied().and_then(|a| {
            (self.x.native.layout.get(a.param)?.root == RootType::Descriptor).then_some(
                Source::Built {
                    address: a,
                    bits: 0,
                    words: [0, 0],
                },
            )
        });
        let Some(rest) = name.strip_prefix("s_buffer_load_dword").or_else(|| {
            root_desc
                .is_some()
                .then(|| name.strip_prefix("s_load_dword"))
                .flatten()
        }) else {
            return self.descriptor_data(inst, site);
        };
        let count = match rest {
            "" => 1,
            "x2" => 2,
            "x4" => 4,
            "x8" => 8,
            "x16" => 16,
            _ => return Err("a scalar buffer load"),
        };
        let Some(Operand::Scalar { code, .. }) = *data else {
            return Err("a scalar buffer load");
        };
        let c = self.cur;
        let (src, addr, constant) = if let Some(ref built) = root_desc {
            let (base, extra) = self.built_offset(built)?;
            let mut addr = self.s.m.i32(base);
            if let Some(x) = extra {
                addr = self.s.m.bin(c, Bin::Add, addr, x);
            }
            (built, addr, extra.is_none().then_some(base))
        } else {
            let src = self
                .descriptors
                .get(&(site, false))
                .copied()
                .ok_or("an untracked descriptor")?;
            let (base, extra) = self.built_offset(src)?;
            let off = (*offset as u32).wrapping_add(base);
            let mut addr = self.s.m.i32(off);
            if let Some(x) = extra {
                addr = self.s.m.bin(c, Bin::Add, addr, x);
            }
            let has_soff = !matches!(soffset, Operand::Scalar { code: NULL, .. });
            if has_soff {
                let so = self.operand(inst, soffset)?;
                addr = self.s.m.bin(c, Bin::Add, addr, so);
            }
            (src, addr, (!has_soff && extra.is_none()).then_some(off))
        };
        let kind = self.used_kind(src);
        if matches!(src, Source::Built { address, .. } if address.param == CODE) {
            // A descriptor built around a table in the code.
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let x = self.code_dword(a)?;
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        if let Some(u) = kind
            && matches!(u.kind, ResourceType::Srv | ResourceType::Uav)
        {
            let u_bit = if u.kind == ResourceType::Uav { 0x1000 } else { 0 };
            // Scalar buffer loads address bytes, even when the descriptor carries a stride.
            let h = self.handle(site, src, false, (11 | u_bit, 0))?;
            let elem_off = self.s.m.undef(self.s.t.i32);
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let mask = self.s.m.i8(1);
                let al = self.s.m.i32(4);
                let r = self.s.op(
                    c,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    139,
                    &[h, a, elem_off, mask, al],
                );
                let x = self.s.m.extract(c, r, 0);
                let x = self.as_int(x);
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let h = self.handle(site, src, false, (13, 0x1_0000))?;
        let mut rows: HashMap<u32, V> = HashMap::new();
        for k in 0..count {
            let x = match constant {
                Some(base) => {
                    let byte = base + 4 * k;
                    let row = match rows.get(&(byte >> 4)) {
                        Some(&r) => r,
                        None => {
                            let ri = self.s.m.i32(byte >> 4);
                            let r = self.s.op(
                                c,
                                "cbufferLoadLegacy.i32",
                                Attr::ReadOnly,
                                self.s.t.cbuf_ret,
                                59,
                                &[h, ri],
                            );
                            rows.insert(byte >> 4, r);
                            r
                        }
                    };
                    self.s.m.extract(c, row, (byte >> 2) & 3)
                }
                None => {
                    let off = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(c, Bin::Add, addr, off);
                    self.cbuffer_dword(h, a)
                }
            };
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// A descriptor load (`s_load_dword*` from a table or heap) whose dwords
    /// the code also reads as numbers: the console compiler computes sizes
    /// from descriptor bits (an image's width, height, depth, levels; a
    /// buffer's record count) instead of querying them. Those fields are
    /// rebuilt from the host view's dimensions in the GFX10 layout (image
    /// dword 1 bits 30-31 and dword 2 bits 0-13 width - 1, dword 2 bits
    /// 14-29 height - 1, dword 3 identity swizzle, last level at 16, type at
    /// 28, dword 4 depth or layers - 1; buffer dword 2 the record count).
    // shortcut: the base level reads 0 (the host view starts at its own
    // level 0, so sizes the code shifts by it come out right) and the other
    // dwords (addresses, formats, tiling) have no host value: reading them
    // stops the program.
    fn descriptor_data(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        const OTHER: Why = "a descriptor's address or format read as data";
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar memory read as data without a scalar destination");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory read as data of another kind"),
        };
        // How the program uses the descriptor tells the view's kind. Prefer an
        // image or vector buffer use when a descriptor has both scalar and
        // vector uses, since those carry format and indexing cues.
        let mut use_at: Option<(Site, u8)> = None;
        for d in self
            .x
            .res
            .descriptors
            .iter()
            .filter(|d| d.role == Role::Resource)
        {
            for l in d.source.leaves() {
                if let Source::Loaded { load, first } = *l
                    && load == site
                {
                    let ui = &self.x.p.blocks[d.site.0].insts[d.site.1];
                    let rich = matches!(ui.fields, Fields::Image { .. } | Fields::Buffer { .. });
                    if use_at.is_none() || rich {
                        use_at = Some((d.site, first));
                        if rich {
                            break;
                        }
                    }
                }
            }
            if let Some((at, _)) = use_at
                && matches!(
                    self.x.p.blocks[at.0].insts[at.1].fields,
                    Fields::Image { .. } | Fields::Buffer { .. }
                )
            {
                break;
            }
        }
        let Some((at, first)) = use_at else {
            return Err("a descriptor read only as data");
        };
        let src = Source::Loaded { load: site, first };
        let used = self.used_kind(&src);
        let ui: &Inst = &self.x.p.blocks[at.0].insts[at.1];
        let un = ui.name();
        let uav = un.contains("store")
            || un.contains("atomic")
            || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let (props, image) = match &ui.fields {
            Fields::Image { dim, .. } => {
                let kind = match dim {
                    0 => 1,
                    1 => 2,
                    2 => 4,
                    4 => 6,
                    5 => 7,
                    _ => return Err("an image dimension (cube or MSAA)"),
                };
                ((kind | u, 0x409), Some(kind))
            }
            Fields::Buffer { idxen, format, .. } => {
                let stride = used.map_or(0, |u| u.stride);
                let props = if un.starts_with("buffer_load_format")
                    || un.starts_with("buffer_store_format")
                {
                    (10 | u, 0x409)
                } else if *idxen && format.is_none() && stride >= 4 && stride.is_multiple_of(4) {
                    (12 | u, stride)
                } else {
                    (11 | u, 0)
                };
                if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                    return Err("a constant buffer's descriptor read as data");
                }
                (props, None)
            }
            Fields::Smem { .. } if un.starts_with("s_buffer_") => {
                if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                    return Err("a constant buffer's descriptor read as data");
                }
                ((11 | u, 0), None)
            }
            Fields::Smem { .. } => {
                return Err("a descriptor read as data and by a scalar load");
            }
            Fields::Flat { .. } => {
                return Err("a descriptor read as data and by a global access");
            }
            _ => return Err("a descriptor read as data and by another instruction"),
        };
        let h = self.handle(site, &src, false, props)?;
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let mip = if image.is_some() && !uav {
            self.s.m.i32(0)
        } else {
            self.s.m.undef(i32t)
        };
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let field = |l: &mut Self, k: u32| l.s.m.extract(c, d, k);
        let words: Vec<(u32, V)> = match image {
            Some(kind) => {
                let one = self.s.m.i32(1);
                let w = field(self, 0);
                let (height, depth) = match kind {
                    1 => (one, one),
                    2 => (field(self, 1), one),
                    6 => (one, field(self, 1)),
                    _ => (field(self, 1), field(self, 2)),
                };
                // The level count only feeds word 3. A level query of a 3D
                // image inside a loop crashes NVIDIA's Vulkan compiler (vkd3d-
                // proton turns it into OpImageQueryLevels), so query it only
                // when the program reads that word as data.
                let reads_word3 = (0..count)
                    .filter(|&k| k.wrapping_sub(u32::from(first)) == 3)
                    .filter_map(|k| inst.write(Reg::S(code + k as u8)))
                    .any(|v| self.read_as_data(v));
                let levels = if uav || !reads_word3 {
                    one
                } else {
                    field(self, 3)
                };
                let less = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Sub, x, one);
                let (w, height, depth, levels) = (
                    less(self, w),
                    less(self, height),
                    less(self, depth),
                    less(self, levels),
                );
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (k30, k2, k14, k16, mask) = (
                    k(self, 30),
                    k(self, 2),
                    k(self, 14),
                    k(self, 16),
                    k(self, 0x3FFF),
                );
                let d1 = self.s.m.bin(c, Bin::Shl, w, k30);
                let hi = self.s.m.bin(c, Bin::LShr, w, k2);
                let hi = self.s.m.bin(c, Bin::And, hi, mask);
                let ht = self.s.m.bin(c, Bin::Shl, height, k14);
                let d2 = self.s.m.bin(c, Bin::Or, hi, ht);
                let ty = [0, 8, 9, 0, 10, 0, 12, 13][kind as usize];
                let base = k(self, 0xFAC | ty << 28);
                let last = self.s.m.bin(c, Bin::Shl, levels, k16);
                let d3 = self.s.m.bin(c, Bin::Or, base, last);
                vec![(1, d1), (2, d2), (3, d3), (4, depth)]
            }
            None => vec![(2, field(self, 0))],
        };
        for k in 0..count {
            let rel = k.wrapping_sub(u32::from(first));
            match words.iter().find(|w| w.0 == rel) {
                Some(&(_, x)) => self.write(inst, Reg::S(code + k as u8), x, None),
                None => {
                    if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                        self.vals.insert(v, Val::Missing(OTHER));
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether an instruction or phi reads `v` other than as a resource
    /// operand of a memory instruction (which takes the descriptor whole).
    fn read_as_data(&self, v: Value) -> bool {
        self.x.p.blocks.iter().any(|b| {
            b.phis.iter().any(|p| p.args.contains(&v))
                || b.insts.iter().any(|i| {
                    !matches!(
                        i.fields,
                        Fields::Image { .. } | Fields::Buffer { .. } | Fields::Smem { .. }
                    ) && i.reads.iter().any(|r| r.1 == v)
                })
        })
    }

    /// `s_load_dword*` from the program's constant table.
    fn code_smem(&mut self, inst: &Inst, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar load from the code"),
        };
        let c = self.cur;
        let mut addr = self.s.m.i32(a.offset);
        if let Some(d) = a.dynamic {
            let d = self.i32_of(d)?;
            addr = self.s.m.bin(c, Bin::Add, addr, d);
        }
        for k in 0..count {
            let off = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, addr, off);
            let x = self.code_dword(at)?;
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// The dword at a computed byte address of a constant buffer.
    fn cbuffer_dword(&mut self, h: V, a: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let ri = self.s.m.bin(c, Bin::LShr, a, four);
        let r = self.s.op(
            c,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, ri],
        );
        let two = self.s.m.i32(2);
        let three = self.s.m.i32(3);
        let comp = self.s.m.bin(c, Bin::LShr, a, two);
        let comp = self.s.m.bin(c, Bin::And, comp, three);
        let mut x = self.s.m.extract(c, r, 0);
        for j in 1..4 {
            let e = self.s.m.extract(c, r, j);
            let jc = self.s.m.i32(j);
            let is = self.s.m.cmp(c, pred::EQ, comp, jc);
            x = self.s.m.select(c, is, e, x);
        }
        x
    }

    /// `channels` channels of the element at byte `off` of raw buffer `h`,
    /// converted from the GFX10 buffer format `format` as the hardware's
    /// typed loads do (channels past the format's read 0, 0, 0, 1).
    // shortcut: the descriptor's channel swizzle is taken as the identity
    // (it is for raw views and the descriptors code builds), elements of 4
    // bytes or more as dword-aligned, and the packed float and 10_10_10_2
    // layouts are refused; decode them when a program needs them. Norm
    // formats divide, which the AMD driver does 1 ULP off the hardware's
    // correctly rounded conversion for some values; correct it with an
    // integer check if a title compares converted values exactly.
    fn format_load(&mut self, h: V, off: V, format: u8, channels: u32) -> Result<Vec<V>, Why> {
        let (bits, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let total: u32 = bits.iter().sum();
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let load = |l: &mut Self, at: V, n: u32| -> Vec<V> {
            let mask = l.s.m.i8((1u8 << n) - 1);
            let r = l.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                l.s.t.res_ret,
                139,
                &[h, at, u, mask, align],
            );
            (0..n)
                .map(|k| {
                    let x = l.s.m.extract(c, r, k);
                    l.as_int(x)
                })
                .collect()
        };
        let words = if total < 32 {
            // A byte or half inside the dword that holds it.
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let w = load(self, at, 1)[0];
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            vec![self.s.m.bin(c, Bin::LShr, w, sh)]
        } else {
            load(self, off, total / 32)
        };
        let float = |l: &mut Self, f: V| l.as_int(f);
        let mut out = Vec::new();
        let mut at = 0;
        for &w in bits {
            let word = words[(at / 32) as usize];
            let field = if w == 32 {
                word
            } else {
                let sh = self.s.m.i32(at % 32);
                let t = self.s.m.bin(c, Bin::LShr, word, sh);
                let m = self.s.m.i32((1 << w) - 1);
                self.s.m.bin(c, Bin::And, t, m)
            };
            at += w;
            // Sign-extended from the field's top bit.
            let signed = |l: &mut Self| {
                if w == 32 {
                    return field;
                }
                let k = l.s.m.i32(32 - w);
                let t = l.s.m.bin(c, Bin::Shl, field, k);
                l.s.m.bin(c, Bin::AShr, t, k)
            };
            let f32t = self.s.t.f32;
            let x = match numeric {
                Numeric::Unorm => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    let max = self.s.m.f32((((1u64 << w) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    float(self, q)
                }
                Numeric::Snorm => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    let max = self.s.m.f32((((1u64 << (w - 1)) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    let lo = self.s.m.f32((-1.0f32).to_bits());
                    let q = self.binary(35, q, lo);
                    float(self, q)
                }
                Numeric::Uscaled => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    float(self, f)
                }
                Numeric::Sscaled => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    float(self, f)
                }
                Numeric::Uint => field,
                Numeric::Sint => signed(self),
                Numeric::Float if w == 32 => field,
                Numeric::Float => {
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[field]);
                    float(self, f)
                }
            };
            out.push(x);
        }
        let one = match numeric {
            Numeric::Uint | Numeric::Sint => 1,
            _ => 1.0f32.to_bits(),
        };
        for k in out.len() as u32..channels {
            out.push(self.s.m.i32(if k == 3 { one } else { 0 }));
        }
        out.truncate(channels as usize);
        Ok(out)
    }

    /// The byte offset a buffer descriptor built around a root descriptor adds
    /// (constant and computed parts); none for other descriptors.
    fn built_offset(&mut self, src: &Source) -> Result<(u32, Option<V>), Why> {
        match src {
            Source::Built { address, .. } => {
                let x = match address.dynamic {
                    Some(v) => Some(self.i32_of(v)?),
                    None => None,
                };
                Ok((address.offset, x))
            }
            _ => Ok((0, None)),
        }
    }

    /// The used-resource entry a descriptor loaded from a table falls in.
    fn used_kind(&self, src: &Source) -> Option<&'a UsedResource> {
        if let Source::Phi { args, .. } | Source::Select { args, .. } | Source::Lanes { args, .. } =
            src
        {
            return self.used_kind(args.first()?);
        }
        if let Source::Built { address, .. } = src {
            let rp = self.x.native.layout.get(address.param)?;
            return self
                .x
                .native
                .used
                .iter()
                .find(|u| u.root == RootType::Descriptor && u.user_data == rp.user_data);
        }
        let (Source::Loaded { load, first } | Source::Fetched { load, first }) = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        let rp = self.x.native.layout.get(a.param)?;
        if rp.root != RootType::Table {
            return None;
        }
        let dword = (a.offset / 4) + u32::from(*first);
        self.x
            .native
            .used
            .iter()
            .filter(|u| {
                u.root == RootType::Table
                    && u.user_data == rp.user_data
                    && u.table_offset_dwords <= dword
            })
            .max_by_key(|u| u.table_offset_dwords)
    }

    /// A handle for the descriptor `src` read at `site`.
    fn handle(
        &mut self,
        site: Site,
        src: &Source,
        sampler: bool,
        props: (u32, u32),
    ) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => {
                // The element type is part of the key: one UAV read as float
                // and used atomically needs two handles (an atomic on a float
                // handle is invalid and crashed the NVIDIA compiler).
                let key = (self.cur, *load, *first, sampler, props.0, props.1);
                if let Some(&h) = self.handles.get(&key) {
                    return Ok(h);
                }
                let index = self.heap_index(*load, *first, sampler)?;
                let h = self.s.heap_handle(self.cur, index, sampler, false, props);
                self.handles.insert(key, h);
                Ok(h)
            }
            Source::Constant(words) if sampler => {
                let words: [u32; 4] = words
                    .get(..4)
                    .and_then(|w| w.try_into().ok())
                    .ok_or("a sampler constant")?;
                let n = match self.samplers.iter().position(|w| *w == words) {
                    Some(n) => n,
                    None => {
                        self.samplers.push(words);
                        self.samplers.len() - 1
                    }
                };
                if props.0 & 0x8000 != 0 {
                    // shortcut: one sampler used with and without comparison
                    // would need two static samplers; it becomes a comparison one.
                    self.s.cmp_samplers |= 1 << n;
                }
                Ok(self.s.sampler_handle(self.cur, n as u32))
            }
            Source::Constant(_) => Err("a constant resource descriptor"),
            Source::Built { address, .. } => {
                let rp = *self
                    .x
                    .native
                    .layout
                    .get(address.param)
                    .ok_or("a descriptor built around a table in the code")?;
                let kind = self
                    .x
                    .native
                    .used
                    .iter()
                    .find(|u| u.user_data == rp.user_data)
                    .map(|u| u.kind);
                match (rp.root, kind, props.0) {
                    (RootType::Descriptor, Some(ResourceType::Cbv) | None, 13) => {
                        Ok(self.s.root_cbv_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 11) => {
                        Ok(self.s.root_srv_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Uav) | None, p)
                        if p == 11 | 0x1000 =>
                    {
                        Ok(self.s.root_uav_handle(rp.slot))
                    }
                    _ => Err("a typed root view"),
                }
            }
            Source::Phi { block, args } => {
                // Each predecessor computes its heap index at its end; the
                // join picks one with a phi.
                let _ = site;
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let (Some(tail), Source::Loaded { load, first }) = (self.tail[q], arg) else {
                        return Err("a descriptor chosen by control flow");
                    };
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let here = std::mem::replace(&mut self.cur, at);
                    let index = self.heap_index(*load, *first, sampler);
                    self.cur = here;
                    incoming.push((index?, at));
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(self.s.heap_handle(self.cur, phi, sampler, false, props))
            }
            Source::Select { scc, args } => {
                // The heap index of each choice, picked by SCC (uniform).
                let [
                    Source::Loaded { load: a, first: fa },
                    Source::Loaded { load: b, first: fb },
                ] = args[..]
                else {
                    return Err("a descriptor chosen by a condition");
                };
                let cond = match self.get(*scc)? {
                    Val::Bit(x) => x,
                    _ => return Err("SCC without a value"),
                };
                let set = self.heap_index(a, fa, sampler)?;
                let clear = self.heap_index(b, fb, sampler)?;
                let index = self.s.m.select(self.cur, cond, set, clear);
                Ok(self.s.heap_handle(self.cur, index, sampler, false, props))
            }
            Source::Fetched { .. } | Source::Lanes { .. } => {
                // A descriptor each lane has its own of (a waterfall loop
                // reads it from the first active lane): each lane's heap
                // index, non-uniform.
                let index = self.lane_heap_index(src, sampler)?;
                Ok(self.s.heap_handle(self.cur, index, sampler, true, props))
            }
        }
    }

    /// The view-heap slot of `src` when it comes from a descriptor table or
    /// heap (`Loaded` or `Phi` of `Loaded`).
    fn heap_slot(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Phi { block, args } => {
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let (Some(tail), Source::Loaded { load, first }) = (self.tail[q], arg) else {
                        return Err("a descriptor chosen by control flow");
                    };
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let here = std::mem::replace(&mut self.cur, at);
                    let index = self.heap_index(*load, *first, sampler);
                    self.cur = here;
                    incoming.push((index?, at));
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            _ => Err("a descriptor outside the heaps"),
        }
    }

    /// Byte offset `index * stride + off` for a structured buffer whose
    /// stride was not recovered statically, reading `stride` from the host's
    /// runtime stride table (`t0`, [`super::STRIDE_SPACE`]).
    fn runtime_stride_offset(&mut self, src: &Source, index: V, off: V) -> Result<V, Why> {
        let slot = self
            .heap_slot(src, false)
            .map_err(|_| "a structured buffer of unknown stride")?;
        let c = self.cur;
        let table = self.s.stride_table_handle();
        let four = self.s.m.i32(4);
        let addr = self.s.m.bin(c, Bin::Mul, slot, four);
        let iu = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8(1);
        let align = self.s.m.i32(4);
        let res = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[table, addr, iu, mask, align],
        );
        let stride_f = self.s.m.extract(c, res, 0);
        let stride = self.as_int(stride_f);
        let elem_off = self.s.m.bin(c, Bin::Mul, index, stride);
        Ok(self.s.m.bin(c, Bin::Add, elem_off, off))
    }

    /// The heap index of a descriptor that may differ between lanes,
    /// computed in the current block.
    fn lane_heap_index(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Fetched { load, first } => self.fetched_index(*load, *first, sampler),
            Source::Lanes { mask, args } | Source::Select { scc: mask, args } => {
                let [a, b] = &args[..] else {
                    return Err("a descriptor chosen per lane");
                };
                let cond = if matches!(src, Source::Lanes { .. }) {
                    let &m = self
                        .x
                        .st
                        .masks
                        .get(mask)
                        .ok_or("a descriptor chosen per lane by an untracked mask")?;
                    self.boolean(m, self.cur)?
                } else {
                    match self.get(*mask)? {
                        Val::Bit(x) => x,
                        _ => return Err("SCC without a value"),
                    }
                };
                let set = self.lane_heap_index(a, sampler)?;
                let clear = self.lane_heap_index(b, sampler)?;
                Ok(self.s.m.select(self.cur, cond, set, clear))
            }
            _ => Err("a per-lane descriptor from neither a heap load nor a choice of them"),
        }
    }

    /// The heap index of the descriptor at dword `first` of what the global
    /// load at `load` reads, in each lane: its address (base plus offsets)
    /// over the descriptor size.
    fn fetched_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        const WHY: Why = "a per-lane descriptor load";
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Flat {
            addr: Some(Operand::Vector { reg, .. }),
            saddr,
            offset,
            ..
        } = li.fields
        else {
            return Err(WHY);
        };
        let c = self.cur;
        // The low dword of the address: SGPR base plus VGPR offset, or the
        // VGPR pair's low half (heap addresses fit in 32 bits).
        let mut at = self.i32_of(li.read(Reg::V(reg)).ok_or(WHY)?)?;
        if let Some(Operand::Scalar { code, .. }) = saddr
            && code != NULL
        {
            let base = self.i32_of(li.read(Reg::S(code)).ok_or(WHY)?)?;
            at = self.s.m.bin(c, Bin::Add, base, at);
        }
        let k = self
            .s
            .m
            .i32((offset as i32 as u32).wrapping_add(4 * u32::from(first)));
        at = self.s.m.bin(c, Bin::Add, at, k);
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, at, shift))
    }

    /// The heap index of the descriptor at dword `first` of what the scalar
    /// load at `load` reads, computed in the current block.
    fn heap_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Smem {
            base,
            offset,
            soffset,
            ..
        } = &li.fields
        else {
            return Err("a descriptor load");
        };
        let Operand::Scalar { code, .. } = *base else {
            return Err("a descriptor load");
        };
        let c = self.cur;
        let base = li.read(Reg::S(code)).ok_or("a descriptor load")?;
        let mut addr = self.i32_of(base)?;
        let k = self
            .s
            .m
            .i32((*offset as u32).wrapping_add(4 * u32::from(first)));
        addr = self.s.m.bin(c, Bin::Add, addr, k);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(li, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, addr, shift))
    }

    fn valu(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if self.x.res.waterfall.compares.contains(&(b, i)) {
            // The first lane's dwords of a waterfall loop are each lane's
            // own (see `resource::Waterfall`).
            self.lane(inst, Some(true));
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("v_cmp_").or(name.strip_prefix("v_cmpx_")) {
            let r = if v.dpp.is_some() {
                Err("a compare with DPP")
            } else {
                self.compare(inst, rest)
            };
            if let Err(why) = r {
                for &(_, w) in &inst.writes {
                    self.lane_fail.insert(w, why);
                }
            }
            return r;
        }
        let mut guard = self.guard(b, i)?;
        if let Some(d) = v.dpp {
            let (x, enable) = self.dpp(inst, d)?;
            guard = match (guard, enable) {
                (Some(g), Some(e)) => Some(self.s.m.bin(self.cur, Bin::And, g, e)),
                (g, e) => g.or(e),
            };
            self.src0 = Some(x);
        }
        let r = self.valu_op(inst, b, i, guard);
        self.src0 = None;
        r
    }

    /// DPP: src0 as read from another lane, and which lanes write their
    /// result (none: all).
    fn dpp(&mut self, inst: &Inst, dpp: Dpp) -> Result<(V, Option<V>), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else {
            return Err("a DPP operand");
        };
        let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("a DPP operand")?)?;
        // Quad permutations and DPP8 stay in groups the pixel layout shares.
        let quad = matches!(dpp, Dpp::Row { ctrl: 0..=0xFF, .. });
        let mesh_quad = self.s.mesh.is_some() && quad;
        if !mesh_quad {
            self.cross_lane(quad)?;
        }
        let c = self.cur;
        let emulate = self.wave64 == Wave64Mode::Emulate && !mesh_quad;
        let lane = if emulate || self.s.mesh.is_some() {
            self.lane_index()
        } else {
            self.host_lane_index()
        };
        let mask_lane = if !emulate && self.s.mesh.is_none() && self.emulating_wave64() {
            self.lane_index()
        } else {
            lane
        };
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (sixteen_mask, low) = (k(self, !15), k(self, 15));
        let row = self.s.m.bin(c, Bin::And, lane, sixteen_mask);
        let pos = self.s.m.bin(c, Bin::And, lane, low);
        // Source lane, and whether it lies in the row.
        let (src, inside, masks, zero) = match dpp {
            Dpp::Lanes { sel, .. } => {
                let (seven, not7, three) = (k(self, 7), k(self, !7), k(self, 3));
                let p = self.s.m.bin(c, Bin::And, lane, seven);
                let sh = self.s.m.bin(c, Bin::Mul, p, three);
                let sel = k(self, sel);
                let pick = self.s.m.bin(c, Bin::LShr, sel, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, seven);
                let base = self.s.m.bin(c, Bin::And, lane, not7);
                (self.s.m.bin(c, Bin::Or, base, pick), None, None, false)
            }
            Dpp::Row {
                ctrl,
                row_mask,
                bank_mask,
                bound_ctrl,
                ..
            } => {
                let n = u32::from(ctrl & 15);
                let (src, inside) = match ctrl {
                    0..=0xFF => {
                        let (three, one, not3) = (k(self, 3), k(self, 1), k(self, !3));
                        let p = self.s.m.bin(c, Bin::And, lane, three);
                        let sh = self.s.m.bin(c, Bin::Shl, p, one);
                        let perm = k(self, u32::from(ctrl));
                        let pick = self.s.m.bin(c, Bin::LShr, perm, sh);
                        let pick = self.s.m.bin(c, Bin::And, pick, three);
                        let base = self.s.m.bin(c, Bin::And, lane, not3);
                        (self.s.m.bin(c, Bin::Or, base, pick), None)
                    }
                    0x101..=0x10F | 0x111..=0x11F => {
                        let nn = k(self, n);
                        let left = ctrl < 0x110;
                        let (op, p) = if left {
                            (Bin::Add, pred::IULT)
                        } else {
                            (Bin::Sub, pred::IUGE)
                        };
                        let s = self.s.m.bin(c, op, lane, nn);
                        // Left: pos + n < 16; right: pos >= n.
                        let bound = if left { k(self, 16 - n) } else { nn };
                        (s, Some(self.s.m.cmp(c, p, pos, bound)))
                    }
                    0x121..=0x12F => {
                        let nn = k(self, n);
                        let t = self.s.m.bin(c, Bin::Sub, pos, nn);
                        let t = self.s.m.bin(c, Bin::And, t, low);
                        (self.s.m.bin(c, Bin::Or, row, t), None)
                    }
                    0x140 | 0x141 => {
                        let m = k(self, if ctrl == 0x140 { 15 } else { 7 });
                        let t = self.s.m.bin(c, Bin::Xor, lane, m);
                        (t, None)
                    }
                    0x150..=0x15F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Or, row, nn), None)
                    }
                    0x160..=0x16F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Xor, lane, nn), None)
                    }
                    _ => return Err("a DPP lane pattern"),
                };
                let masks = (row_mask != 0xF || bank_mask != 0xF).then_some((row_mask, bank_mask));
                (src, inside, masks, bound_ctrl)
            }
        };
        let mut r = if mesh_quad {
            self.quad_pick(x, src)
        } else if emulate {
            self.exchange_64(x, src)?
        } else {
            self.read_lane(x, src)
        };
        let mut enable = None;
        if let Some(inside) = inside {
            if zero {
                let z = k(self, 0);
                r = self.s.m.select(c, inside, r, z);
            } else {
                enable = Some(inside);
            }
        }
        if let Some((rows, banks)) = masks {
            // Row (16 lanes) and bank (4 lanes of a row) write masks. When a
            // Wave64 shader runs at WaveSize(32), the row index (0..3) comes
            // from the console lane (`console_flat & 63`) so rows 2 and 3 still
            // select the upper half-wave.
            let (four, two, three) = (k(self, 4), k(self, 2), k(self, 3));
            let ri = self.s.m.bin(c, Bin::LShr, mask_lane, four);
            let ri = self.s.m.bin(c, Bin::And, ri, three);
            let bi = self.s.m.bin(c, Bin::LShr, mask_lane, two);
            let bi = self.s.m.bin(c, Bin::And, bi, three);
            let (rm, bm) = (k(self, u32::from(rows)), k(self, u32::from(banks)));
            let rb = self.s.m.bin(c, Bin::LShr, rm, ri);
            let bb = self.s.m.bin(c, Bin::LShr, bm, bi);
            let both = self.s.m.bin(c, Bin::And, rb, bb);
            let (one, z) = (k(self, 1), k(self, 0));
            let both = self.s.m.bin(c, Bin::And, both, one);
            let on = self.s.m.cmp(c, pred::NE, both, z);
            enable = Some(match enable {
                Some(e) => self.s.m.bin(c, Bin::And, e, on),
                None => on,
            });
        }
        Ok((r, enable))
    }

    fn valu_op(&mut self, inst: &Inst, b: usize, i: usize, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let c = self.cur;
        let f = |l: &mut Self, k| l.fsrc(inst, k);
        let s = |l: &mut Self, k| l.src(inst, k);
        let float = |l: &mut Self, r: V| l.write_f(inst, r, guard);
        let int = |l: &mut Self, r: V| l.write_i(inst, r, guard);
        if let Some(r) = self.valu16(inst, name, guard) {
            return r;
        }
        if let Some(op) = name.strip_prefix("v_pk_") {
            return self.packed16(inst, op, guard);
        }
        match name {
            "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => {
                let n = s(self, 0)?;
                let x = self.src64(inst, 1)?;
                let i64t = self.s.m.int(64);
                let k63 = self.s.m.i32(63);
                let n = self.s.m.bin(c, Bin::And, n, k63);
                let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                let op = match name {
                    "v_lshlrev_b64" => Bin::Shl,
                    "v_lshrrev_b64" => Bin::LShr,
                    _ => Bin::AShr,
                };
                let r = self.s.m.bin(c, op, x, n);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 64-bit result");
                };
                let (lo, hi) = self.split64(r);
                self.write(inst, Reg::V(reg), lo, guard);
                self.write(inst, Reg::V(reg + 1), hi, guard);
                Ok(())
            }
            "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => {
                // Cube-map face coordinates (AMD's ISA): the major axis is z
                // when |z| >= |x| and |z| >= |y|, else y when |y| >= |x|,
                // else x.
                let (x, y, z) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let (ax, ay, az) = (self.unary(6, x), self.unary(6, y), self.unary(6, z));
                let zx = self.s.m.cmp(c, pred::OGE, az, ax);
                let zy = self.s.m.cmp(c, pred::OGE, az, ay);
                let zmaj = self.s.m.bin(c, Bin::And, zx, zy);
                let ymaj = self.s.m.cmp(c, pred::OGE, ay, ax);
                let zero = self.s.m.f32(0);
                let neg = |l: &mut Self, v: V| l.s.m.cmp(c, pred::OLT, v, zero);
                let minus = |l: &mut Self, v: V| {
                    let z = l.s.m.f32(0x8000_0000);
                    l.s.m.bin(c, Bin::Sub, z, v)
                };
                let pick = |l: &mut Self, zv: V, yv: V, xv: V| {
                    let t = l.s.m.select(c, ymaj, yv, xv);
                    l.s.m.select(c, zmaj, zv, t)
                };
                let r = match name {
                    "v_cubeid_f32" => {
                        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
                        let (nz, ny, nx) = (neg(self, z), neg(self, y), neg(self, x));
                        let (f5, f4, f3, f2, f1, f0) = (
                            k(self, 5.0),
                            k(self, 4.0),
                            k(self, 3.0),
                            k(self, 2.0),
                            k(self, 1.0),
                            k(self, 0.0),
                        );
                        let zf = self.s.m.select(c, nz, f5, f4);
                        let yf = self.s.m.select(c, ny, f3, f2);
                        let xf = self.s.m.select(c, nx, f1, f0);
                        pick(self, zf, yf, xf)
                    }
                    "v_cubesc_f32" => {
                        let (nz, nx) = (neg(self, z), neg(self, x));
                        let (mx, mz) = (minus(self, x), minus(self, z));
                        let zf = self.s.m.select(c, nz, mx, x);
                        let xf = self.s.m.select(c, nx, z, mz);
                        pick(self, zf, x, xf)
                    }
                    "v_cubetc_f32" => {
                        let ny = neg(self, y);
                        let (my, mz) = (minus(self, y), minus(self, z));
                        let yf = self.s.m.select(c, ny, mz, z);
                        pick(self, my, yf, my)
                    }
                    _ => {
                        let t = pick(self, z, y, x);
                        let two = self.s.m.f32(2.0f32.to_bits());
                        self.s.m.bin(c, Bin::Mul, t, two)
                    }
                };
                float(self, r)
            }
            "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32" => {
                // M0 indexes the source (movrels), the destination (movreld)
                // or both, from the encoded register up to the program's
                // VGPR count (the registers the instruction reads or writes).
                let m0 = self.i32_of(inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?)?;
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("an M0-relative move");
                };
                let indexed_src = name != "v_movreld_b32";
                let value = if indexed_src {
                    let mut r = self.s.m.i32(0);
                    for k in 0..=255u8.saturating_sub(sreg) {
                        let Some(x) = inst.read(Reg::V(sreg + k)) else {
                            break;
                        };
                        let x = self.i32_of(x)?;
                        let kk = self.s.m.i32(u32::from(k));
                        let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                        r = self.s.m.select(c, is, x, r);
                    }
                    r
                } else {
                    s(self, 0)?
                };
                if name == "v_movrels_b32" {
                    return int(self, value);
                }
                for k in 0..=255u8.saturating_sub(dreg) {
                    let Some(_) = inst.write(Reg::V(dreg + k)) else {
                        break;
                    };
                    let old = match inst.read(Reg::V(dreg + k)) {
                        Some(o) => self.i32_of(o)?,
                        None => self.s.m.i32(0),
                    };
                    let kk = self.s.m.i32(u32::from(k));
                    let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                    let x = self.s.m.select(c, is, value, old);
                    self.write_placed(inst, Reg::V(dreg + k), x, guard);
                }
                Ok(())
            }
            "v_swap_b32" => {
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("v_swap_b32");
                };
                let a = self.i32_of(inst.read(Reg::V(dreg)).ok_or("v_swap_b32")?)?;
                let b = self.i32_of(inst.read(Reg::V(sreg)).ok_or("v_swap_b32")?)?;
                self.write_placed(inst, Reg::V(dreg), b, guard);
                if dreg != sreg {
                    self.write_placed(inst, Reg::V(sreg), a, guard);
                }
                Ok(())
            }
            "v_ffbh_u32" | "v_ffbh_i32" => {
                // Counted from the top bit, -1 when none (DXIL's too).
                let a = s(self, 0)?;
                let op = if name == "v_ffbh_u32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                int(self, r)
            }
            "v_mov_b32" => {
                if v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 {
                    let a = f(self, 0)?;
                    return float(self, a);
                }
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32"
            | "v_max_f32" | "v_min_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let r = match name {
                    "v_add_f32" => self.s.m.bin(c, Bin::Add, a, bb),
                    "v_sub_f32" => self.s.m.bin(c, Bin::Sub, a, bb),
                    "v_subrev_f32" => self.s.m.bin(c, Bin::Sub, bb, a),
                    "v_mul_f32" => self.s.m.bin(c, Bin::Mul, a, bb),
                    "v_mul_legacy_f32" => self.legacy_mul(a, bb),
                    "v_max_f32" => self.max_min(true, a, bb),
                    _ => self.max_min(false, a, bb),
                };
                float(self, r)
            }
            "v_fma_f32" | "v_fmaak_f32" | "v_fmamk_f32" => {
                // shortcut: DXIL has no fused f32 multiply-add; FMad lets the driver fuse.
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = self.fmad(a, bb, cc);
                float(self, r)
            }
            "v_fmac_f32" | "v_mac_f32" | "v_mac_legacy_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let acc = inst.read(Reg::V(reg)).ok_or("an accumulator")?;
                let acc = match self.floats.get(&acc) {
                    Some(&x) => x,
                    None => {
                        let x = self.i32_of(acc)?;
                        self.as_float(x)
                    }
                };
                let r = match name {
                    "v_fmac_f32" => self.fmad(a, bb, acc),
                    "v_mac_f32" => {
                        let m = self.s.m.bin(c, Bin::Mul, a, bb);
                        self.s.m.bin(c, Bin::Add, m, acc)
                    }
                    _ => {
                        let m = self.legacy_mul(a, bb);
                        self.s.m.bin(c, Bin::Add, m, acc)
                    }
                };
                float(self, r)
            }
            "v_mad_f32" | "v_madak_f32" | "v_madmk_f32" | "v_mad_legacy_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let m = if name == "v_mad_legacy_f32" {
                    self.legacy_mul(a, bb)
                } else {
                    self.s.m.bin(c, Bin::Mul, a, bb)
                };
                let r = self.s.m.bin(c, Bin::Add, m, cc);
                float(self, r)
            }
            "v_max3_f32" | "v_min3_f32" | "v_med3_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = match name {
                    "v_max3_f32" => {
                        let t = self.max_min(true, a, bb);
                        self.max_min(true, t, cc)
                    }
                    "v_min3_f32" => {
                        let t = self.max_min(false, a, bb);
                        self.max_min(false, t, cc)
                    }
                    _ => self.med3(a, bb, cc),
                };
                float(self, r)
            }
            "v_rcp_f32" | "v_rcp_iflag_f32" => {
                // shortcut: the hardware reciprocal is approximate (1 ULP); this divides.
                let a = f(self, 0)?;
                let one = self.s.m.f32(1.0f32.to_bits());
                let r = self.s.m.bin(c, Bin::SDiv, one, a);
                float(self, r)
            }
            "v_rsq_f32" | "v_sqrt_f32" | "v_log_f32" | "v_exp_f32" | "v_fract_f32"
            | "v_trunc_f32" | "v_floor_f32" | "v_ceil_f32" | "v_rndne_f32" | "v_sin_f32"
            | "v_cos_f32" => {
                let a = f(self, 0)?;
                let op = match name {
                    "v_rsq_f32" => 25,
                    "v_sqrt_f32" => 24,
                    "v_log_f32" => 23,
                    "v_exp_f32" => 21,
                    "v_fract_f32" => 22,
                    "v_trunc_f32" => 29,
                    "v_floor_f32" => 27,
                    "v_ceil_f32" => 28,
                    "v_rndne_f32" => 26,
                    "v_sin_f32" => 13,
                    _ => 12,
                };
                let r = match op {
                    12 | 13 => self.trig_turns(a, op == 12),
                    22 => self.fract(a, 1.0, f32::from_bits(0x3F7F_FFFF)),
                    _ => self.unary(op, a),
                };
                float(self, r)
            }
            "v_cvt_i32_f32" | "v_cvt_u32_f32" | "v_cvt_flr_i32_f32" | "v_cvt_rpi_i32_f32" => {
                let mut a = f(self, 0)?;
                if name == "v_cvt_rpi_i32_f32" {
                    let h = self.s.m.f32(0.5f32.to_bits());
                    a = self.s.m.bin(c, Bin::Add, a, h);
                }
                if name != "v_cvt_i32_f32" && name != "v_cvt_u32_f32" {
                    a = self.unary(27, a);
                }
                // D3D's conversions saturate and give 0 for NaN, as the
                // hardware does (checked on the RX 6700 XT and WARP).
                let op = if name == "v_cvt_u32_f32" {
                    Cast::FpToUi
                } else {
                    Cast::FpToSi
                };
                let r = self.s.m.cast(c, op, a, self.s.t.i32);
                int(self, r)
            }
            "v_cvt_f32_i32" | "v_cvt_f32_u32" | "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1"
            | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => {
                let mut a = s(self, 0)?;
                if let Some(n) = name.strip_prefix("v_cvt_f32_ubyte") {
                    let sh = self.s.m.i32(8 * u32::from(n.as_bytes()[0] - b'0'));
                    let ff = self.s.m.i32(0xff);
                    a = self.s.m.bin(c, Bin::LShr, a, sh);
                    a = self.s.m.bin(c, Bin::And, a, ff);
                }
                let op = if name == "v_cvt_f32_i32" {
                    Cast::SiToFp
                } else {
                    Cast::UiToFp
                };
                let r = self.s.m.cast(c, op, a, self.s.t.f32);
                float(self, r)
            }
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                // Each source is f32, or with op_sel_hi the low (op_sel: high)
                // f16 half; neg_hi is the absolute value here.
                let mut xs = Vec::new();
                for i in 0..3 {
                    let op = v.src.get(i).ok_or("a missing operand")?;
                    let x = self.operand(inst, op)?;
                    let mut x = if v.opsel_hi >> i & 1 != 0 {
                        let h = if v.opsel >> i & 1 != 0 {
                            let k16 = self.s.m.i32(16);
                            self.s.m.bin(c, Bin::LShr, x, k16)
                        } else {
                            x
                        };
                        self.s
                            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h])
                    } else {
                        self.as_float(x)
                    };
                    if v.neg_hi >> i & 1 != 0 {
                        x = self.unary(6, x);
                    }
                    if v.neg >> i & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        x = self.s.m.bin(c, Bin::Sub, z, x);
                    }
                    xs.push(x);
                }
                let mut r = self.fmad(xs[0], xs[1], xs[2]);
                if name == "v_fma_mix_f32" {
                    return float(self, r);
                }
                if v.clamp {
                    r = self.unary(7, r);
                }
                // The f16 result goes to one half; the other half stays.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a mixed-precision result");
                };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.i32_of(o)?,
                    None => self.s.m.i32(0),
                };
                let h = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[r]);
                let (keep, h) = if name == "v_fma_mixhi_f16" {
                    let k16 = self.s.m.i32(16);
                    (0xFFFF, self.s.m.bin(c, Bin::Shl, h, k16))
                } else {
                    (0xFFFF_0000, h)
                };
                let keep = self.s.m.i32(keep);
                let old = self.s.m.bin(c, Bin::And, old, keep);
                let x = self.s.m.bin(c, Bin::Or, old, h);
                self.write(inst, Reg::V(reg), x, guard);
                Ok(())
            }
            "v_cvt_pkrtz_f16_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let lo = self.f16_toward_zero(a);
                let hi = self.f16_toward_zero(bb);
                let sh = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, hi, sh);
                let r = self.s.m.bin(c, Bin::Or, lo, hi);
                int(self, r)
            }
            "v_cvt_f16_f32" if v.opsel == 0 => {
                // Rounded to nearest even (the MODE default) into the low half.
                let a = f(self, 0)?;
                let r = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[a]);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 16-bit result outside a VGPR");
                };
                let r = self.low_half(inst, reg, r);
                int(self, r)
            }
            "v_pack_b32_f16" => {
                // Two f16 sources (op_sel: their high halves; abs and neg on
                // the sign bit) into the low and high half.
                let mut halves = Vec::new();
                for k in 0..2 {
                    let op = v.src.get(k).ok_or("a missing operand")?;
                    let mut x = self.operand(inst, op)?;
                    if v.opsel >> k & 1 != 0 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let mask = if v.abs >> k & 1 != 0 { 0x7FFF } else { 0xFFFF };
                    let mask = self.s.m.i32(mask);
                    x = self.s.m.bin(c, Bin::And, x, mask);
                    if v.neg >> k & 1 != 0 {
                        let sign = self.s.m.i32(0x8000);
                        x = self.s.m.bin(c, Bin::Xor, x, sign);
                    }
                    halves.push(x);
                }
                let k16 = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
                let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
                int(self, r)
            }
            "v_cvt_f32_f16" if v.opsel == 0 => {
                let a = s(self, 0)?;
                let r = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[a]);
                float(self, r)
            }
            "v_cndmask_b32" => {
                let cond = self.select_condition(inst, v.src.get(2))?;
                if v.abs != 0 || v.neg != 0 {
                    let (a, bb) = (f(self, 0)?, f(self, 1)?);
                    let r = self.s.m.select(c, cond, bb, a);
                    return float(self, r);
                }
                let (a, bb) = (s(self, 0)?, s(self, 1)?);
                let r = self.s.m.select(c, cond, bb, a);
                int(self, r)
            }
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32"
            | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => {
                // The carry (borrow) out is a lane mask: this lane's bit.
                let (mut a, mut bb) = (s(self, 0)?, s(self, 1)?);
                if name.contains("subrev") {
                    std::mem::swap(&mut a, &mut bb);
                }
                let cin = if name.ends_with("_ci_u32") {
                    let m = match v.src.get(2) {
                        Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)),
                        _ => None,
                    };
                    let m = m
                        .and_then(|m| self.x.st.masks.get(&m).copied())
                        .ok_or("a carry-in from an untracked mask")?;
                    Some(self.boolean(m, c)?)
                } else {
                    None
                };
                let (r, carry) = self.add_sub_carry(c, name.starts_with("v_add"), a, bb, cin);
                int(self, r)?;
                if let Some(Operand::Scalar { code, .. }) = v.carry
                    && let Some(w) = inst.write(Reg::S(code))
                {
                    self.lane_bits.insert(w, carry);
                }
                for &(_, w) in &inst.writes {
                    self.vals
                        .entry(w)
                        .or_insert(Val::Missing("a lane mask read as data"));
                }
                Ok(())
            }
            "v_readlane_b32" => {
                if let Some(&spill) = self.resolved_reads.get(&(b, i)) {
                    let r = match spill {
                        Spill::Value(val) => self.i32_of(val)?,
                        Spill::Const(k) => self.s.m.i32(k),
                    };
                    return int(self, r);
                }
                if let Some(&moved) = self.spills.get(&(b, i)) {
                    let r = match moved {
                        Moved::Const(k) => self.s.m.i32(k),
                        Moved::Value(sv) => self.i32_of(sv)?,
                    };
                    return int(self, r);
                }
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                if self.emulating_wave64() {
                    self.cross_lane(false)?;
                    let r = self.exchange_64(a, lane)?;
                    return int(self, r);
                }
                if self.wave64 == Wave64Mode::Wave32
                    && self.total_threads <= 32
                    && let Some(op_lane) = v.src.get(1).and_then(operand_const_lane)
                    && op_lane >= 32
                {
                    return Err("a lane >= 32 read in a <=32-thread dispatch");
                }
                // A computed lane in a pixel or vertex program is a host lane
                // (from a mask or `v_mbcnt`); a constant one is the console's.
                if self.mask_data == MaskData::Host && !matches!(lane, V::Const(_)) {
                    self.s.flags |= flags::WAVE_OPS;
                } else {
                    self.cross_lane(false)?;
                }
                let r = self.read_lane(a, lane);
                int(self, r)
            }
            "v_writelane_b32" => {
                if self.pure_spill_writes.contains(&(b, i)) {
                    let zero = self.s.m.i32(0);
                    return int(self, zero);
                }
                // The lane given takes the scalar; the others keep their value.
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a lane write");
                };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.i32_of(o)?,
                    None => self.s.m.i32(0),
                };
                self.cross_lane(false)?;
                let me = self.lane_index();
                let is = self.s.m.cmp(c, pred::EQ, me, lane);
                let r = self.s.m.select(c, is, a, old);
                int(self, r)
            }
            "v_permlanex16_b32" => {
                // Each lane reads src0 of a lane of the *other* row of 16 in
                // its 32-lane half; lane i's pick is nibble i of the {src2,
                // src1} selector (v_permlane16_b32 reads the same row). Both
                // stay inside a 32-lane half, so the lane math is unchanged
                // when a console wave64 program runs as two wave32 halves.
                if v.opsel != 0 {
                    return Err("op_sel on v_permlanex16_b32");
                }
                let (a, lo, hi) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                self.cross_lane(false)?;
                let lane = self.lane_index();
                let (k2, k8, k15, k16, k31, n31) = (
                    self.s.m.i32(2),
                    self.s.m.i32(8),
                    self.s.m.i32(15),
                    self.s.m.i32(16),
                    self.s.m.i32(31),
                    self.s.m.i32(!31),
                );
                let nib = self.s.m.bin(c, Bin::And, lane, k15);
                let sh = self.s.m.bin(c, Bin::Shl, nib, k2);
                let sh = self.s.m.bin(c, Bin::And, sh, k31);
                let top = self.s.m.cmp(c, pred::IUGE, nib, k8);
                let word = self.s.m.select(c, top, hi, lo);
                let pick = self.s.m.bin(c, Bin::LShr, word, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, k15);
                let half = self.s.m.bin(c, Bin::And, lane, n31);
                let row = self.s.m.bin(c, Bin::And, lane, k16);
                let row = self.s.m.bin(c, Bin::Xor, row, k16);
                let base = self.s.m.bin(c, Bin::Or, half, row);
                let idx = self.s.m.bin(c, Bin::Or, base, pick);
                let val = if self.emulating_wave64() {
                    self.exchange_64(a, idx)?
                } else {
                    self.read_lane(a, idx)
                };
                int(self, val)
            }
            "v_readfirstlane_b32"
                if inst
                    .writes
                    .first()
                    .is_some_and(|w| self.x.res.waterfall.reads.contains(&w.1)) =>
            {
                // A waterfall loop's read of a descriptor dword: the lane's own.
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_readfirstlane_b32" => {
                let a = s(self, 0)?;
                let non_uniform = v.src.first().is_some_and(|op| match *op {
                    Operand::Vector { reg, .. } => inst.read(Reg::V(reg)).is_some_and(|val| {
                        !self
                            .val_uniform
                            .get(val.0 as usize)
                            .copied()
                            .unwrap_or(false)
                    }),
                    _ => false,
                });
                if self.emulating_wave64() && (self.console_lanes || non_uniform) {
                    self.cross_lane(false)?;
                    let r = self.emulated_readfirstlane(a, guard)?;
                    return int(self, r);
                }
                self.s.flags |= flags::WAVE_OPS;
                let r = self.s.op(
                    c,
                    "waveReadLaneFirst.i32",
                    Attr::NoUnwind,
                    self.s.t.i32,
                    118,
                    &[a],
                );
                int(self, r)
            }
            _ => {
                let r = self.integer(inst, name)?;
                int(self, r)
            }
        }
    }

    /// A 64-bit source (a register pair; inline constants sign-extend).
    fn src64(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit operand modifiers");
        }
        let c = self.cur;
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let pair = |l: &mut Self, r: Reg, r1: Reg| -> Result<V, Why> {
            let lo = l.i32_of(inst.read(r).ok_or("an unread operand")?)?;
            let hi = l.i32_of(inst.read(r1).ok_or("an unread operand")?)?;
            Ok(l.join64(lo, hi))
        };
        match op {
            Operand::Vector { reg, .. } => pair(self, Reg::V(reg), Reg::V(reg + 1)),
            Operand::Scalar { code, .. } if code != NULL => {
                pair(self, Reg::S(code), Reg::S(code + 1))
            }
            Operand::Const(k @ 128..=208) => {
                let x = crate::resource::inline(k).ok_or("a special operand")?;
                Ok(self.s.m.int_const(i64t, i64::from(x as i32)))
            }
            // A signed 64-bit operand sign-extends its 32-bit literal, as
            // the hardware does (LLVM's AMDGPU operand rules).
            Operand::Literal(x) if inst.name().ends_with("_i64") => {
                Ok(self.s.m.int_const(i64t, i64::from(x as i32)))
            }
            _ => {
                // Literals (of unsigned and bit operands) and the null
                // register: zero-extended.
                let x = self.operand(inst, &op)?;
                Ok(self.s.m.cast(c, Cast::ZExt, x, i64t))
            }
        }
    }

    /// The low and high dwords of an i64.
    fn split64(&mut self, x: V) -> (V, V) {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let k32 = self.s.m.int_const(i64t, 32);
        let lo = self.s.m.cast(c, Cast::Trunc, x, self.s.t.i32);
        let hi = self.s.m.bin(c, Bin::LShr, x, k32);
        let hi = self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32);
        (lo, hi)
    }

    /// Source `k` of a 16-bit operation in the low 16 bits (the rest 0):
    /// op_sel (or an SDWA word select) picks the high half; inline float
    /// constants are f16 values here.
    fn half_src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let mut high = v.opsel >> k & 1 != 0;
        if let Some(sd) = v.sdwa {
            match sd.src_sel.get(k).copied().unwrap_or(6) {
                4 => {}
                5 => high = true,
                6 => {}
                _ => return Err("a byte select on a 16-bit operation"),
            }
        }
        let x = match (op, self.src0) {
            (_, Some(x)) if k == 0 => x,
            (Operand::Const(c @ 240..=248), _) => {
                let h = [
                    0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
                ][usize::from(c - 240)];
                self.s.m.i32(h)
            }
            _ => self.operand(inst, &op)?,
        };
        let c = self.cur;
        let x = if high {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// An f16 source as an f32 (exact), with its abs/neg modifiers.
    fn half_fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let h = self.half_src(inst, k)?;
        let f32t = self.s.t.f32;
        let mut f = self
            .s
            .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    /// Writes a 16-bit result (low bits of `x`) into the destination's low
    /// half, or its high half with op_sel bit 3 (or an SDWA WORD_1 select);
    /// the other half is kept (GFX10).
    fn write_half(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a 16-bit result outside a VGPR");
        };
        let mut high = v.opsel >> 3 & 1 != 0;
        // SDWA: the word written, the other one kept (preserve) or zero
        // (pad; a sign extension above the high word adds nothing).
        let mut keep = true;
        if let Some(sd) = v.sdwa {
            match (sd.dst_sel, sd.dst_unused) {
                (4, 2) => {}
                (5, 2) => high = true,
                (4 | 6, _) => keep = false,
                (5, _) => (high, keep) = (true, false),
                _ => return Err("an SDWA destination of a 16-bit operation"),
            }
        }
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) if keep => o,
            _ => self.s.m.i32(0),
        };
        let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
        let x = self.s.m.bin(c, Bin::And, x, low);
        let r = if high {
            let o = self.s.m.bin(c, Bin::And, old, low);
            let x = self.s.m.bin(c, Bin::Shl, x, k16);
            self.s.m.bin(c, Bin::Or, o, x)
        } else {
            let keep = self.s.m.i32(0xFFFF_0000);
            let o = self.s.m.bin(c, Bin::And, old, keep);
            self.s.m.bin(c, Bin::Or, o, x)
        };
        self.write_placed(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    /// 16-bit operations, computed in f32 or i32: f16 sums, differences and
    /// products are exact there and round once to f16 (f32 has the 2 x 11 +
    /// 2 bits that make double rounding harmless); `None` if `name` is none.
    fn valu16(&mut self, inst: &Inst, name: &str, guard: Option<V>) -> Option<Result<(), Why>> {
        const FLOAT2: &[&str] = &[
            "v_add_f16",
            "v_sub_f16",
            "v_subrev_f16",
            "v_mul_f16",
            "v_max_f16",
            "v_min_f16",
        ];
        const FLOAT3: &[&str] = &[
            "v_fma_f16",
            "v_mad_f16",
            "v_fmac_f16",
            "v_mac_f16",
            "v_max3_f16",
            "v_min3_f16",
            "v_med3_f16",
        ];
        const FLOAT1: &[&str] = &[
            "v_rcp_f16",
            "v_sqrt_f16",
            "v_rsq_f16",
            "v_log_f16",
            "v_exp_f16",
            "v_fract_f16",
            "v_trunc_f16",
            "v_floor_f16",
            "v_ceil_f16",
            "v_rndne_f16",
            "v_sin_f16",
            "v_cos_f16",
            "v_cvt_f16_u16",
            "v_cvt_f16_i16",
        ];
        const INT: &[&str] = &[
            "v_add_nc_u16",
            "v_sub_nc_u16",
            "v_subrev_nc_u16",
            "v_mul_lo_u16",
            "v_lshlrev_b16",
            "v_lshrrev_b16",
            "v_ashrrev_i16",
            "v_max_u16",
            "v_max_i16",
            "v_min_u16",
            "v_min_i16",
            "v_cvt_u16_f16",
            "v_cvt_i16_f16",
        ];
        let all = [FLOAT1, FLOAT2, FLOAT3, INT];
        if !all.iter().any(|l| l.contains(&name))
            && !matches!(name, "v_cvt_f16_f32" | "v_cvt_f32_f16")
        {
            return None;
        }
        Some(self.valu16_op(inst, name, guard, FLOAT1, FLOAT2, FLOAT3))
    }

    #[allow(clippy::too_many_arguments)]
    fn valu16_op(
        &mut self,
        inst: &Inst,
        name: &str,
        guard: Option<V>,
        float1: &[&str],
        float2: &[&str],
        float3: &[&str],
    ) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let i32t = self.s.t.i32;
        let f32t = self.s.t.f32;
        if name == "v_cvt_f32_f16" {
            let f = self.half_fsrc(inst, 0)?;
            return self.write_f(inst, f, guard);
        }
        let float_result = |l: &mut Self, mut f: V| -> Result<(), Why> {
            if v.omod != 0 {
                return Err("an output modifier on a 16-bit operation");
            }
            if v.clamp {
                f = l.unary(7, f);
            }
            let h =
                l.s.op(l.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
            l.write_half(inst, h, guard)
        };
        if name == "v_cvt_f16_f32" {
            let f = self.fsrc(inst, 0)?;
            return float_result(self, f);
        }
        if float2.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let r = match name {
                "v_add_f16" => self.s.m.bin(c, Bin::Add, a, b),
                "v_sub_f16" => self.s.m.bin(c, Bin::Sub, a, b),
                "v_subrev_f16" => self.s.m.bin(c, Bin::Sub, b, a),
                "v_mul_f16" => self.s.m.bin(c, Bin::Mul, a, b),
                "v_max_f16" => self.max_min(true, a, b),
                _ => self.max_min(false, a, b),
            };
            return float_result(self, r);
        }
        if float3.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let acc = if matches!(name, "v_fmac_f16" | "v_mac_f16") {
                // The accumulator is the destination's half that is written.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("an accumulator")?)?;
                let x = if v.opsel >> 3 & 1 != 0 {
                    let k16 = self.s.m.i32(16);
                    self.s.m.bin(c, Bin::LShr, x, k16)
                } else {
                    x
                };
                let m = self.s.m.i32(0xFFFF);
                let x = self.s.m.bin(c, Bin::And, x, m);
                self.s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
            } else {
                self.half_fsrc(inst, 2)?
            };
            let r = match name {
                // shortcut: fused f16 multiply-adds as FMad in f32 (the
                // product is exact; the sum rounds twice).
                "v_fma_f16" | "v_fmac_f16" | "v_mad_f16" | "v_mac_f16" => self.fmad(a, b, acc),
                "v_max3_f16" => {
                    let t = self.max_min(true, a, b);
                    self.max_min(true, t, acc)
                }
                "v_min3_f16" => {
                    let t = self.max_min(false, a, b);
                    self.max_min(false, t, acc)
                }
                _ => self.med3(a, b, acc),
            };
            return float_result(self, r);
        }
        if float1.contains(&name) {
            let r = match name {
                "v_cvt_f16_u16" | "v_cvt_f16_i16" => {
                    let mut x = self.half_src(inst, 0)?;
                    let op = if name == "v_cvt_f16_i16" {
                        let k16 = self.s.m.i32(16);
                        let t = self.s.m.bin(c, Bin::Shl, x, k16);
                        x = self.s.m.bin(c, Bin::AShr, t, k16);
                        Cast::SiToFp
                    } else {
                        Cast::UiToFp
                    };
                    self.s.m.cast(c, op, x, f32t)
                }
                "v_rcp_f16" => {
                    let a = self.half_fsrc(inst, 0)?;
                    let one = self.s.m.f32(1.0f32.to_bits());
                    self.s.m.bin(c, Bin::SDiv, one, a)
                }
                _ => {
                    let a = self.half_fsrc(inst, 0)?;
                    let op = match name {
                        "v_sqrt_f16" => 24,
                        "v_rsq_f16" => 25,
                        "v_log_f16" => 23,
                        "v_exp_f16" => 21,
                        "v_fract_f16" => 22,
                        "v_trunc_f16" => 29,
                        "v_floor_f16" => 27,
                        "v_ceil_f16" => 28,
                        "v_rndne_f16" => 26,
                        "v_sin_f16" => 13,
                        _ => 12,
                    };
                    match op {
                        12 | 13 => self.trig_turns(a, op == 12),
                        // Any f16 in (-2^-11, 0) has fract >= 1 - 2^-12; clamp to 0x3bff (1 - 2^-11).
                        22 => self.fract(a, 1.0 - 1.0 / 4096.0, 1.0 - 1.0 / 2048.0),
                        _ => self.unary(op, a),
                    }
                }
            };
            return float_result(self, r);
        }
        // 16-bit integers: computed on the low halves, the low 16 bits kept.
        if v.clamp || v.omod != 0 || v.abs != 0 || v.neg != 0 {
            return Err("modifiers on a 16-bit integer operation");
        }
        let r = match name {
            "v_cvt_u16_f16" | "v_cvt_i16_f16" => {
                // shortcut: converted through 32 bits, so out-of-range values
                // wrap instead of saturating at 16 bits.
                let a = self.half_fsrc(inst, 0)?;
                let op = if name == "v_cvt_u16_f16" {
                    Cast::FpToUi
                } else {
                    Cast::FpToSi
                };
                self.s.m.cast(c, op, a, i32t)
            }
            _ => {
                let (mut a, mut b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if name.ends_with("_i16") {
                    a = sext(self, a);
                    b = sext(self, b);
                }
                let k15 = self.s.m.i32(15);
                match name {
                    "v_add_nc_u16" => self.s.m.bin(c, Bin::Add, a, b),
                    "v_sub_nc_u16" => self.s.m.bin(c, Bin::Sub, a, b),
                    "v_subrev_nc_u16" => self.s.m.bin(c, Bin::Sub, b, a),
                    "v_mul_lo_u16" => self.s.m.bin(c, Bin::Mul, a, b),
                    "v_lshlrev_b16" | "v_lshrrev_b16" | "v_ashrrev_i16" => {
                        // src0 is the shift (4 bits), src1 the value.
                        let sh = self.s.m.bin(c, Bin::And, a, k15);
                        let x = if name == "v_ashrrev_i16" {
                            sext(self, b)
                        } else {
                            b
                        };
                        let op = match name {
                            "v_lshlrev_b16" => Bin::Shl,
                            "v_lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        self.s.m.bin(c, op, x, sh)
                    }
                    _ => {
                        let p = match name {
                            "v_max_u16" => pred::IUGT,
                            "v_min_u16" => pred::IULT,
                            "v_max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, a, b);
                        self.s.m.select(c, t, a, b)
                    }
                }
            }
        };
        self.write_half(inst, r, guard)
    }

    /// Half `high` of source `k` of a packed (VOP3P) operation, in the low
    /// 16 bits: op_sel picks the half the low result reads, op_sel_hi the
    /// half the high result reads. Inline float constants are f16 values.
    fn packed_src(&mut self, inst: &Inst, k: usize, high: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let sel = if high { v.opsel_hi } else { v.opsel } >> k & 1 != 0;
        let x = match op {
            Operand::Const(c @ 240..=248) => {
                let h = [
                    0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
                ][usize::from(c - 240)];
                // shortcut: taken to fill both halves.
                self.s.m.i32(h | h << 16)
            }
            _ => self.operand(inst, &op)?,
        };
        let c = self.cur;
        let x = if sel {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// Packed 16-bit operations (`v_pk_*`): each half computed as the
    /// 16-bit operation does, f16 through f32 as in [`Self::valu16`].
    fn packed16(&mut self, inst: &Inst, op: &str, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let float = op.ends_with("_f16");
        let n = match op {
            "fma_f16" | "mad_u16" | "mad_i16" => 3,
            "add_f16" | "mul_f16" | "min_f16" | "max_f16" | "add_u16" | "sub_u16" | "add_i16"
            | "sub_i16" | "mul_lo_u16" | "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16"
            | "max_u16" | "min_u16" | "max_i16" | "min_i16" => 2,
            _ => return Err("a packed 16-bit operation"),
        };
        if v.omod != 0 || !float && (v.clamp || v.neg != 0 || v.neg_hi != 0) {
            return Err("modifiers on a packed integer operation");
        }
        let c = self.cur;
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let mut halves = Vec::new();
        for high in [false, true] {
            let mut xs = Vec::new();
            for k in 0..n {
                let h = self.packed_src(inst, k, high)?;
                if float {
                    let mut f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    let neg = if high { v.neg_hi } else { v.neg };
                    if neg >> k & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        f = self.s.m.bin(c, Bin::Sub, z, f);
                    }
                    xs.push(f);
                } else {
                    xs.push(h);
                }
            }
            let r = if float {
                let mut f = match op {
                    "add_f16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "mul_f16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "min_f16" => self.max_min(false, xs[0], xs[1]),
                    "max_f16" => self.max_min(true, xs[0], xs[1]),
                    // shortcut: FMad in f32, as v_fma_f16.
                    _ => self.fmad(xs[0], xs[1], xs[2]),
                };
                if v.clamp {
                    f = self.unary(7, f);
                }
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
            } else {
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if op.ends_with("_i16") {
                    for x in &mut xs {
                        *x = sext(self, *x);
                    }
                }
                let k15 = self.s.m.i32(15);
                match op {
                    "add_u16" | "add_i16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "sub_u16" | "sub_i16" => self.s.m.bin(c, Bin::Sub, xs[0], xs[1]),
                    "mul_lo_u16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "mad_u16" | "mad_i16" => {
                        let m = self.s.m.bin(c, Bin::Mul, xs[0], xs[1]);
                        self.s.m.bin(c, Bin::Add, m, xs[2])
                    }
                    "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16" => {
                        let sh = self.s.m.bin(c, Bin::And, xs[0], k15);
                        let o = match op {
                            "lshlrev_b16" => Bin::Shl,
                            "lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        let x = if op == "ashrrev_i16" {
                            sext(self, xs[1])
                        } else {
                            xs[1]
                        };
                        self.s.m.bin(c, o, x, sh)
                    }
                    _ => {
                        let p = match op {
                            "max_u16" => pred::IUGT,
                            "min_u16" => pred::IULT,
                            "max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, xs[0], xs[1]);
                        self.s.m.select(c, t, xs[0], xs[1])
                    }
                }
            };
            let m = self.s.m.i32(0xFFFF);
            halves.push(self.s.m.bin(c, Bin::And, r, m));
        }
        let k16 = self.s.m.i32(16);
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a packed result outside a VGPR");
        };
        self.write(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    fn integer(&mut self, inst: &Inst, name: &'static str) -> Result<V, Why> {
        let c = self.cur;
        let s = |l: &mut Self, k| l.src(inst, k);
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let m31 = |l: &mut Self, x: V| {
            let k31 = l.s.m.i32(31);
            l.s.m.bin(c, Bin::And, x, k31)
        };
        Ok(match name {
            "v_and_b32" | "v_or_b32" | "v_xor_b32" | "v_add_nc_u32" | "v_sub_nc_u32"
            | "v_subrev_nc_u32" | "v_mul_lo_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (op, a, b) = match name {
                    "v_and_b32" => (Bin::And, a, b),
                    "v_or_b32" => (Bin::Or, a, b),
                    "v_xor_b32" => (Bin::Xor, a, b),
                    "v_add_nc_u32" => (Bin::Add, a, b),
                    "v_sub_nc_u32" => (Bin::Sub, a, b),
                    "v_subrev_nc_u32" => (Bin::Sub, b, a),
                    _ => (Bin::Mul, a, b),
                };
                self.s.m.bin(c, op, a, b)
            }
            "v_xnor_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let x = self.s.m.bin(c, Bin::Xor, a, b);
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, x, ones)
            }
            "v_not_b32" => {
                let a = s(self, 0)?;
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, a, ones)
            }
            "v_lshlrev_b32" | "v_lshrrev_b32" | "v_ashrrev_i32" => {
                let (sh, a) = (s(self, 0)?, s(self, 1)?);
                let sh = m31(self, sh);
                let op = match name {
                    "v_lshlrev_b32" => Bin::Shl,
                    "v_lshrrev_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                self.s.m.bin(c, op, a, sh)
            }
            "v_lshl_add_u32" | "v_lshl_or_b32" | "v_add_lshl_u32" | "v_and_or_b32"
            | "v_or3_b32" | "v_xad_u32" | "v_add3_u32" | "v_xor3_b32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                match name {
                    "v_lshl_add_u32" | "v_lshl_or_b32" => {
                        let b = m31(self, b);
                        let t = self.s.m.bin(c, Bin::Shl, a, b);
                        let op = if name == "v_lshl_add_u32" {
                            Bin::Add
                        } else {
                            Bin::Or
                        };
                        self.s.m.bin(c, op, t, d)
                    }
                    "v_add_lshl_u32" => {
                        let t = self.s.m.bin(c, Bin::Add, a, b);
                        let d = m31(self, d);
                        self.s.m.bin(c, Bin::Shl, t, d)
                    }
                    _ => {
                        let (o1, o2) = match name {
                            "v_and_or_b32" => (Bin::And, Bin::Or),
                            "v_or3_b32" => (Bin::Or, Bin::Or),
                            "v_xad_u32" => (Bin::Xor, Bin::Add),
                            "v_add3_u32" => (Bin::Add, Bin::Add),
                            _ => (Bin::Xor, Bin::Xor),
                        };
                        let t = self.s.m.bin(c, o1, a, b);
                        self.s.m.bin(c, o2, t, d)
                    }
                }
            }
            "v_mul_u32_u24" | "v_mul_i32_i24" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (sh, op) = (
                    k(self, 8),
                    if name == "v_mul_i32_i24" {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                self.s.m.bin(c, Bin::Mul, a, b)
            }
            "v_bfe_u32" | "v_bfe_i32" => {
                let (a, off, w) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let w = m31(self, w);
                self.bfe(a, off, w, name == "v_bfe_i32")
            }
            "v_bfi_b32" => {
                let (m, a, b) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let ones = k(self, u32::MAX);
                let nm = self.s.m.bin(c, Bin::Xor, m, ones);
                let x = self.s.m.bin(c, Bin::And, m, a);
                let y = self.s.m.bin(c, Bin::And, nm, b);
                self.s.m.bin(c, Bin::Or, x, y)
            }
            "v_max_i32" | "v_min_i32" | "v_max_u32" | "v_min_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let p = match name {
                    "v_max_i32" => pred::SGT,
                    "v_min_i32" => pred::SLT,
                    "v_max_u32" => pred::IUGT,
                    _ => pred::IULT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                self.s.m.select(c, t, a, b)
            }
            "v_mul_hi_u32" | "v_mul_hi_i32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                self.mul_hi(a, b, name == "v_mul_hi_i32")
            }
            "v_alignbit_b32" | "v_alignbyte_b32" => {
                // The low dword of {a, b} >> shift.
                let (a, b, sh) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let sh = if name == "v_alignbyte_b32" {
                    let three = k(self, 3);
                    let t = self.s.m.bin(c, Bin::And, sh, three);
                    let k3 = k(self, 3);
                    self.s.m.bin(c, Bin::Shl, t, k3)
                } else {
                    m31(self, sh)
                };
                let k32 = k(self, 32);
                let lo = self.s.m.bin(c, Bin::LShr, b, sh);
                let left = self.s.m.bin(c, Bin::Sub, k32, sh);
                let left = m31(self, left);
                let hi = self.s.m.bin(c, Bin::Shl, a, left);
                let zero = k(self, 0);
                let none = self.s.m.cmp(c, pred::EQ, sh, zero);
                let hi = self.s.m.select(c, none, zero, hi);
                self.s.m.bin(c, Bin::Or, lo, hi)
            }
            "v_bfrev_b32" => {
                let a = s(self, 0)?;
                self.s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a])
            }
            "v_bcnt_u32_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[a]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            "v_mad_u32_u24" | "v_mad_i32_i24" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let (sh, op) = (
                    k(self, 8),
                    if name == "v_mad_i32_i24" {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                let m = self.s.m.bin(c, Bin::Mul, a, b);
                self.s.m.bin(c, Bin::Add, m, d)
            }
            "v_min3_i32" | "v_max3_i32" | "v_min3_u32" | "v_max3_u32" | "v_med3_i32"
            | "v_med3_u32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let signed = name.ends_with("i32");
                let (lt, gt) = if signed {
                    (pred::SLT, pred::SGT)
                } else {
                    (pred::IULT, pred::IUGT)
                };
                let pick = |l: &mut Self, p, x, y| {
                    let t = l.s.m.cmp(c, p, x, y);
                    l.s.m.select(c, t, x, y)
                };
                match &name[2..6] {
                    "min3" => {
                        let t = pick(self, lt, a, b);
                        pick(self, lt, t, d)
                    }
                    "max3" => {
                        let t = pick(self, gt, a, b);
                        pick(self, gt, t, d)
                    }
                    _ => {
                        let lo = pick(self, lt, a, b);
                        let hi = pick(self, gt, a, b);
                        let t = pick(self, lt, hi, d);
                        pick(self, gt, lo, t)
                    }
                }
            }
            "v_perm_b32" => {
                // Byte i of the result is byte S of {src0, src1} (src1 the
                // low dword) for selector byte S below 8; 8-11 repeat the
                // sign of bytes 1 and 3 of src1, then of src0; 12 is 0x00,
                // higher 0xFF.
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                let sel = match v.src.get(2) {
                    Some(&Operand::Literal(x)) => x,
                    Some(&Operand::Const(k)) => crate::resource::inline(k).ok_or(name)?,
                    _ => {
                        // A computed selector: each byte picked at run time.
                        let (a, b, sv) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                        let i64t = self.s.m.int(64);
                        self.s.flags |= flags::INT64;
                        let wa = self.s.m.cast(c, Cast::ZExt, a, i64t);
                        let wb = self.s.m.cast(c, Cast::ZExt, b, i64t);
                        let k32 = self.s.m.int_const(i64t, 32);
                        let wa = self.s.m.bin(c, Bin::Shl, wa, k32);
                        let both = self.s.m.bin(c, Bin::Or, wa, wb);
                        let mut r = k(self, 0);
                        for i in 0..4u32 {
                            let (sh, ff) = (k(self, 8 * i), k(self, 0xFF));
                            let sel = self.s.m.bin(c, Bin::LShr, sv, sh);
                            let sel = self.s.m.bin(c, Bin::And, sel, ff);
                            // Bytes 0-7 of {src0, src1}.
                            let k3 = k(self, 3);
                            let bits = self.s.m.bin(c, Bin::Shl, sel, k3);
                            let bits = self.s.m.cast(c, Cast::ZExt, bits, i64t);
                            let x = self.s.m.bin(c, Bin::LShr, both, bits);
                            let x = self.s.m.cast(c, Cast::Trunc, x, self.s.t.i32);
                            let mut byte = self.s.m.bin(c, Bin::And, x, ff);
                            // 8-11: the sign of bytes 1, 3 of src1, then of src0.
                            for (n, src, bit) in [(8, b, 15), (9, b, 31), (10, a, 15), (11, a, 31)]
                            {
                                let left = k(self, 31 - bit);
                                let t = self.s.m.bin(c, Bin::Shl, src, left);
                                let k31 = k(self, 31);
                                let t = self.s.m.bin(c, Bin::AShr, t, k31);
                                let t = self.s.m.bin(c, Bin::And, t, ff);
                                let kn = k(self, n);
                                let is = self.s.m.cmp(c, pred::EQ, sel, kn);
                                byte = self.s.m.select(c, is, t, byte);
                            }
                            let (k12, z) = (k(self, 12), k(self, 0));
                            let is12 = self.s.m.cmp(c, pred::EQ, sel, k12);
                            byte = self.s.m.select(c, is12, z, byte);
                            let above = self.s.m.cmp(c, pred::IUGT, sel, k12);
                            byte = self.s.m.select(c, above, ff, byte);
                            let at = k(self, 8 * i);
                            let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                            r = self.s.m.bin(c, Bin::Or, r, byte);
                        }
                        return Ok(r);
                    }
                };
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let mut r = k(self, 0);
                for i in 0..4 {
                    let pick = sel >> (8 * i) & 0xFF;
                    let byte = match pick {
                        0..=7 => {
                            let (src, n) = if pick < 4 { (b, pick) } else { (a, pick - 4) };
                            let sh = k(self, 8 * n);
                            let t = self.s.m.bin(c, Bin::LShr, src, sh);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        8..=11 => {
                            let src = if pick < 10 { b } else { a };
                            let bit = if pick % 2 == 0 { 15 } else { 31 };
                            let left = k(self, 31 - bit);
                            let t = self.s.m.bin(c, Bin::Shl, src, left);
                            let k31 = k(self, 31);
                            let t = self.s.m.bin(c, Bin::AShr, t, k31);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        12 => continue,
                        _ => k(self, 0xFF),
                    };
                    let at = k(self, 8 * i);
                    let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                    r = self.s.m.bin(c, Bin::Or, r, byte);
                }
                r
            }
            "v_bfm_b32" => {
                // ((1 << src0[4:0]) - 1) << src1[4:0]
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let one = k(self, 1);
                let (a, b) = (m31(self, a), m31(self, b));
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                self.s.m.bin(c, Bin::Shl, m, b)
            }
            "v_cvt_pk_u8_f32" => {
                // src0 rounded to nearest even and saturated to a byte, into
                // byte src1[1:0] of src2.
                let f = self.fsrc(inst, 0)?;
                let (b, d) = (s(self, 1)?, s(self, 2)?);
                let f = self.unary(26, f);
                let (zero, max) = (self.s.m.f32(0), self.s.m.f32(255.0f32.to_bits()));
                let f = self.binary(35, f, zero);
                let f = self.binary(36, f, max);
                let x = self.s.m.cast(c, Cast::FpToUi, f, self.s.t.i32);
                let (three, k3) = (k(self, 3), k(self, 3));
                let sh = self.s.m.bin(c, Bin::And, b, three);
                let sh = self.s.m.bin(c, Bin::Shl, sh, k3);
                let x = self.s.m.bin(c, Bin::Shl, x, sh);
                let ff = k(self, 0xFF);
                let hole = self.s.m.bin(c, Bin::Shl, ff, sh);
                let ones = k(self, u32::MAX);
                let hole = self.s.m.bin(c, Bin::Xor, hole, ones);
                let d = self.s.m.bin(c, Bin::And, d, hole);
                self.s.m.bin(c, Bin::Or, d, x)
            }
            "v_ffbl_b32" => {
                // -1 for zero, as DXIL's first-bit-low.
                let a = s(self, 0)?;
                self.s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 32, &[a])
            }
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                // Bits of the mask's low (high) half below this lane, plus src1:
                // with an all-ones mask, the lane index.
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                // Mesh lanes are the console's by construction; pixel and
                // vertex lanes are the host's, as their masks are.
                if self.mask_data == MaskData::Host {
                    // Host lanes: the mask must be one (a ballot) or all lanes.
                    let Fields::Valu(v) = &inst.fields else {
                        unreachable!()
                    };
                    let host = match v.src.first() {
                        Some(&Operand::Const(193)) => true,
                        Some(&Operand::Scalar { code, .. }) => {
                            inst.read(Reg::S(code)).is_some_and(|m| {
                                self.x.st.masks.contains_key(&m) || self.x.st.highs.contains_key(&m)
                            })
                        }
                        _ => false,
                    };
                    if !host {
                        return Err("v_mbcnt of a mask that is no ballot (console lane layout)");
                    }
                    self.s.flags |= flags::WAVE_OPS;
                } else if self.s.mesh.is_none() {
                    self.cross_lane(false)?;
                }
                // On a host wave wider than the console's the upper half is a
                // second console wave: `lo` counts bits of the lane's own
                // half's mask below its lane in it, and a wave32 mask's high
                // half is empty.
                let host_halves = self.mask_data == MaskData::Host && !self.x.native.wave64;
                let lane = self.lane_index();
                let below = if name == "v_mbcnt_lo_u32_b32" {
                    if host_halves {
                        let k31 = k(self, 31);
                        self.s.m.bin(c, Bin::And, lane, k31)
                    } else {
                        lane
                    }
                } else {
                    let (k32, z) = (k(self, 32), k(self, 0));
                    let t = self.s.m.bin(c, Bin::Sub, lane, k32);
                    let low = self.s.m.cmp(c, pred::IULT, lane, k32);
                    self.s.m.select(c, low, z, t)
                };
                // (1 << below) - 1, all ones from 32 on.
                let (one, k32, ones) = (k(self, 1), k(self, 32), k(self, u32::MAX));
                let sh = m31(self, below);
                let m = self.s.m.bin(c, Bin::Shl, one, sh);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                let full = self.s.m.cmp(c, pred::IUGE, below, k32);
                let m = self.s.m.select(c, full, ones, m);
                let x = if host_halves && name == "v_mbcnt_hi_u32_b32" {
                    k(self, 0)
                } else {
                    self.s.m.bin(c, Bin::And, a, m)
                };
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[x]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            _ => return Err(name),
        })
    }

    fn compare(&mut self, inst: &Inst, rest: &str) -> Result<(), Why> {
        let (op, ty) = rest.rsplit_once('_').ok_or("a compare")?;
        let c = self.cur;
        let r = match ty {
            "f32" if op == "class" => {
                // Bit k of src1 tests class k: signalling NaN, quiet NaN,
                // -inf, -normal, -denormal, -0, +0, +denormal, +normal, +inf.
                let x = self.fsrc(inst, 0)?;
                let x = self.as_int(x);
                let mask = self.src(inst, 1)?;
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (k23, k31, kff, mant, quiet) = (
                    k(self, 23),
                    k(self, 31),
                    k(self, 0xFF),
                    k(self, 0x7F_FFFF),
                    k(self, 0x40_0000),
                );
                let e = self.s.m.bin(c, Bin::LShr, x, k23);
                let e = self.s.m.bin(c, Bin::And, e, kff);
                let m = self.s.m.bin(c, Bin::And, x, mant);
                let neg = self.s.m.bin(c, Bin::LShr, x, k31);
                let zero = k(self, 0);
                let (e0, e255) = (
                    self.s.m.cmp(c, pred::EQ, e, zero),
                    self.s.m.cmp(c, pred::EQ, e, kff),
                );
                let m0 = self.s.m.cmp(c, pred::EQ, m, zero);
                let q = self.s.m.bin(c, Bin::And, x, quiet);
                let q = self.s.m.cmp(c, pred::NE, q, zero);
                // Positive classes 6-9; negative ones are 11 minus those.
                let (normal, inf, zero_c, denorm) =
                    (k(self, 8), k(self, 9), k(self, 6), k(self, 7));
                let t = self.s.m.select(c, m0, zero_c, denorm);
                let t = self.s.m.select(c, e0, t, normal);
                let t = self.s.m.select(c, e255, inf, t);
                let eleven = k(self, 11);
                let mirrored = self.s.m.bin(c, Bin::Sub, eleven, t);
                let is_neg = self.s.m.cmp(c, pred::NE, neg, zero);
                let t = self.s.m.select(c, is_neg, mirrored, t);
                let (one, snan) = (k(self, 1), k(self, 0));
                let nan_class = self.s.m.select(c, q, one, snan);
                let yes = self.s.m.bool(true);
                let nan = self.s.m.bin(c, Bin::Xor, m0, yes);
                let nan = self.s.m.bin(c, Bin::And, nan, e255);
                let t = self.s.m.select(c, nan, nan_class, t);
                let bit = self.s.m.bin(c, Bin::LShr, mask, t);
                let bit = self.s.m.bin(c, Bin::And, bit, one);
                self.s.m.cmp(c, pred::NE, bit, zero)
            }
            "f32" | "f16" => {
                let (a, b) = if ty == "f16" {
                    (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?)
                } else {
                    (self.fsrc(inst, 0)?, self.fsrc(inst, 1)?)
                };
                // `fsrc` may emit two separate bitcasts for the same register,
                // which host CSE later merges into `fcmp ... %f, %f`.
                let same_src = a == b
                    || (self.src0.is_none()
                        && matches!(
                            &inst.fields,
                            Fields::Valu(v)
                                if v.src.len() >= 2
                                    && v.src[0] == v.src[1]
                                    && (v.abs & 1) == ((v.abs >> 1) & 1)
                                    && (v.neg & 1) == ((v.neg >> 1) & 1)
                                    && (v.opsel & 1) == ((v.opsel >> 1) & 1)
                                    && v.sdwa.is_none()
                        ));
                // Some D3D12 drivers (observed on Intel Iris Xe) assume no NaNs
                // for plain `fcmp`, folding `fcmp ord` to true, `fcmp uno` to
                // false (even across distinct registers), and self-comparisons
                // like `fcmp oeq/une x, x` to constants. Use DXIL `IsNaN`
                // (`dx.op.isSpecialFloat.f32`, opcode 8) for ordered/unordered
                // and same-value NaN-sensitive predicates.
                let is_nan = |l: &mut Self, x: V| {
                    l.s.op(c, "isSpecialFloat.f32", Attr::ReadNone, l.s.t.i1, 8, &[x])
                };
                let any_nan = |l: &mut Self, x: V, y: V| {
                    let nx = is_nan(l, x);
                    if same_src {
                        nx
                    } else {
                        let ny = is_nan(l, y);
                        l.s.m.bin(c, Bin::Or, nx, ny)
                    }
                };
                match op {
                    "u" => any_nan(self, a, b),
                    "o" => {
                        let u = any_nan(self, a, b);
                        let yes = self.s.m.bool(true);
                        self.s.m.bin(c, Bin::Xor, u, yes)
                    }
                    "eq" | "le" | "ge" if same_src => {
                        let n = is_nan(self, a);
                        let yes = self.s.m.bool(true);
                        self.s.m.bin(c, Bin::Xor, n, yes)
                    }
                    "neq" | "nge" | "nle" if same_src => is_nan(self, a),
                    _ => {
                        let p = match op {
                            "f" => pred::FALSE,
                            "lt" => pred::OLT,
                            "eq" => pred::OEQ,
                            "le" => pred::OLE,
                            "gt" => pred::OGT,
                            "lg" => pred::ONE,
                            "ge" => pred::OGE,
                            "nge" => pred::ULT,
                            "nlg" => pred::UEQ,
                            "ngt" => pred::ULE,
                            "nle" => pred::UGT,
                            "neq" => pred::UNE,
                            "nlt" => pred::UGE,
                            "tru" => pred::TRUE,
                            _ => return Err(inst.name()),
                        };
                        self.s.m.cmp(c, p, a, b)
                    }
                }
            }
            "u64" | "i64" => {
                let (a, b) = (self.src64(inst, 0)?, self.src64(inst, 1)?);
                let signed = ty == "i64";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    _ => return Err("a 64-bit compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            "u32" | "i32" | "u16" | "i16" => {
                let (a, b) = if ty.ends_with("16") {
                    let (mut a, mut b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
                    if ty == "i16" {
                        let k16 = self.s.m.i32(16);
                        for x in [&mut a, &mut b] {
                            let t = self.s.m.bin(c, Bin::Shl, *x, k16);
                            *x = self.s.m.bin(c, Bin::AShr, t, k16);
                        }
                    }
                    (a, b)
                } else {
                    (self.src(inst, 0)?, self.src(inst, 1)?)
                };
                let signed = ty == "i32" || ty == "i16";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    "f" | "t" => {
                        self.lane(inst, Some(op == "t"));
                        return Ok(());
                    }
                    _ => return Err("an integer compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            _ => return Err(inst.name()),
        };
        self.lane_bits_for(inst, r);
        Ok(())
    }

    fn lane(&mut self, inst: &Inst, value: Option<bool>) {
        let b = self.s.m.bool(value == Some(true));
        self.lane_bits_for(inst, b);
    }

    fn lane_bits_for(&mut self, inst: &Inst, r: V) {
        let Fields::Valu(v) = &inst.fields else {
            return;
        };
        let code = if inst.name().starts_with("v_cmpx") {
            Some(EXEC)
        } else {
            match v.dst {
                Some(Operand::Scalar { code, .. }) => Some(code),
                _ => None,
            }
        };
        if let Some(lo) = code.and_then(|c| inst.write(Reg::S(c))) {
            self.lane_bits.insert(lo, r);
        }
        for &(_, w) in &inst.writes {
            self.vals
                .entry(w)
                .or_insert(Val::Missing("a lane mask read as data"));
        }
    }

    fn interp(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Some(&interp) = self.interps.get(&site) else {
            return Err("the first half of an interpolation");
        };
        let name = inst.name();
        let (dst, src) = match &inst.fields {
            Fields::Vintrp { dst, src, .. } => (*dst, Some(*src)),
            Fields::Valu(v) => {
                if v.abs != 0 || v.neg != 0 {
                    return Err("interpolation operand modifiers");
                }
                (v.dst.ok_or("an interpolation")?, v.src.get(1).copied())
            }
            _ => unreachable!(),
        };
        let Operand::Vector { reg, .. } = dst else {
            return Err("an interpolation");
        };
        // A float operand of the instruction.
        let float = |l: &mut Self, op: Option<Operand>| -> Result<V, Why> {
            let x = l.operand(inst, &op.ok_or("an interpolation operand")?)?;
            Ok(l.as_float(x))
        };
        // What the result adds to the evaluated channel (a second half's
        // first half, when the halves evaluate on their own).
        let mut add = None;
        let (attribute, eval) = match *interp {
            Interp::Interpolated {
                attribute,
                barycentric,
                ..
            } => {
                let mode = interp_mode(barycentric).ok_or("pull-model interpolation")?;
                let eval = if self.per_vertex(attribute.row) {
                    // The element is read per vertex elsewhere, so it is
                    // constant: weigh its vertices with SV_Barycentrics.
                    let mut ij = [None; 2];
                    for (index, w) in ij.iter_mut().enumerate() {
                        let x = match self.barycentric(barycentric, index as u8) {
                            Val::I32(x) => x,
                            Val::Missing(w) => return Err(w),
                            Val::Bit(_) => unreachable!(),
                        };
                        *w = Some(self.as_float(x));
                    }
                    Eval::Weights {
                        p0: true,
                        i: ij[0],
                        j: ij[1],
                    }
                } else {
                    Eval::Mode(mode)
                };
                (attribute, eval)
            }
            // P0, the provoking vertex: a flat (nointerpolation) input.
            Interp::Vertex {
                attribute,
                vertex: 2,
            } => (attribute, Eval::Mode(1)),
            // P10 and P20: what the first and second halves weigh, P1 - P0
            // and P2 - P0, which is what the hardware holds for an
            // attribute the program also interpolates.
            Interp::Vertex { attribute, vertex } if self.interpolated(attribute.row) => {
                let one = Some(self.s.m.f32(1f32.to_bits()));
                let (i, j) = if vertex == 0 {
                    (one, None)
                } else {
                    (None, one)
                };
                (attribute, Eval::Weights { p0: false, i, j })
            }
            Interp::Vertex { .. } => return Err("a non-provoking vertex value"),
            // A half of an interpolation with computed i or j: P0 + i P10,
            // or the old dst (f16: the third operand) + j P20.
            Interp::Weighted { attribute } if name.starts_with("v_interp_p1") => {
                let i = Some(float(self, src)?);
                (
                    attribute,
                    Eval::Weights {
                        p0: true,
                        i,
                        j: None,
                    },
                )
            }
            Interp::Weighted { attribute } => {
                let old = match &inst.fields {
                    Fields::Valu(v) => v.src.get(2).copied(),
                    _ => Some(dst),
                };
                add = Some(float(self, old)?);
                let j = Some(float(self, src)?);
                (
                    attribute,
                    Eval::Weights {
                        p0: false,
                        i: None,
                        j,
                    },
                )
            }
        };
        let packing = self.half_inputs.get(&attribute.row).copied();
        let f16 = name.ends_with("_f16");
        // The evaluated channel plus `add`, as a float.
        let total = |l: &mut Self, row: u8, ch: u8| -> Result<V, Why> {
            let x = l.eval_channel(row, ch, eval)?;
            let x = l.as_float(x);
            Ok(match add {
                Some(a) => l.s.m.bin(l.cur, Bin::Add, a, x),
                None => x,
            })
        };
        let x = match packing {
            None if f16 || attribute.high => return Err("a 16-bit read of a 32-bit input"),
            None => {
                let x = total(self, attribute.row, attribute.channel)?;
                self.as_int(x)
            }
            Some((pack2, b)) => {
                // 16-bit values: one in the low half of each dword, or two
                // per dword (`packed_half`); the host's are 32-bit floats.
                let half = |l: &mut Self, h: u8| -> Result<Option<V>, Why> {
                    let src = if pack2 {
                        packed_half(&l.s.inputs, attribute.row, b, h)
                    } else {
                        h.is_multiple_of(2).then_some((attribute.row, h / 2))
                    };
                    match src {
                        Some((row, ch)) => total(l, row, ch).map(Some),
                        None => Ok(None),
                    }
                };
                let to_f16 = |l: &mut Self, f: V| {
                    let i32t = l.s.t.i32;
                    l.s.op(l.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
                };
                let h = 2 * attribute.channel + u8::from(attribute.high);
                let c = self.cur;
                if name == "v_interp_p1ll_f16" {
                    // An f32 result from 16-bit vertex values.
                    let f = half(self, h)?.ok_or("a packed input half no element holds")?;
                    self.as_int(f)
                } else if f16 {
                    let f = half(self, h)?.ok_or("a packed input half no element holds")?;
                    let lo = to_f16(self, f);
                    self.low_half(inst, reg, lo)
                } else if matches!(eval, Eval::Mode(1)) {
                    // The whole dword of a flat input: both halves.
                    let h = 2 * attribute.channel;
                    let z = self.s.m.i32(0);
                    let lo = match half(self, h)? {
                        Some(f) => to_f16(self, f),
                        None => z,
                    };
                    let hi = match half(self, h + 1)? {
                        Some(f) => to_f16(self, f),
                        None => z,
                    };
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                } else {
                    return Err("32-bit interpolation of a 16-bit input");
                }
            }
        };
        // A pixel interpolates in every lane; guards only hide inactive ones.
        self.write(inst, Reg::V(reg), x, None);
        Ok(())
    }

    /// Whether the program reads attribute row `row` per vertex (an
    /// interpolation with computed barycentrics, or P10 or P20): then its
    /// element is declared constant and read with `attributeAtVertex`.
    fn per_vertex(&self, row: u8) -> bool {
        self.x.io.interps.iter().any(|(_, i)| match *i {
            Interp::Weighted { attribute } => attribute.row == row,
            Interp::Vertex { attribute, vertex } => {
                attribute.row == row && vertex != 2 && self.interpolated(row)
            }
            Interp::Interpolated { .. } => false,
        })
    }

    /// Whether the program interpolates attribute row `row` (so the
    /// hardware holds P1 - P0 and P2 - P0 for it).
    fn interpolated(&self, row: u8) -> bool {
        self.x.io.interps.iter().any(|(_, i)| match *i {
            Interp::Interpolated { attribute, .. } | Interp::Weighted { attribute } => {
                attribute.row == row
            }
            Interp::Vertex { .. } => false,
        })
    }

    /// Channel `ch` of input row `row` evaluated as `eval` says, as raw bits.
    fn eval_channel(&mut self, row: u8, ch: u8, eval: Eval) -> Result<V, Why> {
        let (id, col) = self
            .find_input(row, ch)
            .ok_or("an input the signature lacks")?;
        let mode = match eval {
            Eval::Mode(mode) => mode,
            Eval::Weights { .. } => 1,
        };
        let e = &mut self.s.inputs[id];
        if e.interp != mode && e.usage != 0 {
            return Err("one input interpolated two ways");
        }
        e.interp = mode;
        let Eval::Weights { p0, i, j } = eval else {
            return match self.load_any_input(id, col) {
                Val::I32(x) => Ok(x),
                _ => Err("an input"),
            };
        };
        let v0 = self.vertex_input(id, col, 0);
        let c = self.cur;
        let mut acc = p0.then_some(v0);
        for (k, w) in [(1, i), (2, j)] {
            let Some(w) = w else { continue };
            let vk = self.vertex_input(id, col, k);
            let d = self.s.m.bin(c, Bin::Sub, vk, v0);
            acc = Some(match acc {
                Some(a) => self.fmad(w, d, a),
                None => self.s.m.bin(c, Bin::Mul, w, d),
            });
        }
        let acc = acc.ok_or("an interpolation")?;
        Ok(self.as_int(acc))
    }

    /// Vertex `k` (0 the provoking vertex, as `barycentric` orders them)
    /// of a constant input element's channel, as a float.
    fn vertex_input(&mut self, id: usize, col: u8, k: u8) -> V {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let int = e.comp != 9;
        let (ty, suffix) = if int {
            (self.s.t.i32, ".i32")
        } else {
            (self.s.t.f32, ".f32")
        };
        self.s.flags |= flags::BARYCENTRICS;
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.i8(k),
        ];
        let v = self.s.op(
            entry,
            &format!("attributeAtVertex{suffix}"),
            Attr::ReadNone,
            ty,
            137,
            &args,
        );
        if int {
            self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.f32)
        } else {
            v
        }
    }

    /// `x` (an f16 in the low bits) as the low half of VGPR `reg`: 16-bit
    /// results keep the register's high half (GFX10).
    fn low_half(&mut self, inst: &Inst, reg: u8, x: V) -> V {
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let (high, low) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF));
        let old = self.s.m.bin(c, Bin::And, old, high);
        let x = self.s.m.bin(c, Bin::And, x, low);
        self.s.m.bin(c, Bin::Or, old, x)
    }

    fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Image {
            data,
            addr,
            dmask,
            dim,
            unorm,
            a16,
            d16,
            tfe,
            lwe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        if *tfe || *lwe {
            return Err("an image access with tfe or lwe");
        }
        let name = inst.name();
        // A load or sample with no channel enabled returns the first.
        let dmask = if *dmask == 0 { &1 } else { dmask };
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let (kind, coords) = match dim {
            0 => (1, 1),
            1 => (2, 2),
            2 => (4, 3),
            // Sampled (face coordinates and face, see `cube_direction`) or
            // sized only.
            3 if name.starts_with("image_sample")
                || name.starts_with("image_gather")
                || name == "image_get_resinfo" =>
            {
                (5, 3)
            }
            4 => (6, 2),
            5 => (7, 3),
            _ => return Err("an image dimension (cube loads or MSAA)"),
        };
        // Address VGPRs in order.
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        // Full 32-bit address VGPR (offsets, bias, comparison reference stay
        // 32-bit even when `a16` packs the remaining coordinates/gradients/LOD).
        let addr_dword = |l: &mut Self, idx: usize| -> Result<V, Why> {
            let r = *regs.get(idx).ok_or("an image address")?;
            let v = inst.read(Reg::V(r)).ok_or("an image address")?;
            l.i32_of(v)
        };
        // Component `k` starting at VGPR `start`: with `a16`, two 16-bit
        // components per VGPR, low half first.
        let addr_val = |l: &mut Self, start: usize, k: usize| -> Result<V, Why> {
            let x = addr_dword(l, start + if *a16 { k / 2 } else { k })?;
            if !*a16 {
                return Ok(x);
            }
            let c = l.cur;
            let (sh, mask) = (l.s.m.i32(16 * (k as u32 % 2)), l.s.m.i32(0xFFFF));
            let x = l.s.m.bin(c, Bin::LShr, x, sh);
            Ok(l.s.m.bin(c, Bin::And, x, mask))
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked image")?;
        let store = name.starts_with("image_store");
        let uav = store
            || self
                .used_kind(src)
                .is_some_and(|u| u.kind == ResourceType::Uav);
        let props = (kind | if uav { 0x1000 } else { 0 }, 0x409);
        if let Some(op) = name.strip_prefix("image_atomic_") {
            // GLC returns the old value. DXIL implements subtraction by adding
            // the two's complement of the operand at the same width.
            let code = match op {
                "add" | "sub" => 0,
                "and" => 1,
                "or" => 2,
                "xor" => 3,
                "smin" => 4,
                "smax" => 5,
                "umin" => 6,
                "umax" => 7,
                "swap" => 8,
                "cmpswap" => u32::MAX,
                _ => return Err(name),
            };
            let Fields::Image { cache, .. } = &inst.fields else {
                unreachable!()
            };
            if *d16 || *a16 {
                return Err("a 16-bit image atomic");
            }
            // Compare-exchange consumes a replacement followed by a comparand.
            let compare = code == u32::MAX;
            let wide = match (compare, *dmask) {
                (false, 1) | (true, 3) => false,
                (false, 3) | (true, 15) => true,
                _ => return Err("an image atomic data mask"),
            };
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, 0, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image atomic data");
            };
            let width = if wide { 2 } else { 1 };
            let vals = self.vgprs(inst, data, if compare { 2 * width } else { width })?;
            let (ty, component) = if wide {
                self.s.flags |= flags::INT64 | flags::ATOMIC_TYPED64 | flags::ATOMIC_HEAP64;
                (self.s.m.int(64), 0x107) // one uint64_t in an R32G32_UINT view
            } else {
                (self.s.t.i32, 0x105)
            };
            let original = if wide {
                self.join64(vals[0], vals[1])
            } else {
                vals[0]
            };
            let value = if op == "sub" {
                let zero = self.s.m.int_const(ty, 0);
                self.s.m.bin(c, Bin::Sub, zero, original)
            } else {
                original
            };
            let h = self.handle(site, src, false, (kind | 0x1000, component))?;
            let mut args = vec![h];
            let (intrinsic, opcode) = if compare {
                let comparand = if wide {
                    self.join64(vals[2], vals[3])
                } else {
                    vals[1]
                };
                args.extend(cs);
                args.extend([comparand, value]);
                let intrinsic = if wide {
                    "atomicCompareExchange.i64"
                } else {
                    "atomicCompareExchange.i32"
                };
                (intrinsic, 79)
            } else {
                args.push(self.s.m.i32(code));
                args.extend(cs);
                args.push(value);
                (
                    if wide {
                        "atomicBinOp.i64"
                    } else {
                        "atomicBinOp.i32"
                    },
                    78,
                )
            };
            let at = self.begin_guarded(guard);
            let r = self
                .s
                .op(self.cur, intrinsic, Attr::NoUnwind, ty, opcode, &args);
            if cache.glc {
                let old = self.end_guarded(at, &[(r, original)])[0];
                if wide {
                    let (lo, hi) = self.split64(old);
                    self.write(inst, Reg::V(reg), lo, None);
                    self.write(inst, Reg::V(reg + 1), hi, None);
                } else {
                    self.write(inst, Reg::V(reg), old, None);
                }
            } else {
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if store {
            if name != "image_store" {
                return Err("an image store to a mip level");
            }
            // shortcut: stores are taken to write all four channels of a
            // float format (the console converts by the view's format). The
            // data VGPRs hold the dmask's channels in order; the others are
            // written as 0.
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, 0, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image store data");
            };
            let mut vals = Vec::new();
            for ch in 0..4u8 {
                if dmask >> ch & 1 == 0 {
                    vals.push(self.s.m.f32(0));
                    continue;
                }
                let k = (dmask & ((1 << ch) - 1)).count_ones() as u8;
                // d16: two f16 channels per VGPR, low half first.
                let r = if *d16 { k / 2 } else { k };
                let v = inst.read(Reg::V(reg + r)).ok_or("image store data")?;
                vals.push(if *d16 {
                    let mut x = self.i32_of(v)?;
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(self.cur, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    self.s
                        .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
                } else {
                    self.float_of(v)?
                });
            }
            let h = self.handle(site, src, false, props)?;
            let at = self.begin_guarded(guard);
            let mut args = vec![h];
            args.extend(cs);
            args.extend(vals);
            args.push(self.s.m.i8(0xF));
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                "textureStore.f32",
                Attr::NoUnwind,
                void,
                67,
                &args,
            );
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // Samples and gathers: the DXIL operation follows from the form.
        let form = sample_form(name);
        let gather = form.is_some_and(|f| f.gather);
        let (op, sample) = match name {
            "image_get_lod" if self.s.pixel => (81, true),
            "image_load" | "image_load_mip" => (66, false),
            "image_get_resinfo" => (72, false),
            _ if form.is_some() => (0, true),
            _ => return Err(name),
        };
        if op == 72 {
            // Width, height, depth or layers, levels (per dmask) of the level
            // in the first address: DXIL's dimensions; a channel DXIL leaves
            // out reads 1 (UAVs: one level).
            let h = self.handle(site, src, false, props)?;
            let mip = if uav {
                self.s.m.undef(self.s.t.i32)
            } else {
                addr_val(self, 0, 0)?
            };
            let i32t = self.s.t.i32;
            let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
            let d = self
                .s
                .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
            let one = self.s.m.i32(1);
            let present = match kind {
                1 => [true, false, false, true],
                2 | 5 | 6 => [true, true, false, true],
                _ => [true, true, true, true],
            };
            let Operand::Vector { reg, .. } = *data else {
                return Err("an image result");
            };
            for (k, ch) in (0..4u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                let x = if ch == 3 && uav || !present[ch as usize] {
                    one
                } else {
                    self.s.m.extract(c, d, ch)
                };
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if uav && op != 66 {
            return Err("sampling a UAV");
        }
        let h = self.handle(site, src, false, props)?;
        let fu = self.s.m.undef(self.s.t.f32);
        let iu = self.s.m.undef(self.s.t.i32);
        let r = if sample {
            let ss = self
                .descriptors
                .get(&(site, true))
                .copied()
                .ok_or("an untracked sampler")?;
            let f = form.unwrap_or_default();
            if f.c && dmask.count_ones() != 1 {
                return Err("a comparison sample of several channels");
            }
            // Comparison: the sampler is a comparison sampler.
            let sh = self.handle(site, ss, true, (14 | if f.c { 0x8000 } else { 0 }, 0))?;
            // Coordinates the size divides (not the array slice).
            let spatial = coords - usize::from(kind == 6 || kind == 7);
            // Address dwords in order (LLVM ImageDimIntrinsics.td, Mesa
            // ac_build_image_opcode): the texel offset, bias and comparison
            // reference, a dword each (an a16 bias is an f16 in the low
            // half, LLVM SIISelLowering::lowerImage); the derivatives, x
            // then y; the coordinates, then the level or level clamp.
            // Derivatives are 16-bit in the g16 opcodes, coordinates and
            // levels with a16 (GFX10.3 keeps the two apart, as LLVM's
            // lowerImage does where hasG16): two per dword, low half first,
            // each direction starting a dword.
            let whole = usize::from(f.o) + usize::from(f.b) + usize::from(f.c);
            let bias_at = (usize::from(f.o), a16.then_some(0));
            let grads = if f.d { spatial } else { 0 };
            let g16 = f.g16;
            let per_direction = if g16 { grads.div_ceil(2) } else { grads };
            let grad_at = |dir: usize, k: usize| {
                if g16 {
                    (whole + dir * per_direction + k / 2, Some(k as u32 % 2))
                } else {
                    (whole + dir * grads + k, None)
                }
            };
            let first = whole + 2 * per_direction;
            // Coordinate k; k = coords is the level or the level clamp.
            let coord_at = |k: usize| {
                if *a16 {
                    (first + k / 2, Some(k as u32 % 2))
                } else {
                    (first + k, None)
                }
            };
            let raw = |l: &mut Self, at: usize| -> Result<V, Why> {
                let r = *regs.get(at).ok_or("an image address")?;
                let v = inst.read(Reg::V(r)).ok_or("an image address")?;
                l.i32_of(v)
            };
            // A float address: a dword, or an f16 half of one.
            let float = |l: &mut Self, (at, half): (usize, Option<u32>)| -> Result<V, Why> {
                let x = raw(l, at)?;
                let Some(h) = half else {
                    return Ok(l.as_float(x));
                };
                let c = l.cur;
                let (sh, mask) = (l.s.m.i32(16 * h), l.s.m.i32(0xFFFF));
                let x = l.s.m.bin(c, Bin::LShr, x, sh);
                let x = l.s.m.bin(c, Bin::And, x, mask);
                let f32t = l.s.t.f32;
                Ok(l.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]))
            };
            // A float address known at translation (an f16 only when 0).
            let known =
                |l: &mut Self, (at, half): (usize, Option<u32>)| -> Result<Option<f32>, Why> {
                    let x = raw(l, at)?;
                    Ok(l.s.m.known(x).and_then(|bits| match half {
                        None => Some(f32::from_bits(bits as u32)),
                        Some(h) => (bits >> (16 * h) & 0x7FFF == 0).then_some(0.0),
                    }))
                };
            let level = if f.l {
                known(self, coord_at(coords))?
            } else {
                f.lz.then_some(0.0)
            };
            let level_zero = level == Some(0.0);
            // The DXIL operation. SM 6.6 has no comparison at a level, with
            // a bias or with gradients (SampleCmpLevel is SM 6.7,
            // SampleCmpBias and SampleCmpGrad 6.8) and gathers level 0 only;
            // those forms are taken where a level or bias known to be 0 makes
            // them one SM 6.6 has.
            let op = if op == 81 {
                81
            } else if f.gather {
                if f.b {
                    return Err("a gather with a level bias");
                }
                if !f.l && !f.lz {
                    return Err("a gather with implicit level");
                }
                if !level_zero && (!f.l || f.c) {
                    return Err("a gather at a level not known to be 0");
                }
                if f.l && !level_zero && kind != 2 {
                    return Err("a gather with explicit level on a non-2D texture");
                }
                if dmask.count_ones() != 1 {
                    return Err("a gather of several channels");
                }
                if !matches!(kind, 2 | 5 | 7) {
                    return Err("a gather from a 1D or 3D image");
                }
                if f.c { 74 } else { 73 }
            } else if f.c {
                if f.d {
                    return Err("a comparison sample with explicit derivatives");
                }
                if f.l && !level_zero {
                    return Err("a comparison sample at a level not known to be 0");
                }
                if f.b && known(self, bias_at)?.is_none_or(|b| b != 0.0) {
                    return Err("a comparison sample with a level bias");
                }
                if f.l || f.lz {
                    65
                } else if self.s.pixel || self.s.threads.is_some() && self.s.mesh.is_none() {
                    // Implicit derivatives: pixel and compute shaders.
                    64
                } else {
                    return Err(
                        "a comparison sample with implicit derivatives outside pixel and compute shaders",
                    );
                }
            } else if f.d {
                63
            } else if f.l || f.lz {
                62
            } else if f.b {
                61
            } else {
                60
            };
            let mut cs = vec![fu; 4];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = float(self, coord_at(k))?;
            }
            if kind == 5 {
                if *unorm || f.d || op == 81 {
                    return Err("a cube sample with texel coordinates, gradients or a level query");
                }
                if f.o {
                    return Err("a texel offset on a cube image");
                }
                let dir = self.cube_direction(cs[0], cs[1], cs[2]);
                cs[..3].copy_from_slice(&dir);
            }
            if *unorm {
                // Texel coordinates: divided by the size of level 0.
                let i32t = self.s.t.i32;
                let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                let z = self.s.m.i32(0);
                let d = self
                    .s
                    .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                for (k, slot) in cs.iter_mut().enumerate().take(spatial) {
                    let n = self.s.m.extract(c, d, k as u32);
                    let n = self.s.m.cast(c, Cast::UiToFp, n, self.s.t.f32);
                    *slot = self.s.m.bin(c, Bin::SDiv, *slot, n);
                }
            }
            if op == 81 {
                // The clamped then the unclamped level of detail.
                let Operand::Vector { reg, .. } = *data else {
                    return Err("an image result");
                };
                let f32t = self.s.t.f32;
                for (k, ch) in (0..2u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                    let clamped = self.s.m.bool(ch == 0);
                    let lod = self.s.op(
                        c,
                        "calculateLOD.f32",
                        Attr::ReadOnly,
                        f32t,
                        81,
                        &[
                            h,
                            sh,
                            cs[0],
                            if spatial > 1 { cs[1] } else { fu },
                            if spatial > 2 { cs[2] } else { fu },
                            clamped,
                        ],
                    );
                    let x = self.as_int(lod);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let offsets = if kind == 5 { 0 } else { spatial.min(3) };
            let zero = self.s.m.i32(0);
            let mut os = vec![iu; 3];
            for o in os.iter_mut().take(offsets) {
                *o = zero;
            }
            if f.o {
                // Texel offsets: x in bits 0-5, y in 8-13, z in 16-21, signed.
                let w = raw(self, 0)?;
                let field = |w: i64, k: usize| (w as i32) << (26 - 8 * k) >> 26;
                let immediate = self
                    .s
                    .m
                    .known(w)
                    .filter(|&w| (0..offsets).all(|k| (-8..=7).contains(&field(w, k))));
                if let (Some(w), false) = (immediate, f.gather) {
                    for (k, o) in os.iter_mut().enumerate().take(offsets) {
                        *o = self.s.m.i32(field(w, k) as u32);
                    }
                } else {
                    // Bits at..at+5 sign-extended: shift up, then down.
                    let k26 = self.s.m.i32(26);
                    for (k, o) in os.iter_mut().enumerate().take(offsets) {
                        let left = self.s.m.i32(26 - 8 * k as u32);
                        let t = self.s.m.bin(c, Bin::Shl, w, left);
                        *o = self.s.m.bin(c, Bin::AShr, t, k26);
                    }
                }
                if !f.gather && immediate.is_none() {
                    // A sample's offsets are immediates in -8..7 (gathers
                    // take any, per lane). At a level known here, the
                    // coordinates move by offset / that level's size instead:
                    // the same texels, the coordinate rounded once more (the
                    // console adds the offset to the texel coordinate). A
                    // sampler whose LOD clamp or bias moves the sampled
                    // level away from the one given, or a level past the
                    // view's last, differs.
                    let Some(n) = level.filter(|n| *n >= 0.0 && n.fract() == 0.0) else {
                        return Err(
                            "a texel offset past -8..7 or per lane at a level not known here",
                        );
                    };
                    let i32t = self.s.t.i32;
                    let f32t = self.s.t.f32;
                    let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                    let n = self.s.m.i32(n as u32);
                    let d = self
                        .s
                        .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, n]);
                    for (k, o) in os.iter_mut().enumerate().take(offsets) {
                        let size = self.s.m.extract(c, d, k as u32);
                        let size = self.s.m.cast(c, Cast::UiToFp, size, f32t);
                        let texels = self.s.m.cast(c, Cast::SiToFp, *o, f32t);
                        let shift = self.s.m.bin(c, Bin::SDiv, texels, size);
                        cs[k] = self.s.m.bin(c, Bin::Add, cs[k], shift);
                        *o = zero;
                    }
                }
            }
            // The level clamp, after the coordinates.
            let clamp = if f.cl {
                float(self, coord_at(coords))?
            } else {
                fu
            };
            let mut args = vec![h, sh];
            args.extend(&cs);
            if self.s.threads.is_some() && self.s.mesh.is_none() && matches!(op, 60 | 61 | 64 | 81)
            {
                self.s.cs_derivatives = true;
            }
            if f.cl && matches!(op, 60 | 61 | 63 | 64) {
                self.s.sample_clamp = true;
            }
            match op {
                60 => {
                    args.extend(&os);
                    args.push(clamp);
                    self.s
                        .op(c, "sample.f32", Attr::ReadOnly, self.s.t.res_ret, 60, &args)
                }
                61 => {
                    let bias = float(self, bias_at)?;
                    args.extend(&os);
                    args.extend([bias, clamp]);
                    self.s.op(
                        c,
                        "sampleBias.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        61,
                        &args,
                    )
                }
                63 => {
                    // shortcut: the coarse forms (`_cd`) compute one level
                    // per quad; the host's per pixel is the same when the
                    // gradients agree across the quad.
                    let mut g = vec![fu; 6];
                    for k in 0..spatial {
                        g[k] = float(self, grad_at(0, k))?;
                        g[3 + k] = float(self, grad_at(1, k))?;
                    }
                    args.extend(&os);
                    args.extend(g);
                    args.push(clamp);
                    self.s.op(
                        c,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                64 | 65 => {
                    let at = usize::from(f.o) + usize::from(f.b);
                    let reference = float(self, (at, None))?;
                    args.extend(&os);
                    args.push(reference);
                    let name = if op == 64 {
                        args.push(clamp);
                        "sampleCmp.f32"
                    } else {
                        "sampleCmpLevelZero.f32"
                    };
                    self.s
                        .op(c, name, Attr::ReadOnly, self.s.t.res_ret, op, &args)
                }
                62 => {
                    let lod = if f.l {
                        float(self, coord_at(coords))?
                    } else {
                        self.s.m.f32(0)
                    };
                    args.extend(&os);
                    args.push(lod);
                    self.s.op(
                        c,
                        "sampleLevel.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        62,
                        &args,
                    )
                }
                _ => {
                    if f.l && !level_zero {
                        // Console-only Gather at an explicit mip level: DXIL's
                        // textureGather has no LOD argument, so emulate with
                        // four clamped textureLoad calls at that mip in HLSL
                        // gather order: (-,+), (+,+), (+,-), (-,-).
                        let lod = float(self, coord_at(coords))?;
                        let i32t = self.s.t.i32;
                        let f32t = self.s.t.f32;
                        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                        let d0 = self
                            .s
                            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, zero]);
                        let levels = self.s.m.extract(c, d0, 3);
                        let one = self.s.m.i32(1);
                        let max_mip = self.s.m.bin(c, Bin::Sub, levels, one);
                        let fzero = self.s.m.f32(0);
                        let lod = self.binary(35, lod, fzero);
                        let mip = self.s.m.cast(c, Cast::FpToUi, lod, i32t);
                        let over = self.s.m.cmp(c, pred::IUGT, mip, max_mip);
                        let mip = self.s.m.select(c, over, max_mip, mip);
                        let dm = self
                            .s
                            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
                        let w = self.s.m.extract(c, dm, 0);
                        let ht = self.s.m.extract(c, dm, 1);
                        let half = self.s.m.f32(0.5f32.to_bits());
                        let clamp_pair = |l: &mut Self, coord: V, dim: V, off: V| {
                            let df = l.s.m.cast(c, Cast::UiToFp, dim, f32t);
                            let p = l.s.m.bin(c, Bin::Mul, coord, df);
                            let p = l.s.m.bin(c, Bin::Sub, p, half);
                            let fl = l.s.op(c, "unary.f32", Attr::ReadNone, f32t, 27, &[p]);
                            let t0 = l.s.m.cast(c, Cast::FpToSi, fl, i32t);
                            let t0 = l.s.m.bin(c, Bin::Add, t0, off);
                            let t1 = l.s.m.bin(c, Bin::Add, t0, one);
                            let hi = l.s.m.bin(c, Bin::Sub, dim, one);
                            let mut cl = |x: V| {
                                let lo_c = l.s.m.cmp(c, pred::SLT, x, zero);
                                let x = l.s.m.select(c, lo_c, zero, x);
                                let hi_c = l.s.m.cmp(c, pred::SGT, x, hi);
                                l.s.m.select(c, hi_c, hi, x)
                            };
                            (cl(t0), cl(t1))
                        };
                        let (x0, x1) = clamp_pair(self, cs[0], w, os[0]);
                        let (y0, y1) = clamp_pair(self, cs[1], ht, os[1]);
                        let ch = dmask.trailing_zeros();
                        let Operand::Vector { reg, .. } = *data else {
                            return Err("an image result");
                        };
                        let mut xs = Vec::with_capacity(4);
                        for (xi, yi) in [(x0, y1), (x1, y1), (x1, y0), (x0, y0)] {
                            let tl = self.s.op(
                                c,
                                "textureLoad.f32",
                                Attr::ReadOnly,
                                self.s.t.res_ret,
                                66,
                                &[h, mip, xi, yi, iu, iu, iu, iu],
                            );
                            let x = self.s.m.extract(c, tl, ch);
                            xs.push(if *d16 {
                                self.s
                                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[x])
                            } else {
                                self.as_int(x)
                            });
                        }
                        if *d16 {
                            let k16 = self.s.m.i32(16);
                            xs = xs
                                .chunks(2)
                                .map(|pair| {
                                    let lo = pair[0];
                                    match pair.get(1) {
                                        Some(&hi) => {
                                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                                            self.s.m.bin(c, Bin::Or, lo, hi)
                                        }
                                        None => lo,
                                    }
                                })
                                .collect();
                        }
                        for (k, x) in xs.into_iter().enumerate() {
                            self.write(inst, Reg::V(reg + k as u8), x, guard);
                        }
                        return Ok(());
                    }
                    args.extend(&os[..2]);
                    args.push(self.s.m.i32(dmask.trailing_zeros()));
                    let name = if op == 74 {
                        let reference = float(self, (usize::from(f.o), None))?;
                        args.push(reference);
                        "textureGatherCmp.f32"
                    } else {
                        "textureGather.f32"
                    };
                    self.s
                        .op(c, name, Attr::ReadOnly, self.s.t.res_ret, op, &args)
                }
            }
        } else {
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, 0, k)?;
            }
            let mip = if name == "image_load_mip" {
                addr_val(self, 0, coords)?
            } else if uav {
                // Our declarations have four channels.
                self.s.flags |= flags::TYPED_UAV_LOADS;
                iu
            } else {
                self.s.m.i32(0)
            };
            let mut args = vec![h, mip];
            args.extend(&cs);
            args.extend([iu, iu, iu]);
            self.s.op(
                c,
                "textureLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                66,
                &args,
            )
        };
        let Operand::Vector { reg, .. } = *data else {
            return Err("an image result");
        };
        let channels: Vec<u32> = if gather {
            vec![0, 1, 2, 3]
        } else {
            (0..4).filter(|ch| dmask >> ch & 1 != 0).collect()
        };
        let mut xs = Vec::new();
        for ch in channels {
            let x = self.s.m.extract(c, r, ch);
            xs.push(if *d16 {
                // shortcut: d16 results are taken as floats (rounded to
                // nearest even); integer formats would keep 16 bits.
                let i32t = self.s.t.i32;
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[x])
            } else {
                self.as_int(x)
            });
        }
        if *d16 {
            // Two channels per VGPR, low half first (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            xs = xs
                .chunks(2)
                .map(|p| match *p {
                    [lo, hi] => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, lo, hi)
                    }
                    _ => p[0],
                })
                .collect();
        }
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// The direction a cube sample's face coordinates stand for: the
    /// console takes s and t in [1, 2] (`v_cubesc`/`v_cubetc` over
    /// `|v_cubema|`, plus 1.5) and the face id (`v_cubeid`), which the
    /// host's cube sampling turns back into the same texel.
    // shortcut: the view is taken as a cube, not a cube array (whose face
    // operand adds 8 × the slice).
    fn cube_direction(&mut self, s: V, t: V, face: V) -> [V; 3] {
        let c = self.cur;
        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
        let (two, three) = (k(self, 2.0), k(self, 3.0));
        let u = self.s.m.bin(c, Bin::Mul, s, two);
        let u = self.s.m.bin(c, Bin::Sub, u, three);
        let v = self.s.m.bin(c, Bin::Mul, t, two);
        let v = self.s.m.bin(c, Bin::Sub, v, three);
        let z = self.s.m.f32(0x8000_0000);
        let (nu, nv) = (
            self.s.m.bin(c, Bin::Sub, z, u),
            self.s.m.bin(c, Bin::Sub, z, v),
        );
        let (one, mone) = (k(self, 1.0), k(self, -1.0));
        let f = self.s.m.cast(c, Cast::FpToUi, face, self.s.t.i32);
        // Per face +x, -x, +y, -y, +z, -z.
        let table = [
            [one, mone, u, u, u, nu],
            [nv, nv, one, mone, nv, nv],
            [nu, u, v, nv, one, mone],
        ];
        table.map(|row| {
            let mut r = row[0];
            for (n, &x) in row.iter().enumerate().skip(1) {
                let kn = self.s.m.i32(n as u32);
                let is = self.s.m.cmp(c, pred::EQ, f, kn);
                r = self.s.m.select(c, is, x, r);
            }
            r
        })
    }

    fn raw_dword(&mut self, h: V, coords: [V; 2]) -> V {
        let c = self.cur;
        let (mask, align) = (self.s.m.i8(1), self.s.m.i32(4));
        let r = self.s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, self.s.t.res_ret, 139,
            &[h, coords[0], coords[1], mask, align]);
        let x = self.s.m.extract(c, r, 0);
        self.as_int(x)
    }

    fn buffer_compare_exchange(&mut self, h: V, coords: [V; 3], old: V, new: V, wide: bool) -> V {
        let name = if wide { "atomicCompareExchange.i64" } else { "atomicCompareExchange.i32" };
        let ty = self.s.m.type_of(old);
        self.s.op(self.cur, name, Attr::NoUnwind, ty, 79,
            &[h, coords[0], coords[1], coords[2], old, new])
    }

    /// A CAS must run on a real lane and at an in-range address: outside
    /// those cases DXIL can return an unspecified old value.
    fn buffer_bounds(&mut self, h: V, index: Option<V>, off: V, stride: u32, size: u32) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let unused = self.s.m.undef(i32t);
        let dimensions = self.s.op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, unused]);
        let length = self.s.m.extract(c, dimensions, 0);
        if let Some(index) = index {
            let record = self.s.m.cmp(c, pred::IULT, index, length);
            let limit = self.s.m.i32(stride - (size - 1));
            let member = self.s.m.cmp(c, pred::IULT, off, limit);
            return self.s.m.bin(c, Bin::And, record, member);
        }
        let tail = self.s.m.i32(size - 1);
        let last = self.s.m.bin(c, Bin::Add, off, tail);
        let fits = self.s.m.cmp(c, pred::IULT, last, length);
        let no_wrap = self.s.m.cmp(c, pred::IUGE, last, off);
        self.s.m.bin(c, Bin::And, fits, no_wrap)
    }

    fn buffer_store_guard(&mut self, bounds: V, guard: Option<V>) -> V {
        let c = self.cur;
        let g = guard.map_or(bounds, |g| self.s.m.bin(c, Bin::And, g, bounds));
        if !self.s.pixel {
            return g;
        }
        let helper = self.s.op(c, "isHelperLane.i1", Attr::ReadOnly, self.s.t.i1, 221, &[]);
        let no = self.s.m.bool(false);
        self.s.m.select(c, helper, no, g)
    }

    fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err("an MTBUF store requiring a packed-format write");
        }
        if name.contains("_format_d16_hi_") {
            return Err("a typed buffer access to the high half with an untracked numeric format");
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let store = name.starts_with("buffer_store_");
        let atomic = name.strip_prefix("buffer_atomic_");
        let wide = atomic.is_some_and(|n| n.ends_with("_x2"));
        let atomic = atomic.map(|n| n.strip_suffix("_x2").unwrap_or(n));
        // DXIL atomic operations: add, and, or, xor, signed and unsigned
        // min and max, exchange; compare-exchange apart.
        let native_atomic = match atomic {
            None => None,
            Some("add" | "sub") => Some(0),
            Some("and") => Some(1),
            Some("or") => Some(2),
            Some("xor") => Some(3),
            Some("smin") => Some(4),
            Some("smax") => Some(5),
            Some("umin") => Some(6),
            Some("umax") => Some(7),
            Some("swap") => Some(8),
            Some("cmpswap" | "inc" | "dec" | "csub") => None,
            Some("fcmpswap" | "fmin" | "fmax") => {
                // Mesa aco_insert_fp_mode.cpp, vmem_default_needs.
                return Err("a buffer floating-point atomic requiring the console's denormal mode");
            }
            Some(_) => return Err(name),
        };
        let sub_load = name.strip_prefix("buffer_load_").and_then(subword_load);
        let sub_store = match name {
            "buffer_store_byte" | "buffer_store_byte_d16_hi" => Some(8),
            "buffer_store_short" | "buffer_store_short_d16_hi" => Some(16),
            _ => None,
        };
        // d16: two 16-bit channels per VGPR, low half first.
        let d16 = name.starts_with("buffer_") && name.contains("_format_d16_");
        let plain = name.replace("_format_d16_", "_format_");
        let (typed, count) = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() || sub_load.is_some() || sub_store.is_some() => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            _ => return Err(name),
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let mut off = self.s.m.i32(u32::from(*offset));
        if *offen {
            let o = vaddr(self, u8::from(*idxen))?;
            off = self.s.m.bin(c, Bin::Add, off, o);
        }
        if !matches!(
            soffset,
            Operand::Scalar { code: NULL, .. } | Operand::Const(128)
        ) {
            let so = self.operand(inst, soffset)?;
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked buffer")?;
        let used = self.used_kind(src);
        let stride = used.map_or(0, |u| u.stride);
        let (base, extra) = self.built_offset(src)?;
        if base != 0 {
            let b = self.s.m.i32(base);
            off = self.s.m.bin(c, Bin::Add, off, b);
        }
        if let Some(x) = extra {
            off = self.s.m.bin(c, Bin::Add, off, x);
        }
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            if store || atomic.is_some() {
                return Err("a write or atomic operation on a constant buffer");
            }
            // A constant buffer read per lane (a dynamically indexed array).
            // shortcut: the console's CBV descriptor is taken to have a 16-byte
            // stride, as the vector loads of whole rows suggest.
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            let byte = match index {
                Some(i) => {
                    let sixteen = self.s.m.i32(16);
                    let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                    self.s.m.bin(c, Bin::Add, rows, off)
                }
                None => off,
            };
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            if let Some((width, signed, half)) = sub_load {
                let x = self.read_subword(byte, width, signed, |l, at| Ok(l.cbuffer_dword(h, at)))?;
                self.write_subword_result(inst, reg, x, half, guard)?;
                return Ok(());
            }
            for k in 0..count {
                let four = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, byte, four);
                let x = self.cbuffer_dword(h, a);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let uav = store || atomic.is_some() || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let iu = self.s.m.undef(self.s.t.i32);
        if sub_load.is_some() || sub_store.is_some() {
            let data = data.as_ref().ok_or("buffer sub-dword data")?;
            let Operand::Vector { reg, .. } = *data else {
                return Err("buffer sub-dword data");
            };
            let h = if index.is_some() {
                if stride < 4 || !stride.is_multiple_of(4) {
                    return Err("a structured sub-dword buffer access without a dword-aligned stride");
                }
                self.handle(site, src, false, (12 | u, stride))?
            } else {
                self.handle(site, src, false, (11 | u, 0))?
            };
            let coords = |byte| index.map_or([byte, iu], |i| [i, byte]);
            if let Some((width, signed, half)) = sub_load {
                // Root SRVs have no size in DXIL.
                let bounds = (!matches!(src, Source::Built { .. }))
                    .then(|| self.buffer_bounds(h, index, off, stride, width / 8));
                let at = self.begin_guarded(bounds);
                let x = self.read_subword(off, width, signed, |l, at| Ok(l.raw_dword(h, coords(at))))?;
                let zero = self.s.m.i32(0);
                let x = self.end_guarded(at, &[(x, zero)])[0];
                self.write_subword_result(inst, reg, x, half, guard)?;
            } else if let Some(width) = sub_store {
                let mut x = self.vgprs(inst, data, 1)?[0];
                if name.ends_with("_hi") {
                    let sixteen = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, sixteen);
                }
                let bounds = self.buffer_bounds(h, index, off, stride, width / 8);
                let active = self.buffer_store_guard(bounds, guard);
                let at = self.begin_guarded(Some(active));
                self.write_subword(off, width, x, |l, byte, mask, bits| {
                    let [a, b] = coords(byte);
                    let zero = l.s.m.i32(0);
                    l.atomic_update(zero, |l, old| {
                        let x = l.atomic_new_value(AtomicUpdate::MaskOr, old, mask, bits);
                        l.buffer_compare_exchange(h, [a, b, iu], old, x, false)
                    });
                });
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if let Some(op) = atomic {
            let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
            if op == "csub" && !returns {
                // LLVM BUFInstructions.td only selects the returning form
                // on GFX10.3.
                return Err("buffer_atomic_csub without the required GLC bit");
            }
            let d = data.as_ref().ok_or("buffer atomic data")?;
            let words = if wide { 2 } else { 1 };
            let vals = self.vgprs(inst, d, if op == "cmpswap" { 2 * words } else { words })?;
            let val = if wide { self.join64(vals[0], vals[1]) } else { vals[0] };
            let ty = self.s.m.type_of(val);
            let zero = self.s.m.int_const(ty, 0);
            let (h, coords) = match index {
                Some(index) => {
                    if stride >= 4 && stride.is_multiple_of(4) {
                        if wide && !stride.is_multiple_of(8) {
                            return Err(
                                "a 64-bit buffer atomic without an eight-byte-aligned stride",
                            );
                        }
                        (
                            self.handle(site, src, false, (12 | u, stride))?,
                            [index, off, iu],
                        )
                    } else {
                        let byte = self.runtime_stride_offset(src, index, off)?;
                        (self.handle(site, src, false, (11 | u, 0))?, [byte, iu, iu])
                    }
                }
                None => (self.handle(site, src, false, (11 | u, 0))?, [off, iu, iu]),
            };
            let Operand::Vector { reg, .. } = *d else {
                return Err("buffer atomic data");
            };
            if wide {
                self.s.flags |= flags::INT64 | flags::ATOMIC_INT64_HEAP;
            }
            let active = if wide || matches!(op, "sub" | "inc" | "dec" | "csub") {
                let bounds = self.buffer_bounds(h, index, off, stride, 4 * u32::from(words));
                Some(self.buffer_store_guard(bounds, guard))
            } else {
                guard
            };
            let at = self.begin_guarded(active);
            let r = if op == "cmpswap" {
                // cmpswap: data is (new, compare).
                let compare = if wide { self.join64(vals[2], vals[3]) } else { vals[1] };
                self.buffer_compare_exchange(h, coords, compare, val, wide)
            } else if let Some(code) = native_atomic {
                let name = if wide { "atomicBinOp.i64" } else { "atomicBinOp.i32" };
                let code = self.s.m.i32(code);
                let data = if op == "sub" { self.s.m.bin(self.cur, Bin::Sub, zero, val) } else { val };
                self.s.op(self.cur, name, Attr::NoUnwind, ty, 78,
                    &[h, code, coords[0], coords[1], coords[2], data])
            } else {
                let update = match op {
                    "inc" => AtomicUpdate::Inc,
                    "dec" => AtomicUpdate::Dec,
                    "csub" => AtomicUpdate::ClampSub,
                    _ => unreachable!(),
                };
                self.atomic_update(zero, |l, old| {
                    let x = l.atomic_new_value(update, old, val, zero);
                    l.buffer_compare_exchange(h, coords, old, x, wide)
                })
            };
            if returns {
                let r = self.end_guarded(at, &[(r, zero)])[0];
                self.write_atomic_result(inst, reg, r, wide, guard);
            } else {
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if store {
            let regs = if d16 { count.div_ceil(2) } else { count };
            let mut vals =
                self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, regs as u8)?;
            if d16 {
                // shortcut: d16 data is taken as f16 (integer formats would
                // keep 16 bits), widened for the 32-bit float store.
                let mut wide = Vec::new();
                for k in 0..count {
                    let mut x = vals[(k / 2) as usize];
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                    wide.push(self.as_int(f));
                }
                vals = wide;
            }
            let mut args = Vec::new();
            let (op, name) = if typed {
                // shortcut: typed stores are taken to write all four channels
                // of a float format.
                if count != 4 || *offen || *offset != 0 || matches!(off, V::Inst(_)) {
                    return Err("a typed buffer store of some channels or at an offset");
                }
                args.push(self.handle(site, src, false, (10 | u, 0x409))?);
                args.extend([index.unwrap_or(self.s.m.i32(0)), iu]);
                for x in vals {
                    args.push(self.as_float(x));
                }
                (69, "bufferStore.f32")
            } else {
                match index {
                    Some(index) => {
                        let stride = used.map_or(0, |u| u.stride);
                        if stride >= 4 && stride.is_multiple_of(4) {
                            args.push(self.handle(site, src, false, (12 | u, stride))?);
                            args.extend([index, off]);
                        } else {
                            let byte = self.runtime_stride_offset(src, index, off)?;
                            args.push(self.handle(site, src, false, (11 | u, 0))?);
                            args.extend([byte, iu]);
                        }
                    }
                    None => {
                        args.push(self.handle(site, src, false, (11 | u, 0))?);
                        args.extend([off, iu]);
                    }
                }
                args.extend(&vals);
                args.resize(3 + 4, iu);
                (140, "rawBufferStore.i32")
            };
            args.push(self.s.m.i8(if typed { 0xF } else { (1u8 << count) - 1 }));
            if !typed {
                args.push(self.s.m.i32(4));
            }
            let at = self.begin_guarded(guard);
            let void = self.s.t.void;
            self.s.op(self.cur, name, Attr::NoUnwind, void, op, &args);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if let Some(f) = *format {
            if index.is_some() {
                return Err("an indexed MTBUF load requiring a raw view of a structured descriptor");
            }
            let h = self.handle(site, src, false, (11 | u, 0))?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for (k, x) in self.format_load(h, off, f, count)?.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let r = if typed {
            if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            if uav {
                self.s.flags |= flags::TYPED_UAV_LOADS;
            }
            let h = self.handle(site, src, false, (10 | u, 0x409))?;
            let index = index.unwrap_or(self.s.m.i32(0));
            self.s.op(
                c,
                "bufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                68,
                &[h, index, iu],
            )
        } else {
            let mask = self.s.m.i8((1u8 << count) - 1);
            let align = self.s.m.i32(4);
            match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride >= 4 && stride.is_multiple_of(4) {
                        let h = self.handle(site, src, false, (12 | u, stride))?;
                        self.s.op(
                            c,
                            "rawBufferLoad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            139,
                            &[h, index, off, mask, align],
                        )
                    } else {
                        let byte = self.runtime_stride_offset(src, index, off)?;
                        let h = self.handle(site, src, false, (11 | u, 0))?;
                        self.s.op(
                            c,
                            "rawBufferLoad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            139,
                            &[h, byte, iu, mask, align],
                        )
                    }
                }
                None => {
                    let h = self.handle(site, src, false, (11 | u, 0))?;
                    self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, off, iu, mask, align],
                    )
                }
            }
        };
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a buffer result");
        };
        if d16 {
            // Two f16 channels per VGPR (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            let i32t = self.s.t.i32;
            for k in 0..count.div_ceil(2) {
                let mut x = None;
                for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                    let f = self.s.m.extract(c, r, ch);
                    let h = self
                        .s
                        .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
                    x = Some(match x {
                        Some(hi) => {
                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                            self.s.m.bin(c, Bin::Or, h, hi)
                        }
                        None => h,
                    });
                }
                self.write(inst, Reg::V(reg + k as u8), x.expect("a channel"), guard);
            }
            return Ok(());
        }
        for k in 0..count {
            let x = self.s.m.extract(c, r, k);
            let x = self.as_int(x);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    fn export(&mut self, b: usize, i: usize) -> Result<(), Why> {
        let e = *self
            .exports
            .get(&(b, i))
            .ok_or("an export I/O raising did not see")?;
        // Two f16 channels per source dword: only the pixel color and
        // depth targets unpack them; the rest stay refused.
        if e.compressed && !matches!(e.target, Target::Color(_) | Target::Depth) {
            return Err("a compressed export");
        }
        let guard = self.guard(b, i)?;
        let mesh = self.s.mesh.is_some();
        let c = self.cur;
        if let Some(g) = guard
            && !mesh
        {
            if !(e.done && self.s.pixel) {
                return Err("a conditional export");
            }
            let t = self.s.m.bool(true);
            let kill = self.s.m.bin(c, Bin::Xor, g, t);
            self.s
                .op(c, "discard", Attr::NoUnwind, self.s.t.void, 82, &[kill]);
        }
        if mesh && e.target == Target::Primitive {
            return self.emit_indices(e, guard);
        }
        let param_row = |usage: Usage, row: u8| match usage {
            // The 2020 compiler leaves a mesh program's first entry zeroed
            // (usage "position"): its row is the one no other entry names.
            Usage::Other(0) if mesh => self.unnamed_row(),
            _ => Some(row),
        };
        // (output element, channel) per exported channel; per-primitive
        // outputs are elements of `prims`.
        let mut stores = Vec::new();
        let primitive = matches!(e.target, Target::PrimitiveParam { .. });
        let outputs = if primitive {
            &self.s.prims
        } else {
            &self.s.outputs
        };
        let by_kind = |kind: u8| outputs.iter().position(|o| o.kind == kind);
        // 16-bit vertex outputs: the halves each dword carries, as (output
        // element, channel, high half) per channel.
        let halves = match e.target {
            Target::Param { usage, row } => {
                param_row(usage, row).and_then(|r| self.half_outputs.get(&r).map(|&h| (r, h)))
            }
            _ => None,
        };
        if let Some((row, (pack2, b))) = halves {
            for ch in 0..4u8 {
                let Some(value) = e.values[usize::from(ch)] else {
                    continue;
                };
                for high in [false, true] {
                    let h = 2 * ch + u8::from(high);
                    let src = if pack2 {
                        packed_half(outputs, row, b, h)
                    } else {
                        (!high).then_some((row, ch))
                    };
                    let found = src.and_then(|(r, c)| {
                        outputs.iter().position(|o| {
                            o.row == Some(u32::from(r)) && c >= o.col && c < o.col + o.cols
                        })
                    });
                    if let Some(id) = found {
                        let col = src.expect("found").1 - outputs[id].col;
                        stores.push((id, col, value, Some(high)));
                    }
                }
            }
        }
        for ch in 0..4u8 {
            let Some(value) = e.values[usize::from(ch)] else {
                continue;
            };
            if halves.is_some() {
                break;
            }
            let found = match e.target {
                Target::Color(n) => outputs
                    .iter()
                    .position(|o| o.kind == 16 && o.row == Some(u32::from(n)))
                    .map(|id| (id, ch)),
                Target::Position(0) => by_kind(3).map(|id| (id, ch)),
                Target::Depth => match ch {
                    0 => by_kind(17)
                        .or(by_kind(18))
                        .or(by_kind(19))
                        .map(|id| (id, 0)),
                    1 => by_kind(20).map(|id| (id, 0)),
                    2 => by_kind(14).map(|id| (id, 0)),
                    _ => None,
                },
                Target::Param { usage, row } | Target::PrimitiveParam { usage, row } => {
                    let row = param_row(usage, row);
                    outputs
                        .iter()
                        .position(|o| {
                            row.is_some_and(|r| o.row == Some(u32::from(r)))
                                && ch >= o.col
                                && ch < o.col + o.cols
                        })
                        .map(|id| (id, ch - outputs[id].col))
                }
                Target::Null | Target::Unread => None,
                Target::Position(n @ 1..=2) => {
                    // shortcut: without the misc vector (point size, render
                    // target or viewport index) pos1 and pos2 carry the clip
                    // then cull distances, in semantic order.
                    if outputs.iter().any(|o| matches!(o.kind, 4 | 5)) {
                        return Err("render-target or viewport index exports");
                    }
                    let mut dists: Vec<usize> = (0..outputs.len())
                        .filter(|&i| matches!(outputs[i].kind, 6 | 7))
                        .collect();
                    dists.sort_by_key(|&i| (outputs[i].kind, outputs[i].index));
                    let flat: Vec<(usize, u8)> = dists
                        .iter()
                        .flat_map(|&i| (0..outputs[i].cols).map(move |c| (i, c)))
                        .collect();
                    // Channels past the distances the signature declares are
                    // ones the clipper does not enable (`PA_CL_VS_OUT_CNTL`).
                    let k = usize::from(n - 1) * 4 + usize::from(ch);
                    flat.get(k).copied()
                }
                Target::Position(_) => return Err("clip, cull or point-size exports"),
                Target::Primitive => return Err("a primitive export"),
            };
            if let Some((id, col)) = found {
                // Two f16 channels per dword: an odd channel the high half.
                let half = e.compressed.then_some(ch % 2 == 1);
                stores.push((id, col, value, half));
            }
        }
        fn elements<'l>(l: &'l mut Lower, primitive: bool) -> &'l mut Vec<SigElement> {
            if primitive {
                &mut l.s.prims
            } else {
                &mut l.s.outputs
            }
        }
        if stores
            .iter()
            .any(|s| s.3.is_none() && elements(self, primitive)[s.0].half)
        {
            return Err("a 16-bit output the stage data does not mark");
        }
        // A mesh lane writes its own vertex or primitive, where it exports.
        let at = if mesh {
            self.begin_guarded(guard)
        } else {
            None
        };
        let (name, opcode) = match (mesh, primitive) {
            (false, _) => ("storeOutput", 5),
            (true, false) => ("storeVertexOutput", 171),
            (true, true) => ("storePrimitiveOutput", 172),
        };
        for (id, col, value, half) in stores {
            let mut x = self.i32_of(value)?;
            if let Some(high) = half {
                // An f16 half, declared a 32-bit float on the host.
                let c = self.cur;
                if high {
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, k16);
                }
                let f32t = self.s.t.f32;
                let f = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                x = self.as_int(f);
            }
            let el = &mut elements(self, primitive)[id];
            el.usage |= 1 << col;
            let (suffix, x) = match el.comp {
                9 if half.is_some() => (".f32", self.as_float(x)),
                9 => {
                    let f = match self.floats.get(&value) {
                        Some(&f) => f,
                        None => self.as_float(x),
                    };
                    (".f32", f)
                }
                _ => (".i32", x),
            };
            let mut args = vec![
                self.s.m.i32(id as u32),
                self.s.m.i32(0),
                self.s.m.i8(col),
                x,
            ];
            if mesh {
                args.push(self.mesh_thread());
            }
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                &format!("{name}{suffix}"),
                Attr::NoUnwind,
                void,
                opcode,
                &args,
            );
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: `exp prim` as the lane's primitive. Bit 31 culls it (all
    /// indices 0: no area).
    fn emit_indices(&mut self, e: &Export, guard: Option<V>) -> Result<(), Why> {
        let v = e.values[0].ok_or("a primitive export without its indices")?;
        let x = self.i32_of(v)?;
        let at = self.begin_guarded(guard);
        let c = self.cur;
        let z = self.s.m.i32(0);
        let culled = self.s.m.cmp(c, pred::SLT, x, z);
        let mut args = vec![self.mesh_thread()];
        for k in 0..3 {
            let (shift, mask) = (self.s.m.i32(10 * k), self.s.m.i32(0x3FF));
            let i = self.s.m.bin(c, Bin::LShr, x, shift);
            let i = self.s.m.bin(c, Bin::And, i, mask);
            args.push(self.s.m.select(c, culled, z, i));
        }
        let void = self.s.t.void;
        self.s
            .op(c, "emitIndices", Attr::NoUnwind, void, 169, &args);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// The per-vertex arbitrary row no mesh output entry names but one
    /// (the 2020 compiler's zeroed entry).
    fn unnamed_row(&self) -> Option<u8> {
        let StageIo::Ms(ms) = &self.x.native.io else {
            return None;
        };
        let named = |r: u32| {
            ms.outputs
                .iter()
                .any(|o| o.param & 0x80 == 0 && o.usage != Usage::Other(0) && u32::from(o.row) == r)
        };
        let free: std::collections::BTreeSet<u32> = self
            .s
            .outputs
            .iter()
            .filter(|o| o.kind == 0)
            .filter_map(|o| o.row)
            .filter(|&r| !named(r))
            .collect();
        match free.into_iter().collect::<Vec<_>>()[..] {
            [r] => u8::try_from(r).ok(),
            _ => None,
        }
    }

    fn has_active_barrier(&self, b: usize) -> bool {
        !self.no_barrier.contains(&b)
            && self.x.p.blocks[b]
                .insts
                .iter()
                .any(|i| i.name() == "s_barrier")
    }

    fn is_back_edge(&self, from: usize, to: usize) -> bool {
        self.rpo[to] <= self.rpo[from]
    }

    fn exit(&mut self, b: usize) -> Result<(), Stop> {
        if self.emulating_wave64()
            && let LaneExit::Branch { when, .. } = self.x.st.exits[b]
        {
            let mut todo = Vec::new();
            for a in self.x.st.bools.support(when) {
                if let Atom::Opaque(v) = a {
                    let lo = self.x.st.highs.get(&v).copied().unwrap_or(v);
                    if self.x.st.masks.contains_key(&lo)
                        && !self.ballots.contains_key(&lo)
                        && !self.ballot_fail.contains_key(&lo)
                        && !todo.contains(&lo)
                    {
                        todo.push(lo);
                    }
                }
            }
            let uniform_here = self.uniform.get(b).copied().unwrap_or(false);
            for lo in todo {
                if let Err(why) = self.compute_ballot_in_cur(lo, uniform_here) {
                    self.ballot_fail.insert(lo, why);
                }
            }
            self.tail[b] = Some(self.cur);
        }
        let offset = self.x.p.blocks[b]
            .insts
            .last()
            .map_or(self.x.p.blocks[b].offset, |i| i.machine.offset);
        let term = match self.x.st.exits[b] {
            LaneExit::Jump(t) if self.is_back_edge(b, t) => {
                let kind = self.loop_kind(t);
                let (ok, next) = self.step_back_edge(kind, pred::IULT);
                self.push_step_edge(t, self.cur, Some((kind, next)));
                let abort = self.abort_block();
                Term::CondBr(ok, self.block[t].expect("reachable"), abort)
            }
            LaneExit::Jump(t) => {
                self.push_step_edge(t, self.cur, None);
                Term::Br(self.block[t].expect("reachable"))
            }
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                // A negated condition branches the other way round.
                let (mut cond, taken, not_taken, swapped) = match self.x.st.bools.node(when) {
                    Some((atom, Bool::TRUE, Bool::FALSE)) => {
                        let a = self
                            .atom(atom, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (a, not_taken, taken, true)
                    }
                    _ => {
                        let c = self
                            .boolean(when, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken, false)
                    }
                };
                // Keep branches controlling an active `s_barrier` wave-uniform
                // so individual lanes never diverge across `dx.op.barrier`.
                let any_active_barrier = (0..self.x.p.blocks.len())
                    .any(|u| self.block[u].is_some() && self.has_active_barrier(u));
                if let Some(orig_all) = self.x.st.barrier_branches[b]
                    && self.s.threads.is_some()
                    && any_active_barrier
                {
                    let all = if swapped { !orig_all } else { orig_all };
                    let (name, opcode) = if all {
                        ("waveAllTrue", 114)
                    } else {
                        ("waveAnyTrue", 113)
                    };
                    self.s.flags |= flags::WAVE_OPS;
                    cond = self
                        .s
                        .op(self.cur, name, Attr::NoUnwind, self.s.t.i1, opcode, &[cond]);
                }
                // A back edge is taken only below the invocation step budget.
                let (bt, bf) = (self.is_back_edge(b, taken), self.is_back_edge(b, not_taken));
                // (No `xor` of a compare: a loop exit testing one hung the
                // AMD driver.)
                let (cond, upd_t, upd_f) = if bt && bf {
                    let kt = self.loop_kind(taken);
                    let kf = self.loop_kind(not_taken);
                    let (ok, ut, uf) = if kt == kf {
                        let (ok, next) = self.step_back_edge(kt, pred::IULT);
                        (ok, Some((kt, next)), Some((kt, next)))
                    } else {
                        let (ok_t, next_t) = self.step_back_edge(kt, pred::IULT);
                        let (ok_f, next_f) = self.step_back_edge(kf, pred::IULT);
                        let ok = self.s.m.select(self.cur, cond, ok_t, ok_f);
                        (ok, Some((kt, next_t)), Some((kf, next_f)))
                    };
                    let cont = self.s.m.new_block();
                    let abort = self.abort_block();
                    self.s.m.terminate(self.cur, Term::CondBr(ok, cont, abort));
                    self.cur = cont;
                    self.tail[b] = Some(cont);
                    (cond, ut, uf)
                } else if bt {
                    let kind = self.loop_kind(taken);
                    let (ok, next) = self.step_back_edge(kind, pred::IULT);
                    let cond = self.s.m.bin(self.cur, Bin::And, cond, ok);
                    (cond, Some((kind, next)), Some((kind, next)))
                } else if bf {
                    let kind = self.loop_kind(not_taken);
                    let (stop, next) = self.step_back_edge(kind, pred::IUGE);
                    let cond = self.s.m.bin(self.cur, Bin::Or, cond, stop);
                    (cond, Some((kind, next)), Some((kind, next)))
                } else {
                    (cond, None, None)
                };
                // Critical edges get a block of their own, so loops have
                // dedicated exits and preheaders as DXC emits them.
                let (t, f) = (self.edge(b, taken), self.edge(b, not_taken));
                for (to, via, upd) in [(taken, t, upd_t), (not_taken, f, upd_f)] {
                    let from = if Some(via) == self.block[to] {
                        self.cur
                    } else {
                        via
                    };
                    self.push_step_edge(to, from, upd);
                }
                Term::CondBr(cond, t, f)
            }
            LaneExit::End => {
                if self.s.pixel && !self.exported[b] && !self.has_done_export(b) {
                    // A lane leaving without its final export was killed.
                    let t = self.s.m.bool(true);
                    self.s
                        .op(self.cur, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
                }
                Term::Ret
            }
            LaneExit::Indirect => return Err(unsupported(offset, "an indirect jump")),
        };
        let _ = Exit::End;
        self.s.m.terminate(self.cur, term);
        Ok(())
    }

    fn loop_kind(&self, h: usize) -> usize {
        if self.barrier_loops.contains(&h) {
            1
        } else if self.spin.contains(&h) {
            2
        } else {
            0
        }
    }

    /// Increments step counter `kind` at a back-edge and compares against
    /// [`Self::loop_limit`] (or [`Self::spin_limit`] for spin-wait loops) by
    /// `p` (below: continue; at or above: stop).
    fn step_back_edge(&mut self, kind: usize, p: u8) -> (V, V) {
        let c = self.cur;
        let cur = self.steps[kind]
            .as_ref()
            .and_then(|t| t.cur_val)
            .expect("step counter");
        let one = self.s.m.i32(1);
        let next = self.s.m.bin(c, Bin::Add, cur, one);
        let raw_limit = if kind == 2 {
            #[cfg(test)]
            {
                self.spun += 1;
            }
            self.spin_limit
        } else {
            self.loop_limit
        };
        let limit = self.s.m.i32(raw_limit);
        let cmp = self.s.m.cmp(c, p, next, limit);
        (cmp, next)
    }

    /// Records the outgoing step count on an edge to IR block `to` from DXIL
    /// block `from`, overriding counter `upd.0` when the edge stepped a loop.
    fn push_step_edge(&mut self, to: usize, from: usize, upd: Option<(usize, V)>) {
        for (k, slot) in self.steps.iter_mut().enumerate() {
            let Some(track) = slot else { continue };
            let val = match upd {
                Some((uk, next)) if uk == k => next,
                _ => track.cur_val.expect("step counter"),
            };
            track.incoming[to].push((val, from));
        }
    }

    /// The block a lane stopped by [`LOOP_LIMIT`] leaves through.
    fn abort_block(&mut self) -> usize {
        if let Some(a) = self.abort {
            return a;
        }
        let a = self.s.m.new_block();
        if self.s.pixel {
            let t = self.s.m.bool(true);
            self.s
                .op(a, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
        }
        self.s.m.terminate(a, Term::Ret);
        self.abort = Some(a);
        a
    }

    /// The DXIL block a branch from IR block `from` to `to` goes to.
    fn edge(&mut self, from: usize, to: usize) -> usize {
        let target = self.block[to].expect("reachable");
        let preds = self.x.p.blocks[to]
            .preds
            .iter()
            .filter(|&&q| self.block[q].is_some())
            .count();
        // Back edges stay as they are: the exiting block is the latch.
        if preds < 2 || self.rpo[to] <= self.rpo[from] {
            return target;
        }
        let e = self.s.m.new_block();
        self.s.m.terminate(e, Term::Br(target));
        self.edges.insert((from, to), e);
        e
    }

    fn fill_phis(&mut self) -> Result<(), Stop> {
        let p = self.x.p;
        while let Some((v, b, i)) = self.pending_masks.pop() {
            let phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let args = self
                .x
                .st
                .phi_masks
                .get(&phi.value)
                .ok_or_else(|| unsupported(offset, "a lane-mask join of an untracked mask"))?;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let x = self
                    .boolean(args[n], qb)
                    .map_err(|w| unsupported(offset, w))?;
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        for (v, b, i) in std::mem::take(&mut self.data_phis) {
            let phi = &p.blocks[b].phis[i];
            let bit = phi.reg == Reg::Scc;
            // A register initialized only on one branch may carry a dead
            // descriptor dword left over from earlier register reuse on the
            // other branch (guarded by a separate predicate). Only treat a
            // descriptor-dword `Missing` as fatal when no incoming edge carries
            // a real data value.
            let has_data = !bit
                && p.blocks[b]
                    .preds
                    .iter()
                    .enumerate()
                    .filter(|&(_, &q)| self.tail[q].is_some())
                    .any(|(n, _)| matches!(self.vals.get(&phi.args[n]), Some(Val::I32(_))));
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let arg = phi.args[n];
                if self.read_phis.contains(&phi.value) && !self.vals.contains_key(&arg) {
                    let _ = self.lazy_barycentric(arg);
                }
                let x = match (self.vals.get(&arg).copied(), bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                    (Some(Val::Missing(why)), false)
                        if has_data
                            && matches!(
                                why,
                                "a descriptor's address or format read as data"
                                    | "a descriptor read only as data"
                            ) =>
                    {
                        self.s.m.i32(0)
                    }
                    (Some(Val::Bit(_)), false) if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, "an SCC joined with SGPR values"));
                    }
                    (Some(Val::Missing(why)), _) if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, why));
                    }
                    _ if self.read_phis.contains(&phi.value) => {
                        return Err(Stop::Retry(phi.value, "a join of mismatched values"));
                    }
                    _ if bit => self.s.m.bool(false),
                    _ => self.s.m.i32(0),
                };
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        for track in self.steps.iter_mut().flatten() {
            for (b, phi) in track.phis.iter().enumerate() {
                if let Some(phi) = *phi {
                    let incoming = std::mem::take(&mut track.incoming[b]);
                    self.s.m.set_phi(phi, incoming);
                }
            }
        }
        Ok(())
    }
}

/// The DXIL interpolation mode of a kind of barycentrics (none for the
/// pull model).
fn interp_mode(kind: Barycentric) -> Option<u8> {
    Some(match kind {
        Barycentric::PerspectiveCenter => 2,
        Barycentric::PerspectiveCentroid => 3,
        Barycentric::PerspectiveSample => 6,
        Barycentric::LinearCenter => 4,
        Barycentric::LinearCentroid => 5,
        Barycentric::LinearSample => 7,
        Barycentric::PerspectivePullModel => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::{Lowered, Numeric, Parts, buffer_format, lower, plan_lds_barriers};
    use crate::dxbc::{Container, FourCC};
    use crate::emit::{self, HostShader};
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// Lowers a compute program of constants: whether it translates.
    fn lowers(words: &[u32]) -> Result<(), crate::ShaderError> {
        lowers_cs(words, false, 0).map(|_| ())
    }

    fn lowers_cs(
        words: &[u32],
        wave64: bool,
        const_sgprs: u8,
    ) -> Result<super::Lowered, crate::ShaderError> {
        use crate::native::{RootParam, RootType};
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let layout = if const_sgprs > 0 {
            vec![RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: u32::from(const_sgprs),
                base_register: 0,
                user_data: 0,
                slot: 0,
            }]
        } else {
            vec![]
        };
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 256,
            layout,
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: u32::from(const_sgprs) << 1,
                threads: [if wave64 { 64 } else { 32 }, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            options: super::TranslateOptions::default(),
        })
    }

    #[test]
    fn lane_mask_phi_joining_zero_and_compare_lowers() {
        let cnd = 0x81 | (0x82 << 9) | (2 << 18); // v_cndmask_b32 v1, 1, 2, s2
        let words = [
            0xBF06_8000, // 0: s_cmp_eq_u32 s0, 0
            0xBF85_0002, // 4: s_cbranch_scc1 -> 16
            0xBE82_0380, // 8: s_mov_b32 s2, 0
            0xBF82_0002, // 12: s_branch -> 24
            0xD4C2_0002,
            0x0002_0080, // 16: v_cmp_eq_u32 s2, 0, v0
            0xD501_0001,
            cnd, // 24: v_cndmask_b32 v1, 1, 2, s2
            S_ENDPGM,
        ];
        lowers_cs(&words, false, 1).unwrap();
    }

    #[test]
    fn wave64_loop_mask_accumulator_from_zero_lowers() {
        let cnd = 0x81 | (0x82 << 9) | (2 << 18); // v_cndmask_b32 v1, 1, 2, s[2:3]
        let words = [
            0xBE82_0480, // 0: s_mov_b64 s[2:3], 0
            0xBE84_0380, // 4: s_mov_b32 s4, 0
            0xD501_0001,
            cnd,         // 8: v_cndmask_b32 v1, 1, 2, s[2:3]
            0x7D84_0080, // 16: v_cmp_eq_u32 vcc, 0, v0
            0x8882_026A, // 20: s_or_b64 s[2:3], vcc, s[2:3]
            0x8004_8104, // 24: s_add_u32 s4, s4, 1
            0xBF0A_8404, // 28: s_cmp_lt_u32 s4, 4
            0xBF85_FFF9, // 32: s_cbranch_scc1 -> 8
            0xD501_0004,
            cnd, // 36: v_cndmask_b32 v4, 1, 2, s[2:3]
            S_ENDPGM,
        ];
        lowers_cs(&words, true, 0).unwrap();
    }

    #[test]
    fn lane_mask_phi_with_partial_constant_enables_console_lanes() {
        let cnd = 0x81 | (0x82 << 9) | (2 << 18); // v_cndmask_b32 v1, 1, 2, s2
        let words = [
            0xBF06_8000, // 0: s_cmp_eq_u32 s0, 0
            0xBF85_0002, // 4: s_cbranch_scc1 -> 16
            0xBE82_038F, // 8: s_mov_b32 s2, 15 (partial constant mask 0xF)
            0xBF82_0002, // 12: s_branch -> 24
            0xD4C2_0002,
            0x0002_0080, // 16: v_cmp_eq_u32 s2, 0, v0
            0xD501_0001,
            cnd, // 24: v_cndmask_b32 v1, 1, 2, s2
            S_ENDPGM,
        ];
        let lowered = lowers_cs(&words, false, 1).unwrap();
        assert!(
            !lowered.dxil.is_empty(),
            "expected DXIL with WaveSize(32) and WAVE_OPS"
        );
    }

    /// Lowers a compute program with the root parameters `layout` (in user
    /// SGPRs from s0 on) and the used resources `used`.
    fn lower_with(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
    ) -> Result<Lowered, crate::ShaderError> {
        lower_wave(words, layout, used, false)
    }

    /// [`lower_with`] in wave64 if `wave64`.
    fn lower_wave(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
        wave64: bool,
    ) -> Result<Lowered, crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 256,
            layout,
            used,
            io: StageIo::Cs(ComputeIo {
                // The layout's user SGPRs (at least two), 512 bytes of LDS.
                rsrc2: 1 << 15 | sgprs.max(2) << 1,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let res = resource::track(&p, &native.layout, native.stage)?;
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            options: super::TranslateOptions::default(),
        })
    }

    /// A table at s0 holding structured buffers of dwords.
    fn dword_buffers() -> (Vec<RootParam>, Vec<UsedResource>) {
        let root = RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        };
        let used = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        };
        (vec![root], vec![used])
    }

    #[test]
    fn a_spin_wait_on_a_table_buffer_gets_the_spin_limit() {
        // Each lane rereads its flag in a buffer loaded from the table at
        // s0 until it is no longer 0, then copies it to LDS.
        let spin = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE84_0300, // s_mov_b32 s4, s0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xBE82_0380, // s_mov_b32 s2, 0
            0xE030_6000,
            0x8002_0100, // 0x14: buffer_load_dword v1, v0, s[8:11], 0 idxen glc
            0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
            0x8802_026A, // s_or_b32 s2, vcc_lo, s2
            0x8A7E_027E, // s_andn2_b32 exec_lo, exec_lo, s2
            0xBF89_FFFA, // s_cbranch_execnz -> 0x14
            0x887E_027E, // s_or_b32 exec_lo, exec_lo, s2
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let (layout, used) = dword_buffers();
        let l = lower_with(&spin, layout, used).unwrap();
        assert_eq!(l.spin_loops, 1);
    }

    #[test]
    fn a_loop_that_writes_lds_keeps_the_loop_limit() {
        // The same loop with the LDS write inside it: it has an effect each
        // round, so it is no spin wait.
        let writes = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE84_0300, // s_mov_b32 s4, s0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xBE82_0380, // s_mov_b32 s2, 0
            0xE030_6000,
            0x8002_0100, // 0x14: buffer_load_dword v1, v0, s[8:11], 0 idxen glc
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
            0x8802_026A, // s_or_b32 s2, vcc_lo, s2
            0x8A7E_027E, // s_andn2_b32 exec_lo, exec_lo, s2
            0xBF89_FFF8, // s_cbranch_execnz -> 0x14
            0x887E_027E, // s_or_b32 exec_lo, exec_lo, s2
            S_ENDPGM,
        ];
        let (layout, used) = dword_buffers();
        let l = lower_with(&writes, layout, used).unwrap();
        assert_eq!(l.spin_loops, 0);
    }

    /// Root parameter `slot` in user SGPRs from `user_data` on.
    fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: if root == RootType::Descriptor { 2 } else { 1 },
            base_register: 0,
            user_data,
            slot,
        }
    }

    /// Lowers a pixel program with one target: whether it translates.
    fn lowers_ps(words: &[u32]) -> Result<(), crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
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
            code,
        };
        let p = ir::build(&native.code, false, 4).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let target = crate::emit::container::SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        };
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![target],
            prims: vec![],
            mesh_stride: 0,
            options: super::TranslateOptions::default(),
        })
        .map(|_| ())
    }

    /// Lowers a compute program of constants that allocates 512 bytes of LDS
    /// (`COMPUTE_PGM_RSRC2`'s granule field): whether it translates.
    fn lowers_lds(words: &[u32]) -> Result<(), crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15,
                threads: [32, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, 256).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            options: super::TranslateOptions::default(),
        })
        .map(|_| ())
    }

    /// `v_permlanex16_b32` reads a lane of the *other* row of 16 in the
    /// lane's 32-lane half, the pick from nibble i of {src2, src1}. Before
    /// the rule its write was missing and the LDS store consuming it failed
    /// the whole translation.
    #[test]
    fn permlanex16_reads_a_lane_of_the_other_row() {
        let words = [
            // The selector: lane i of a row reads lane i of the other.
            0xBE80_03FF,
            0x7654_3210, // s_mov_b32 s0, 0x76543210
            0xBE81_03FF,
            0xFEDC_BA98, // s_mov_b32 s1, 0xfedcba98
            // v0 is the thread id; lane i reads the lane its nibble names.
            0xD778_0001,
            0x0004_0100, // v_permlanex16_b32 v1, v0, s0, s1
            0x7E04_0280, // v_mov_b32 v2, 0
            0xD834_0000,
            0x0000_0102, // ds_write_b32 v2, v1
            S_ENDPGM,
        ];
        lowers_lds(&words).unwrap();
    }

    #[test]
    fn a_compressed_export_unpacks_two_f16_channels_per_dword() {
        // A compr export: each source dword packs two f16 channels, the
        // low half first, so four channels come from two VGPRs.
        let words = [
            0x7E00_0282, // v_mov_b32 v0, 2
            0x7E02_0282, // v_mov_b32 v1, 2
            0xF800_0C0F,
            0x0000_0100, // exp mrt0, v0, v0, v1, v1 done compr
            S_ENDPGM,
        ];
        lowers_ps(&words).unwrap();
    }

    #[test]
    fn a_scalar_carry_chain_reaches_the_select() {
        // The high dword of a 64-bit add or subtract, then a select on the
        // carry (borrow) the high half leaves in SCC, and a branch on it
        // (only code with an effect needs its operands).
        let add = [
            0xBE80_03C1, // s_mov_b32 s0, -1
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8000_8100, // s_add_u32 s0, s0, 1
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];
        let sub = [
            0xBE80_0380, // s_mov_b32 s0, 0
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8080_8100, // s_sub_u32 s0, s0, 1
            0x8281_8001, // s_subb_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];
        for words in [add, sub] {
            lowers(&words).unwrap();
        }
    }

    /// A wave32 graphics program's host wave may have 64 lanes, whose upper
    /// half is a second console wave: a lane mask read as a number picks the
    /// reader's half of the ballot, so the reader's lane index is needed.
    #[test]
    fn a_wave32_mask_read_takes_its_ballot_half() {
        let words = [
            0x7D8A_0480, // v_cmp_ne_u32 vcc, 0, v2
            0xBE84_0F6A, // s_bcnt1_i32_b32 s4, vcc_lo
            0x7E08_0204, // v_mov_b32 v4, s4
            0xF800_180F,
            0x0404_0404, // exp mrt0, v4, v4, v4, v4 done vm
            S_ENDPGM,
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code,
        };
        let p = ir::build(&native.code, false, 256).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let lowered = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            options: super::TranslateOptions::default(),
        })
        .unwrap();
        // Without the per-half pick, nothing reads the lane index here.
        assert!(
            lowered
                .op_names
                .iter()
                .any(|n| n == "dx.op.waveGetLaneIndex"),
            "the mask read should pick the lane's half of the ballot"
        );
    }

    #[test]
    fn private_memory_is_as_large_as_its_addresses_reach() {
        let bytes = |words: &[u32]| {
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            super::scratch_bytes(&ir::build(&code, false, 256).unwrap())
        };
        let store = [
            0x7E02_0280, // v_mov_b32 v1, 0
            0xDC70_401C, // scratch_store_dword v1, v2, off offset:28
            0x007D_0201,
        ];
        let indexed = [
            0x3606_0087, // v_and_b32 v3, 7, v0
            0x3406_0682, // v_lshlrev_b32 v3, 2, v3
            0xDC30_4000, // scratch_load_dword v4, v3, off
            0x047D_0003,
        ];
        assert_eq!(bytes(&[&store[..], &indexed, &[S_ENDPGM]].concat()), 32);
        // An address the code does not bound (the thread id).
        let free = [0xDC30_4000, 0x047D_0000]; // scratch_load_dword v4, v0, off
        assert_eq!(
            bytes(&[&store[..], &free, &[S_ENDPGM]].concat()),
            super::SCRATCH_CAP
        );
        let far = [
            0x7E02_02FF, // v_mov_b32 v1, 0x100000
            0x0010_0000,
            0xDC70_4000, // scratch_store_dword v1, v2, off
            0x007D_0201,
            S_ENDPGM,
        ];
        assert_eq!(bytes(&far), super::SCRATCH_MAX);
    }

    #[test]
    fn generic_flat_memory_is_refused() {
        // flat_store_dword v[0:1], v2: the address may be private, shared
        // or global memory.
        let e = lowers(&[0xDC70_0000, 0x007D_0200, S_ENDPGM]).unwrap_err();
        assert!(
            matches!(
                e,
                crate::ShaderError::Unsupported {
                    what: "flat memory",
                    ..
                }
            ),
            "{e}"
        );
    }

    #[test]
    fn a_descriptor_selected_by_scc_picks_its_heap_index() {
        // Two buffer descriptors loaded from the table at s0, one picked by
        // a root constant (s1) with s_cselect_b64 per half, and loaded
        // through by index and stored to LDS so the load is needed.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xF408_0302,
            0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
            0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
            0xE030_2000,
            0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        // Both are structured buffers of dwords in the table's range.
        let used = || {
            vec![UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }]
        };
        lower_with(&words, layout(), used()).unwrap();
    }

    /// Tables of 2D textures at s0 and samplers at s1.
    fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let range = |kind, user_data, slot| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
            user_data,
            slot,
        };
        (
            vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
            vec![
                range(ResourceType::Srv, 0, 0),
                range(ResourceType::Sampler, 1, 1),
            ],
        )
    }

    #[test]
    fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
        use crate::resource::tests::{
            WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
        };
        for (words, wave64) in [
            (WATERFALL_FETCHED_WAVE32, false),
            (WATERFALL_FETCHED_WAVE64, true),
            (WATERFALL_PICKED_WAVE32, false),
        ] {
            let (layout, used) = texture_tables();
            let l = lower_wave(words, layout, used, wave64).unwrap();
            // The loop's reads of the first lane became each lane's own
            // dwords: no wave operation is left.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
        }
    }

    #[test]
    fn buffer_formats_by_number() {
        // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
        // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
        assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
        assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
        assert_eq!(
            buffer_format(50),
            Some((&[10, 10, 10, 2][..], Numeric::Unorm))
        );
        assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
        assert_eq!(
            buffer_format(77),
            Some((&[32, 32, 32, 32][..], Numeric::Float))
        );
        // Invalid, a packed float, past the table.
        assert_eq!(buffer_format(0), None);
        assert_eq!(buffer_format(36), None);
        assert_eq!(buffer_format(78), None);
    }

    fn words_to_bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn sop1_mov_imm(sdst: u8, imm: u8) -> u32 {
        (0x17D << 23) | (u32::from(sdst) << 16) | (0x03 << 8) | u32::from(imm)
    }

    fn sop2(op: u32, sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        (0b10 << 30)
            | (op << 23)
            | (u32::from(sdst) << 16)
            | (u32::from(ssrc1) << 8)
            | u32::from(ssrc0)
    }

    fn sopc(op: u32, ssrc0: u8, ssrc1: u8) -> u32 {
        (0x17E << 23) | (op << 16) | (u32::from(ssrc1) << 8) | u32::from(ssrc0)
    }

    fn sopp(op: u32, simm16: i16) -> u32 {
        (0x17F << 23) | (op << 16) | u32::from(simm16 as u16)
    }

    fn s_load_dwordx4(sdst: u8, sbase: u8) -> [u32; 2] {
        [
            (0x3D << 26) | (0x02 << 18) | (u32::from(sdst) << 6) | u32::from(sbase >> 1),
            125u32 << 25,
        ]
    }

    fn vop1(op: u32, vdst: u8, src0: u32) -> u32 {
        (0x3F << 25) | (u32::from(vdst) << 17) | (op << 9) | src0
    }

    fn vop2(op: u32, vdst: u8, src0: u32, vsrc1: u8) -> u32 {
        (op << 25) | (u32::from(vdst) << 17) | (u32::from(vsrc1) << 9) | src0
    }

    fn v_mov_dpp(
        vdst: u8,
        vsrc: u8,
        ctrl: u16,
        row_mask: u8,
        bank_mask: u8,
        bound_ctrl: bool,
    ) -> [u32; 2] {
        let w0 = vop1(0x01, vdst, 0xFA);
        let w1 = (u32::from(row_mask) << 28)
            | (u32::from(bank_mask) << 24)
            | (u32::from(bound_ctrl) << 19)
            | (u32::from(ctrl) << 8)
            | u32::from(vsrc);
        [w0, w1]
    }

    fn ds_write_b32(vaddr: u8, vdata: u8) -> [u32; 2] {
        [
            (0x36 << 26) | (0x0D << 18),
            (u32::from(vdata) << 8) | u32::from(vaddr),
        ]
    }

    fn ds_read_b32(vdst: u8, vaddr: u8) -> [u32; 2] {
        [
            (0x36 << 26) | (0x36 << 18),
            (u32::from(vdst) << 24) | u32::from(vaddr),
        ]
    }

    fn ds_swizzle_b32(vdst: u8, vsrc: u8, offset: u16) -> [u32; 2] {
        [
            (0x36 << 26) | (0x35 << 18) | u32::from(offset),
            (u32::from(vdst) << 24) | u32::from(vsrc),
        ]
    }

    fn buffer_store_dword(vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
        [
            (0x38 << 26) | (0x1C << 18) | (1 << 12),
            (0x80 << 24)
                | (u32::from(srsrc >> 2) << 16)
                | (u32::from(vdata) << 8)
                | u32::from(vaddr),
        ]
    }

    fn emit_synthetic_cs(
        words: &[u32],
        threads: [u32; 3],
        lds_granules: u32,
    ) -> (HostShader, ir::Program, structure::Structured) {
        emit_synthetic_cs_mode(words, threads, lds_granules, super::Wave64Mode::Native)
    }

    fn emit_synthetic_cs_mode(
        words: &[u32],
        threads: [u32; 3],
        lds_granules: u32,
        wave64: super::Wave64Mode,
    ) -> (HostShader, ir::Program, structure::Structured) {
        // user_sgprs = 1 (s0: UAV descriptor table), thread_id_vgprs = 1 (v0), lds = lds_granules * 512B
        let rsrc2 = (1 << 1) | (lds_granules << 15);
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 16,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: Vec::new(),
            io: StageIo::Cs(ComputeIo { rsrc2, threads }),
            code: words_to_bytes(words),
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let empty = super::super::container::build(&[]);
        let c = Container::parse(&empty).unwrap();
        let hs = emit::emit_with(
            &c,
            &native,
            &p,
            &st,
            &io,
            &res,
            super::TranslateOptions { wave64 },
        )
        .unwrap();
        (hs, p, st)
    }

    fn psv_wave_size(dxil: &[u8]) -> Option<u32> {
        let c = Container::parse(dxil).unwrap();
        let psv = c.part(FourCC::PSV0).unwrap().data;
        let min = u32::from_le_bytes(psv[20..24].try_into().unwrap());
        let max = u32::from_le_bytes(psv[24..28].try_into().unwrap());
        (min != 0 && min == max).then_some(min)
    }

    #[cfg(windows)]
    fn validate_and_run_cs(hs: &HostShader, uav_dwords: u32) -> Option<Vec<u32>> {
        use super::super::host::{Dispatch, Host, Root, View};
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut hs.dxil.clone()).unwrap();
        }
        let Ok(host) = Host::new(false) else {
            return None;
        };
        let (root, pso) = match host.compute_pipeline(hs) {
            Ok(p) => p,
            // WARP supports no fixed WaveSize (32 or 64).
            Err(_)
                if psv_wave_size(&hs.dxil).is_some()
                    && std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1") =>
            {
                return None;
            }
            Err(e) => panic!("compute_pipeline failed: {e}"),
        };
        let views = [View::RawUav(uav_dwords)];
        let mut ud = vec![0u32; hs.user_data_dwords() as usize];
        for u in &hs.user_data {
            ud[u.index as usize] = 0;
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        Some(out[0].clone())
    }

    #[test]
    fn wave64_single_wave_lds_adds_barriers_and_drops_wave_size() {
        // Each lane writes `tid + 100` to `lds[tid * 4]`, then without any
        // `s_barrier` reads `lds[(tid ^ 32) * 4]` (crossing the 32-lane half
        // boundary) and stores it to `uav[tid * 4]`.
        let mut w = Vec::new();
        w.push(sop1_mov_imm(1, 0x80)); // s1 = 0
        w.extend(s_load_dwordx4(4, 0)); // s[4:7] = uav0 descriptor
        w.push(vop2(0x1A, 1, 0x82, 0)); // v1 = v0 << 2 (tid * 4)
        w.push(vop2(0x25, 2, 0xFF, 0)); // v2 = v0 + 100
        w.push(100);
        w.extend(ds_write_b32(1, 2)); // lds[tid * 4] = tid + 100
        w.push(vop2(0x1D, 3, 0xA0, 0)); // v3 = v0 ^ 32 (const 32 = 0x80 + 32 = 0xA0)
        w.push(vop2(0x1A, 3, 0x82, 3)); // v3 = (tid ^ 32) * 4
        w.extend(ds_read_b32(4, 3)); // v4 = lds[(tid ^ 32) * 4]
        w.extend(buffer_store_dword(4, 1, 4)); // uav[tid * 4] = v4
        w.push(S_ENDPGM);

        let (hs, p, st) = emit_synthetic_cs_mode(&w, [64, 1, 1], 1, super::Wave64Mode::Wave32);
        let barriers = plan_lds_barriers(&p, &st).expect("uniform LDS barriers");
        assert!(
            !barriers.before.is_empty(),
            "expected RAW barrier before ds_read_b32"
        );
        assert!(barriers.at_end.is_empty());
        assert_eq!(psv_wave_size(&hs.dxil), None);

        #[cfg(windows)]
        if let Some(out) = validate_and_run_cs(&hs, 64) {
            let expected: Vec<u32> = (0..64u32).map(|tid| (tid ^ 32) + 100).collect();
            assert_eq!(out[..64], expected[..]);
        }
    }

    #[test]
    fn wave64_single_wave_lds_uniform_loop_syncs_raw_and_war() {
        // Uniform 4-iteration loop with no `s_barrier`: each iteration writes
        // `acc` to `lds[tid * 4]`, reads `lds[(tid ^ 32) * 4]`, and adds 1.
        // Needs both a RAW barrier before the read and a WAR barrier before
        // the next iteration's write.
        let mut w = Vec::new();
        w.push(sop1_mov_imm(1, 0x80)); // s1 = 0
        w.extend(s_load_dwordx4(4, 0)); // s[4:7] = uav0 descriptor
        w.push(sop1_mov_imm(2, 0x84)); // s2 = 4 (loop counter)
        w.push(vop2(0x1A, 1, 0x82, 0)); // v1 = tid * 4
        w.push(vop2(0x1D, 3, 0xA0, 0)); // v3 = tid ^ 32
        w.push(vop2(0x1A, 3, 0x82, 3)); // v3 = (tid ^ 32) * 4
        w.push(vop2(0x25, 2, 0x80, 0)); // v2 = tid (initial acc)
        let loop_start = w.len();
        w.extend(ds_write_b32(1, 2)); // lds[tid * 4] = acc
        w.extend(ds_read_b32(2, 3)); // acc = lds[(tid ^ 32) * 4]
        w.push(vop2(0x25, 2, 0x81, 2)); // acc += 1
        w.push(sop2(0x01, 2, 2, 0x81)); // s_sub_u32 s2, s2, 1
        w.push(sopc(0x06, 2, 0x80)); // s_cmp_eq_u32 s2, 0
        let branch_idx = w.len();
        let rel = (loop_start as isize - (branch_idx as isize + 1)) as i16;
        w.push(sopp(0x04, rel)); // s_cbranch_scc0 loop_start
        w.extend(buffer_store_dword(2, 1, 4)); // uav[tid * 4] = acc
        w.push(S_ENDPGM);

        let (hs, p, st) = emit_synthetic_cs_mode(&w, [64, 1, 1], 1, super::Wave64Mode::Wave32);
        let barriers = plan_lds_barriers(&p, &st).expect("uniform loop LDS barriers");
        assert_eq!(
            barriers.before.len(),
            2,
            "expected both RAW and WAR barriers: {:?}",
            barriers.before
        );
        assert_eq!(psv_wave_size(&hs.dxil), None);

        #[cfg(windows)]
        if let Some(out) = validate_and_run_cs(&hs, 64) {
            // 4 swaps of (tid <-> tid ^ 32) returns to tid, +4 total.
            let expected: Vec<u32> = (0..64u32).map(|tid| tid + 4).collect();
            assert_eq!(out[..64], expected[..]);
        }
    }

    #[test]
    fn wave64_divergent_lds_hazard_keeps_wave_size_64() {
        // Divergent branch containing both an LDS write and an LDS read
        // without reconverging must refuse barrier planning and keep WaveSize(64).
        let mut w = Vec::new();
        w.push(sop1_mov_imm(1, 0x80));
        w.extend(s_load_dwordx4(4, 0));
        w.push(vop2(0x1A, 1, 0x82, 0));
        // v_cmp_gt_u32 vcc, 32, v0  (32 > v0, i.e., tid < 32)
        w.push((0x3E << 25) | (0xC4 << 17) | 0xA0);
        // s_and_saveexec_b64 s[2:3], vcc (SOP1 op 0x24)
        w.push((0x17D << 23) | (2 << 16) | (0x24 << 8) | 0x6A);
        // s_cbranch_execz +4 dwords (skip ds_write + ds_read)
        w.push(sopp(0x08, 4));
        w.extend(ds_write_b32(1, 0));
        w.extend(ds_read_b32(2, 1));
        // s_or_b64 exec, exec, s[2:3] (SOP2 op 0x11)
        w.push(sop2(0x11, 126, 126, 2));
        w.extend(buffer_store_dword(2, 1, 4));
        w.push(S_ENDPGM);

        let (hs, p, st) = emit_synthetic_cs(&w, [64, 1, 1], 1);
        assert!(plan_lds_barriers(&p, &st).is_none());
        assert_eq!(psv_wave_size(&hs.dxil), Some(64));
    }

    #[test]
    fn wave64_divergent_write_reconverging_before_read_adds_barrier() {
        // Divergent write (`if (tid < 32) lds[tid * 4] = tid + 50`) reconverges
        // before all 64 lanes read `lds[(tid & 31) * 4]`: safe to synchronise
        // at the uniform reconvergence block and drop WaveSize(64).
        let mut w = Vec::new();
        w.push(sop1_mov_imm(1, 0x80));
        w.extend(s_load_dwordx4(4, 0));
        w.push(vop2(0x1A, 1, 0x82, 0)); // v1 = tid * 4
        w.push(vop2(0x25, 2, 0xFF, 0)); // v2 = tid + 50
        w.push(50);
        // v_cmp_gt_u32 vcc, 32, v0 (tid < 32)
        w.push((0x3E << 25) | (0xC4 << 17) | 0xA0);
        // s_and_saveexec_b64 s[2:3], vcc
        w.push((0x17D << 23) | (2 << 16) | (0x24 << 8) | 0x6A);
        // s_cbranch_execz +2 dwords (skip ds_write)
        w.push(sopp(0x08, 2));
        w.extend(ds_write_b32(1, 2));
        // s_or_b64 exec, exec, s[2:3] (reconverge)
        w.push(sop2(0x11, 126, 126, 2));
        // v3 = (tid & 31) * 4  (v_and_b32 = VOP2 op 0x1B, const 31 = 0x9F)
        w.push(vop2(0x1B, 3, 0x9F, 0));
        w.push(vop2(0x1A, 3, 0x82, 3));
        w.extend(ds_read_b32(4, 3));
        w.extend(buffer_store_dword(4, 1, 4));
        w.push(S_ENDPGM);

        let (hs, p, st) = emit_synthetic_cs_mode(&w, [64, 1, 1], 1, super::Wave64Mode::Wave32);
        let barriers = plan_lds_barriers(&p, &st).expect("reconverged LDS barrier");
        assert!(!barriers.before.is_empty());
        assert_eq!(psv_wave_size(&hs.dxil), None);

        #[cfg(windows)]
        if let Some(out) = validate_and_run_cs(&hs, 64) {
            let expected: Vec<u32> = (0..64u32).map(|tid| (tid & 31) + 50).collect();
            assert_eq!(out[..64], expected[..]);
        }
    }

    #[test]
    fn wave64_readfirstlane_of_tid_branch_keeps_wave_size_64() {
        // A scalar branch driven by `v_readfirstlane_b32(v0)` differs between
        // lanes 0..31 and 32..63 at Wave32, so it must be treated as divergent.
        let mut w = Vec::new();
        w.push(sop1_mov_imm(1, 0x80));
        w.extend(s_load_dwordx4(4, 0));
        w.push(vop2(0x1A, 1, 0x82, 0));
        // v_readfirstlane_b32 s2, v0 (VOP1 op 0x02, vdst=sdst=2, src0=256+0)
        w.push(vop1(0x02, 2, 256));
        w.push(sopc(0x06, 2, 0x80)); // s_cmp_eq_u32 s2, 0
        w.push(sopp(0x04, 4)); // s_cbranch_scc0 +4 dwords (skip ds_write + ds_read)
        w.extend(ds_write_b32(1, 0));
        w.extend(ds_read_b32(2, 1));
        w.extend(buffer_store_dword(2, 1, 4));
        w.push(S_ENDPGM);

        let (hs, p, st) = emit_synthetic_cs(&w, [64, 1, 1], 1);
        assert!(plan_lds_barriers(&p, &st).is_none());
        assert_eq!(psv_wave_size(&hs.dxil), Some(64));
    }

    #[test]
    fn wave64_local_swizzle_and_dpp_row_mask_uses_wave_size_32() {
        // Uses `ds_swizzle_b32` (XOR 16, within each 32-lane half-wave) and
        // `v_mov_b32_dpp` with `row_shr:1`, `row_mask:0xC` (only console rows
        // 2 and 3, i.e., lanes 32..63) and `bound_ctrl:1`.
        let mut w = Vec::new();
        w.push(sop1_mov_imm(1, 0x80)); // s1 = 0
        w.extend(s_load_dwordx4(4, 0)); // s[4:7] = uav0 descriptor
        w.push(vop2(0x1A, 1, 0x82, 0)); // v1 = tid * 4
        // ds_swizzle_b32 v2, v0 offset:swizzle(BITMASK_PERM, "01111" and=0x1f or=0 xor=16)
        // offset = (xor << 10) | (or << 5) | and = (16 << 10) | 0x1F = 0x401F
        w.extend(ds_swizzle_b32(2, 0, 0x401F));
        // v3 = 0x9999 (sentinel preserved on masked-off rows 0 and 1)
        w.push(vop1(0x01, 3, 0xFF));
        w.push(0x9999);
        // v_mov_b32_dpp v3, v0 row_shr:1 row_mask:0xC bank_mask:0xF bound_ctrl:1
        w.extend(v_mov_dpp(3, 0, 0x111, 0xC, 0xF, true));
        // Store swizzle to uav[tid * 4] and DPP to uav[(64 + tid) * 4]
        w.extend(buffer_store_dword(2, 1, 4));
        w.push(vop2(0x25, 4, 0xFF, 1)); // v4 = tid * 4 + 256
        w.push(256);
        w.extend(buffer_store_dword(3, 4, 4));
        w.push(S_ENDPGM);

        let (hs, _, _) = emit_synthetic_cs_mode(&w, [64, 1, 1], 0, super::Wave64Mode::Wave32);
        assert_eq!(psv_wave_size(&hs.dxil), Some(32));

        #[cfg(windows)]
        if let Some(out) = validate_and_run_cs(&hs, 128) {
            for tid in 0..64u32 {
                assert_eq!(
                    out[tid as usize],
                    tid ^ 16,
                    "swizzle mismatch at lane {tid}"
                );
                let expected_dpp = if tid < 32 {
                    0x9999
                } else if tid % 16 == 0 {
                    0
                } else {
                    tid - 1
                };
                assert_eq!(
                    out[(64 + tid) as usize],
                    expected_dpp,
                    "DPP row_mask mismatch at lane {tid}"
                );
            }
        }
    }

    #[test]
    fn packed_half_inputs_pair_unlisted_rows() {
        use super::super::container::SigElement;
        use super::half_inputs;
        use crate::native::{PsInput, StageIo, Usage};

        let sig = |row: u32, half: bool| SigElement {
            name: "TEXCOORD".into(),
            index: row,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half,
        };
        let ps_in = |row: u8, attr: u8, pack2: bool, row2: u8| PsInput {
            usage: Usage::Generic,
            row,
            mask: 3,
            attr,
            flat: false,
            fp16: true,
            pack2,
            row2,
        };
        // Row 0 has row2 == row (0) and pairs with unlisted row 1;
        // row 2 explicitly names row2 == 4 (not stolen by row 0);
        // row 5 has row2 == row (5) and no remaining unlisted row after it.
        let inputs = vec![
            sig(0, true),
            sig(1, true),
            sig(2, true),
            sig(3, false),
            sig(4, true),
            sig(5, true),
        ];
        let io = StageIo::Ps {
            inputs: vec![
                ps_in(0, 0, true, 0),
                ps_in(2, 1, true, 4),
                ps_in(5, 3, true, 5),
            ],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        };
        let map = half_inputs(&io, &inputs);
        assert_eq!(map.get(&0), Some(&(true, Some(1))));
        assert_eq!(map.get(&2), Some(&(true, Some(4))));
        assert_eq!(map.get(&5), Some(&(true, None)));

        // Duplicate entries for the same row keep the first entry and do not
        // consume an extra free row; an earlier unpaired row (0) must not steal
        // a free row (3) that comes after a later fp16 row (2).
        let inputs2 = vec![sig(0, true), sig(2, true), sig(3, true)];
        let io2 = StageIo::Ps {
            inputs: vec![
                ps_in(0, 0, true, 0),
                ps_in(0, 9, false, 0),
                ps_in(2, 1, true, 2),
            ],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        };
        let map2 = half_inputs(&io2, &inputs2);
        assert_eq!(map2.get(&0), Some(&(true, None)));
        assert_eq!(map2.get(&2), Some(&(true, Some(3))));
    }
}

#[cfg(test)]
mod select_tests {
    use super::super::emit;
    use crate::native::{ComputeIo, RootParam};
    use crate::{Container, FourCC, HwStage, NativeProgram, ResourceType, RootType, UsedResource};

    #[test]
    fn selects_from_independently_written_mask_words() {
        let words = [
            0xbe81_0380u32, // s1 = 0 (table address high)
            0xf408_0200,
            0xfa00_0000, // load UAV into s[8:11]
            0xbe84_03ff,
            0x8000_0000, // s4 = lane 31
            0xbe85_0381, // s5 = lane 32 in wave64
            0x7e02_028b,
            0x7e04_029d, // v1 = 11, v2 = 29
            0xd501_0003,
            0x0012_0501, // v_cndmask_b32 v3, v1, v2, s[4:5]
            0x3402_0082, // v1 = 4 * v0
            0xe070_1000,
            0x8002_0301, // store v3 to UAV at v1
            0xbf81_0000,
        ];
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        for (wave64, uses_select) in [(false, true), (true, true), (true, false)] {
            let mut words = words;
            if !uses_select {
                words[12] = 0x8002_0101; // store v1 instead of the select result
            }
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64,
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
                io: crate::native::StageIo::Cs(ComputeIo {
                    rsrc2: 2,
                    threads: [if wave64 { 64 } else { 32 }, 1, 1],
                }),
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let p = crate::ir::build(&native.code, wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
            let shader = emit(&c, &native, &p, &st, &io, &res).unwrap();
            let emitted = Container::parse(&shader.dxil).unwrap();
            let features = emitted.part(FourCC::SFI0).unwrap();
            let wave_ops = u64::from_le_bytes(features.data.try_into().unwrap()) & 0x4000 != 0;
            assert_eq!(wave_ops, uses_select);
        }
    }
}

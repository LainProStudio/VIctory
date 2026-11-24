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
    lowers_with_threads(words, 1 << 15, [32, 1, 1])
}

/// [`lowers`], with `rsrc2` saying how much LDS the program allocates.
fn lowers_with(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
    lowers_with_threads(words, rsrc2, [1, 1, 1])
}

fn lowers_with_threads(
    words: &[u32],
    rsrc2: u32,
    threads: [u32; 3],
) -> Result<(), crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo { rsrc2, threads }),
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
fn a_wave_prefix_of_min_lowers() {
    // A whole-wave section that leaves the inclusive prefix of a min:
    // each lane the minimum of the lanes up to and including it. The
    // scan runs with `row_shr` steps inside each row, and the first
    // row's prefix is read out and folded under the second. The store
    // after the section reads the prefix, so the value cannot be left
    // missing: a wave prefix of a min has to lower.
    let words = [
        0x7E00_0280, // v_mov_b32 v0, 0
        // 32-bit scalar forms: the wave32 pass follows their masks.
        0xBE84_3DC1, // s_or_saveexec_b32 s4, -1
        0x876A_C104, // s_and_b32 s106, s4, -1
        0x0202_00C1, // v_cndmask_b32 v1, -1, v0, vcc
        0x2602_02FA, // v_min_u32 v1, v1 row_shr:1, v1
        0xFF01_1101,
        0x2602_02FA, // v_min_u32 v1, v1 row_shr:2, v1
        0xFF01_1201,
        0x2602_02FA, // v_min_u32 v1, v1 row_shr:3, v1
        0xFF01_1301,
        0x2602_02FA, // v_min_u32 v1, v1 row_shr:4, v1
        0xFF01_1401,
        0x2602_02FA, // v_min_u32 v1, v1 row_shr:5, v1
        0xFF01_1501,
        0x2602_02FA, // v_min_u32 v1, v1 row_shr:6, v1
        0xFF01_1601,
        0x2602_02FA, // v_min_u32 v1, v1 row_shr:7, v1
        0xFF01_1701,
        0x2602_02FA, // v_min_u32 v1, v1 row_shr:8, v1
        0xFF01_1801,
        0x2602_02FA, // v_min_u32 v1, v1 row_shr:9, v1
        0xFF01_1901,
        0x2602_02FA, // v_min_u32 v1, v1 row_shr:10, v1
        0xFF01_1A01,
        0x2602_02FA, // v_min_u32 v1, v1 row_shr:11, v1
        0xFF01_1B01,
        0x2602_02FA, // v_min_u32 v1, v1 row_shr:12, v1
        0xFF01_1C01,
        0x2602_02FA, // v_min_u32 v1, v1 row_shr:13, v1
        0xFF01_1D01,
        0x2602_02FA, // v_min_u32 v1, v1 row_shr:14, v1
        0xFF01_1E01,
        0x2602_02FA, // v_min_u32 v1, v1 row_shr:15, v1
        0xFF01_1F01,
        0xD760_0006, // v_readlane_b32 s6, v1, 15
        0x0001_1F01,
        0x7E04_0206, // v_mov_b32 v2, s6
        0x2602_02FA, // v_min_u32 v1, v2 quad_perm:[0,1,2,3] row_mask:0x2, v1
        0x2F00_E402,
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        0xD834_0000, // ds_write_b32 v0, v1
        0x0000_0100,
        S_ENDPGM,
    ];
    lowers_with(&words, 1 << 15).unwrap();
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

#[cfg(windows)]
fn validate(dxil: &[u8]) {
    use super::super::validator::Validator;
    let Ok(mut v) = Validator::load(Validator::default_path().as_deref()) else {
        eprintln!("no dxil.dll; skipped");
        return;
    };
    let mut copy = dxil.to_vec();
    v.validate(&mut copy).unwrap();
    assert_eq!(copy[4..20], dxil[4..20], "container hash");
}

#[cfg(not(windows))]
fn validate(_: &[u8]) {}

#[test]
fn ds_bpermute_lowers_without_lds_in_wave32_and_wave64() {
    // ds_bpermute_b32 v2, v0, v1 offset:12 (DS opcode 0x0b3; no LDS allocated),
    // consumed by a branch so lowering must actually emit the permute.
    let bpermute = |vdst: u8, vaddr: u8, vdata: u8, off: u16| -> [u32; 2] {
        [
            0xD800_0000 | (0x0B3 << 18) | u32::from(off),
            u32::from(vdst) << 24 | u32::from(vdata) << 8 | u32::from(vaddr),
        ]
    };
    let mut words = vec![
        0x7E02_0285, // v_mov_b32 v1, 5
    ];
    words.extend_from_slice(&bpermute(2, 0, 1, 12));
    words.extend_from_slice(&[
        0x7E00_0502, // v_readfirstlane_b32 s0, v2
        0xBF07_8000, // s_cmp_lg_u32 s0, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ]);
    let code = words_to_bytes(&words);
    for (wave64, mode) in [
        (false, super::Wave64Mode::Native),
        (true, super::Wave64Mode::Native),
        (true, super::Wave64Mode::Wave32),
        (true, super::Wave64Mode::Emulate),
    ] {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [if wave64 { 64 } else { 32 }, 1, 1],
            }),
            code: code.clone(),
        };
        let p = ir::build(&native.code, wave64, native.vgprs).unwrap();
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
            options: super::TranslateOptions { wave64: mode },
        })
        .unwrap();
        validate(&lowered.dxil);
    }
}

#[test]
fn ds_bpermute_refuses_gds() {
    // ds_bpermute_b32 with gds=1 (bit 17), consumed by a branch, must fail with GDS.
    let words = [
        0x7E02_0285,                           // v_mov_b32 v1, 5
        0xD800_0000 | (0x0B3 << 18) | 1 << 17, // ds_bpermute_b32 gds
        2 << 24 | 1 << 8,
        0x7E00_0502, // v_readfirstlane_b32 s0, v2
        0xBF07_8000, // s_cmp_lg_u32 s0, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];
    assert!(matches!(
        lowers(&words),
        Err(crate::ShaderError::Unsupported { what: "GDS", .. })
    ));
}

#[test]
fn group_info_and_waveid_in_workgroup_lower_and_validate() {
    // s0 is the thread-group info SGPR (RSRC2 bit 10, TG_SIZE_EN);
    // extract bits 11:6 (wave id), compare with s_get_waveid_in_workgroup s1
    // in a branch so both values are lowered.
    let words = [
        0xF4A8_0040,
        0xFA00_0000, // s_get_waveid_in_workgroup s1
        0x9380_FF00, // s_bfe_u32 s0, s0, lit(6 << 16 | 6)
        6 << 16 | 6,
        0xBF07_0100, // s_cmp_lg_u32 s0, s1
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE82_0380, // s_mov_b32 s2, 0
        S_ENDPGM,
    ];
    for (wave64, mode) in [
        (false, super::Wave64Mode::Native),
        (true, super::Wave64Mode::Native),
        (true, super::Wave64Mode::Wave32),
        (true, super::Wave64Mode::Emulate),
    ] {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 10,
                threads: [64, 2, 1],
            }),
            code: words_to_bytes(&words),
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let l = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            options: super::TranslateOptions { wave64: mode },
        })
        .unwrap();
        validate(&l.dxil);
    }

    // Outside compute or mesh stages there is no thread group: using the
    // wave id in a branch fails with the reason.
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
        code: words_to_bytes(&[
            words[0],
            words[1],
            0xBF07_8001, // s_cmp_lg_u32 s1, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE82_0380, // s_mov_b32 s2, 0
            S_ENDPGM,
        ]),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let res = resource::track(&p, &native.layout, native.stage).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let st = structure::structure(&p).unwrap();
    let err = lower(&Parts {
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
    .err()
    .unwrap();
    assert!(
        matches!(
            err,
            crate::ShaderError::Unsupported {
                what: "s_get_waveid_in_workgroup outside compute",
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn sample_coverage_input_lowers_and_validates() {
    use super::super::container::SigElement;
    // SPI_PS_INPUT_ADDR bit 14 loads the sample coverage mask into v0.
    let words = [
        0x7E00_0D00, // v_cvt_f32_u32 v0, v0
        0xF800_180F,
        0x0000_0000, // exp mrt0 v0, v0, v0, v0 done vm
        S_ENDPGM,
    ];
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 4,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![],
            input_addr: 1 << 14,
            db_shader_control: 0,
            patches: vec![],
        },
        code: words_to_bytes(&words),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let res = resource::track(&p, &native.layout, native.stage).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let st = structure::structure(&p).unwrap();
    let target = SigElement {
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
    let l = lower(&Parts {
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
    .unwrap();
    validate(&l.dxil);
}

#[test]
fn cube_sample_grad_and_cl_lower_and_validate() {
    use super::super::container::SigElement;
    // MIMG image_sample_d (0x22) and image_sample_d_cl (0x23) on dim=3 (cube).
    for op in [0x22u32, 0x23u32] {
        let words = [
            0xBE81_0380u32, // s_mov_b32 s1, 0
            0xF40C_0100,    // s_load_dwordx8 s[4:11], s[0:1], 0
            0xFA00_0000,
            0xBE8C_0380, // s_mov_b32 s12, 0
            0xBE8D_0380, // s_mov_b32 s13, 0
            0xBE8E_0380, // s_mov_b32 s14, 0
            0xBE8F_0380, // s_mov_b32 s15, 0
            0xF000_0000 | (op << 18) | (1 << 8) | (1 << 3),
            (3 << 21) | (1 << 16), // dim=3 (cube), srsrc=s[4:11], ssamp=s[12:15], vaddr=v0, vdata=v0
            0xF800_080F,
            0x0000_0000, // exp mrt0 v0, v0, v0, v0 done
            S_ENDPGM,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 12,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Srv,
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
            code: words_to_bytes(&words),
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let target = SigElement {
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
        let l = lower(&Parts {
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
        .unwrap();
        validate(&l.dxil);
    }
}

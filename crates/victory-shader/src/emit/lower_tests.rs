use super::{
    Lowered, Numeric, Parts, buffer_format, compute_uniformity, half_inputs, lower,
    packed_input_half, plan_lds_barriers,
};
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

fn plan_synthetic_barriers(
    words: &[u32],
    threads: [u32; 3],
    lds_granules: u32,
) -> Option<super::LdsBarriers> {
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
    let (uniform, _) = compute_uniformity(&p, &st, &res, &io.inputs);
    plan_lds_barriers(&p, &st, &uniform)
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

    let (hs, _, _) = emit_synthetic_cs_mode(&w, [64, 1, 1], 1, super::Wave64Mode::Wave32);
    let barriers = plan_synthetic_barriers(&w, [64, 1, 1], 1).expect("uniform LDS barriers");
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

    let (hs, _, _) = emit_synthetic_cs_mode(&w, [64, 1, 1], 1, super::Wave64Mode::Wave32);
    let barriers = plan_synthetic_barriers(&w, [64, 1, 1], 1).expect("uniform loop LDS barriers");
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

    let (hs, _, _) = emit_synthetic_cs(&w, [64, 1, 1], 1);
    assert!(plan_synthetic_barriers(&w, [64, 1, 1], 1).is_none());
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

    let (hs, _, _) = emit_synthetic_cs_mode(&w, [64, 1, 1], 1, super::Wave64Mode::Wave32);
    let barriers = plan_synthetic_barriers(&w, [64, 1, 1], 1).expect("reconverged LDS barrier");
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

    let (hs, _, _) = emit_synthetic_cs(&w, [64, 1, 1], 1);
    assert!(plan_synthetic_barriers(&w, [64, 1, 1], 1).is_none());
    assert_eq!(psv_wave_size(&hs.dxil), Some(64));
}

#[test]
fn wave64_128_threads_intra_wave_lds_with_s_barrier_plans_barriers() {
    // 128 threads with an s_barrier isolating Phase 0/1, then intra-wave
    // WAR + RAW LDS hazards in Phase 1 with no s_barrier: planner must reset
    // hazard state across s_barrier and insert 2 intra-wave barriers in Phase 1.
    let mut w = Vec::new();
    w.push(sop1_mov_imm(1, 0x80));
    w.extend(s_load_dwordx4(4, 0));
    w.push(vop2(0x1A, 1, 0x82, 0)); // v_lshlrev_b32 v1, 2, v0
    w.extend(ds_write_b32(1, 0)); // Phase 0 write
    w.push(0xBF8A_0000); // s_barrier
    w.extend(ds_read_b32(2, 1)); // Phase 1 read (safe after s_barrier)
    w.extend(ds_write_b32(1, 2)); // Phase 1 write (WAR -> barrier 1)
    w.extend(ds_read_b32(3, 1)); // Phase 1 read (RAW -> barrier 2)
    w.extend(buffer_store_dword(3, 1, 4));
    w.push(S_ENDPGM);

    let (hs, _, _) = emit_synthetic_cs_mode(&w, [128, 1, 1], 2, super::Wave64Mode::Wave32);
    let barriers =
        plan_synthetic_barriers(&w, [128, 1, 1], 2).expect("planned intra-wave barriers");
    assert_eq!(barriers.before.len(), 2, "{:?}", barriers.before);
    assert!(barriers.at_end.is_empty());
    assert_eq!(psv_wave_size(&hs.dxil), None);
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
    assert_eq!(map.get(&0), Some(&(true, Some(1), false)));
    assert_eq!(map.get(&2), Some(&(true, Some(4), true)));
    assert_eq!(map.get(&5), Some(&(true, None, false)));

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
    assert_eq!(map2.get(&0), Some(&(true, None, false)));
    assert_eq!(map2.get(&2), Some(&(true, Some(3), false)));
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

#[test]
fn a_missing_value_stays_missing_through_two_live_joins() {
    let mut words = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
        0xBF85_0001, // skip the first overwrite
        0xBE86_0380, // s_mov_b32 s6, 0
        0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
        0xBF85_0001, // skip the second overwrite
        0xBE86_0381, // s_mov_b32 s6, 1
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // use the result to branch
        0xBE83_0380,
        S_ENDPGM,
    ];
    // Compare to a direct read so future timestamp support works in both cases.
    let refusal = |code: &[u32]| match lowers_cs(code, false, 0) {
        Ok(_) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    let direct = [&words[..2], &words[8..]].concat();
    assert_eq!(refusal(&words), refusal(&direct));
    words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
    lowers_cs(&words, false, 0).expect("dead timestamp");
}

#[test]
fn packed_pixel_inputs_find_their_second_row() {
    use super::super::container::SigElement;
    use crate::native::{PsInput, Usage};
    let entry = |row, row2, pack2| PsInput {
        usage: Usage::Generic,
        row,
        mask: 0xF,
        attr: row,
        flat: false,
        fp16: true,
        pack2,
        row2,
    };
    let element = |row| SigElement {
        name: "TEXCOORD".into(),
        index: row,
        kind: 0,
        comp: 9,
        interp: 2,
        row: Some(row),
        col: 0,
        cols: 4,
        usage: 0,
        half: true,
    };
    // Rows 0 and 3 pack and leave their second row out; 5 names its
    // own (6). Unnamed 16-bit rows: 1, 4, 6, 7.
    let io = StageIo::Ps {
        inputs: vec![entry(0, 0, true), entry(3, 3, true), entry(5, 6, true)],
        input_addr: 0,
        db_shader_control: 0,
        patches: vec![],
    };
    let sig: Vec<_> = [0, 1, 3, 4, 5, 6, 7].map(element).into();
    let p = half_inputs(&io, &sig);
    assert_eq!(p.get(&0), Some(&(true, Some(1), false)));
    assert_eq!(p.get(&3), Some(&(true, Some(4), false)));
    assert_eq!(p.get(&5), Some(&(true, Some(6), true)));
}

#[test]
fn explicit_packed_input_tail_is_padding() {
    use super::super::container::SigElement;
    let element = |row, cols| SigElement {
        name: "TEXCOORD".into(),
        index: row,
        kind: 0,
        comp: 9,
        interp: 2,
        row: Some(row),
        col: 0,
        cols,
        usage: 0,
        half: true,
    };
    // Four halves in row 6 followed by only two in its explicit partner
    // row 7. The next two reads address the packing's zero padding.
    let sig = vec![element(6, 4), element(7, 2)];
    assert_eq!(
        packed_input_half(&sig, 6, Some(7), 5, true),
        Ok(Some((7, 1)))
    );
    assert_eq!(packed_input_half(&sig, 6, Some(7), 6, true), Ok(None));
    assert_eq!(packed_input_half(&sig, 6, Some(7), 7, true), Ok(None));
    // With an inferred partner the same gap is not assumed to be zero.
    assert!(packed_input_half(&sig, 6, Some(7), 6, false).is_err());
    assert!(packed_input_half(&sig, 6, None, 4, false).is_err());
}

/// Lowers `words` as a `stage` program (`wave64`: the console's wave
/// size) with no resources or inputs.
fn lower_program(
    words: &[u32],
    stage: HwStage,
    wave64: bool,
    io: StageIo,
) -> Result<Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage,
        wave64,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io,
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

fn pixel_io() -> StageIo {
    StageIo::Ps {
        inputs: vec![],
        input_addr: 0,
        db_shader_control: 0,
        patches: vec![],
    }
}

fn compute_io() -> StageIo {
    StageIo::Cs(ComputeIo {
        rsrc2: 0,
        threads: [1, 1, 1],
    })
}

/// The index of the instruction `pat` names in `lines`.
fn at(lines: &[String], pat: &str) -> usize {
    lines
        .iter()
        .position(|l| l == pat)
        .unwrap_or_else(|| panic!("no `{pat}` in {lines:?}"))
}

/// A wave32 graphics program's host wave may have 64 lanes, whose upper
/// half is a second console wave: a lane mask read as a number picks the
/// reader's half of the ballot, and `v_mbcnt_lo` reports its rank as a
/// host lane number.
#[test]
fn a_wave32_mask_read_takes_its_ballot_half() {
    let words = [
        0x7D8A_0480, // v_cmp_ne_u32 vcc, 0, v2
        0xBE84_0F6A, // s_bcnt1_i32_b32 s4, vcc_lo
        0xBF07_806A, // s_cmp_lg_u32 vcc_lo, 0
        0x7E08_0204, // v_mov_b32 v4, s4
        0xD765_0005,
        0x0001_006A, // v_mbcnt_lo_u32_b32 v5, vcc_lo, 0
        0xD766_0006,
        0x0002_0A6A, // v_mbcnt_hi_u32_b32 v6, vcc_lo, v5
        0xD765_0007,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v7, -1, 0
        0xBF85_0001, // s_cbranch_scc1 -> the export
        S_ENDPGM,    // no lane had its bit: done
        0xF800_180F,
        0x0706_0504, // exp mrt0, v4, v5, v6, v7 done vm
        S_ENDPGM,
    ];
    let lowered = lower_program(&words, HwStage::Ps, false, pixel_io()).unwrap();
    let f = |name| lowered.op_names.iter().position(|n| n == name);
    let (lane, ballot, bits) = (
        f("dx.op.waveGetLaneIndex").expect("no lane index for the pick"),
        f("dx.op.waveActiveBallot").expect("no ballot for the mask read"),
        f("dx.op.unaryBits.i32").expect("no bit count"),
    );
    let l = &lowered.insts;
    let lanes: Vec<usize> = l
        .iter()
        .enumerate()
        .filter(|(_, i)| **i == format!("call f{lane} [k111]"))
        .map(|(i, _)| i)
        .collect();
    assert!(!lanes.is_empty(), "no lane index for the pick");
    let i_ballot = l
        .iter()
        .position(|i| i.starts_with(&format!("call f{ballot} [k116, ")))
        .expect("no ballot");
    let (i_x, i_y) = (
        at(l, &format!("extract i{i_ballot} 0")),
        at(l, &format!("extract i{i_ballot} 1")),
    );
    // The pick: upper host lanes read the ballot's second dword.
    let i_cmp = l
        .iter()
        .position(|i| {
            i.starts_with("cmp:") && lanes.iter().any(|n| i.ends_with(&format!("i{n} k32")))
        })
        .expect("no `lane >= 32` compare");
    at(l, &format!("select i{i_cmp} i{i_y} i{i_x}"));
    // v_mbcnt_lo counts below the lane in its own half; a wave32 mask's
    // high half is empty, so v_mbcnt_hi's count is of no bits.
    assert!(
        lanes
            .iter()
            .any(|n| l.contains(&format!("bin:And i{n} k31"))),
        "no `lane & 31` in {l:?}"
    );
    at(l, &format!("call f{bits} [k31, k0]"));
    // The count is a lane number, so a second wave's rank takes its
    // 32-lane base: `v_mbcnt_lo(-1)` is the host lane index.
    let bases: Vec<usize> = l
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            s.starts_with("bin:And ")
                && lanes.iter().any(|n| **s == format!("bin:And i{n} k32"))
        })
        .map(|(i, _)| i)
        .collect();
    assert!(!bases.is_empty(), "no `lane & 32` wave base in {l:?}");
    for base in &bases {
        assert!(
            l.iter().any(|s| {
                s.starts_with("bin:Add ")
                    && s["bin:Add ".len()..]
                        .split(' ')
                        .any(|a| a == format!("i{base}"))
            }),
            "the wave base `i{base}` is never added in {l:?}"
        );
    }
}

/// On a 64-lane host wave a wave32 program's lane-indexed ops use the
/// lane's own console wave: a constant lane operand takes the wave's
/// 32-lane base, and the first lane comes from its half's ballot.
#[test]
fn a_wave32_lane_op_uses_its_own_console_wave() {
    let words = [
        0xD760_0004,
        0x0001_0B02, // v_readlane_b32 v4, v2, 5
        0xD761_0005,
        0x0001_0A87, // v_writelane_b32 v5, 7, 5
        0x7E0C_0502, // v_readfirstlane_b32 v6, v2
        0x360E_0B04, // v_and_b32 v7, v4, v5
        0x360E_0D07, // v_and_b32 v7, v7, v6
        0x7D8A_0E80, // v_cmp_ne_u32 vcc, 0, v7
        0xBF07_806A, // s_cmp_lg_u32 vcc_lo, 0
        0xBF85_0001, // s_cbranch_scc1 -> the export
        S_ENDPGM,
        0xF800_180F,
        0x0707_0707, // exp mrt0, v7, v7, v7, v7 done vm
        S_ENDPGM,
    ];
    let lowered = lower_program(&words, HwStage::Ps, false, pixel_io()).unwrap();
    let l = &lowered.insts;
    let lane_calls: Vec<usize> = l
        .iter()
        .enumerate()
        .filter(|(_, i)| i.starts_with("call f") && i.ends_with("[k111]"))
        .map(|(i, _)| i)
        .collect();
    assert!(!lane_calls.is_empty(), "no lane index in {l:?}");
    let bases: Vec<usize> = lane_calls
        .iter()
        .flat_map(|n| {
            l.iter()
                .enumerate()
                .filter(|(_, i)| **i == format!("bin:And i{n} k32"))
                .map(|(i, _)| i)
                .collect::<Vec<_>>()
        })
        .collect();
    let used = |v: usize| -> bool {
        l.iter()
            .any(|i| i.ends_with(&format!("i{v}]")) || i.ends_with(&format!(" i{v}")))
    };
    // Console lane 5 of the lane's own wave is host lane 5 | (lane & 32),
    // for both v_readlane (readLaneAt) and v_writelane (lane equality).
    let ors: Vec<usize> = bases
        .iter()
        .filter_map(|b| l.iter().position(|i| i == &format!("bin:Or k5 i{b}")))
        .collect();
    assert_eq!(ors, Vec::from([2, 6]), "no `5 | base` host lanes in {l:?}");
    for or in &ors {
        assert!(used(*or), "the host lane `i{or}` is never used in {l:?}");
    }
    assert!(
        l.iter()
            .any(|i| i.starts_with("call f") && i.contains("k117, ")),
        "no waveReadLaneAt in {l:?}"
    );
    // v_readfirstlane takes the low set bit of the lane's own half of
    // the exec ballot, renumbered to a host lane.
    let ballot = l
        .iter()
        .position(|i| i.starts_with("call f") && i.contains("[k116, "))
        .expect("no exec ballot");
    let (x, y) = (
        at(l, &format!("extract i{ballot} 0")),
        at(l, &format!("extract i{ballot} 1")),
    );
    let own = l
        .iter()
        .position(|i| i.starts_with("select ") && i.ends_with(&format!("i{y} i{x}")))
        .expect("no own-half pick");
    let first = l
        .iter()
        .position(|i| i.starts_with("call f") && i.contains(&format!("k32, i{own}")))
        .expect("no first bit of the own half");
    let host_first = bases
        .iter()
        .find_map(|b| l.iter().position(|i| i == &format!("bin:Or i{first} i{b}")))
        .expect("the first lane is not renumbered");
    assert!(used(host_first), "no read at the first host lane in {l:?}");

    // On a wave64 program console lanes are host lanes one to one: the
    // same reads need no renumbering at all.
    let l64 = lower_program(&words, HwStage::Ps, true, pixel_io()).unwrap();
    let l = &l64.insts;
    assert!(
        !l.iter().any(|i| i.starts_with("bin:Or ")),
        "a wave64 lane operand is renumbered in {l:?}"
    );
    assert!(
        l.iter()
            .any(|i| i.starts_with("call f") && i.contains("k117, ") && i.contains("k5]")),
        "no direct waveReadLaneAt of lane 5 in {l:?}"
    );
    assert!(
        l.iter()
            .any(|i| i.starts_with("call f") && i.contains("k118, ")),
        "a wave64 first lane is not waveReadLaneFirst in {l:?}"
    );
}

/// The ballot-half pick is for wave32 graphics programs only: a wave64
/// program's mask dwords are its own, and the console path (compute) is
/// the whole host wave already, so neither reads the lane index for it.
#[test]
fn only_host_wave32_reads_take_a_ballot_half() {
    let words = [
        0x7D8A_0480, // v_cmp_ne_u32 vcc, 0, v2
        0xBE84_0F6A, // s_bcnt1_i32_b32 s4, vcc_lo
        0x7E08_0204, // v_mov_b32 v4, s4
        0xF800_180F,
        0x0404_0404, // exp mrt0, v4, v4, v4, v4 done vm
        S_ENDPGM,
    ];
    let l64 = lower_program(&words, HwStage::Ps, true, pixel_io()).unwrap();
    assert!(!l64.op_names.iter().any(|n| n == "dx.op.waveGetLaneIndex"));
    let words = [
        0x7D8A_0480, // v_cmp_ne_u32 vcc, 0, v0
        0xBE84_0F6A, // s_bcnt1_i32_b32 s4, vcc_lo
        0xBF07_8004, // s_cmp_lg_u32 s4, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE85_0380, // s_mov_b32 s5, 0
        S_ENDPGM,
    ];
    let cs = lower_program(&words, HwStage::Cs, false, compute_io()).unwrap();
    assert!(!cs.op_names.iter().any(|n| n == "dx.op.waveGetLaneIndex"));
}

#[test]
fn heap_uav_inference_marks_loads_and_scalar_buffer_loads_as_uav() {
    let words = [
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], null
        0xE030_0000,
        0x8002_0100, // buffer_load_dword v1, off, s[8:11], 0
        0xF420_0084,
        0xFA00_0000, // s_buffer_load_dword s2, s[8:11], null
        0x7E04_0202, // v_mov_b32 v2, s2
        0xE070_0000,
        0x8002_0100, // buffer_store_dword v1, off, s[8:11], 0
        0xE070_0004,
        0x8002_0200, // buffer_store_dword v2, off, s[8:11], 0 offset:4
        S_ENDPGM,
    ];
    let layout = vec![root(RootType::ResourceHeap, 0, 0)];
    let l = lower_with(&words, layout, Vec::new()).unwrap();
    validate(&l.dxil);
    assert!(
        !l.handle_props.is_empty() && l.handle_props.iter().all(|&p| p == (0x100B, 0)),
        "expected all heap handles to be raw UAV 0x100B, got {:?}",
        l.handle_props
    );
    let raw_load = l
        .op_names
        .iter()
        .position(|n| n == "dx.op.rawBufferLoad.f32")
        .expect("expected rawBufferLoad.f32");
    assert_eq!(
        l.insts
            .iter()
            .filter(|i| i.starts_with(&format!("call f{raw_load} ")))
            .count(),
        2,
        "expected both buffer_load_dword and s_buffer_load_dword to emit rawBufferLoad.f32: {:?}",
        l.insts
    );
    assert!(l.op_names.iter().any(|n| n == "dx.op.rawBufferStore.i32"));
}

#[test]
fn heap_buffer_load_glc_emits_globally_coherent_uav_and_validates() {
    let words = [
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], null (read-only glc heap view)
        0xF408_0300,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[0:1], 0x20 (store sink heap view)
        0xF408_0400,
        0xFA00_0040, // s_load_dwordx4 s[16:19], s[0:1], 0x40 (s_buffer_load glc heap view)
        0xE030_4000,
        0x8002_0100, // buffer_load_dword v1, off, s[8:11], 0 glc
        0xF421_00C8,
        0xFA00_0000, // s_buffer_load_dword s3, s[16:19], null glc
        0x7E04_0203, // v_mov_b32 v2, s3
        0xE070_0000,
        0x8003_0100, // buffer_store_dword v1, off, s[12:15], 0
        0xE070_0004,
        0x8003_0200, // buffer_store_dword v2, off, s[12:15], 0 offset:4
        S_ENDPGM,
    ];
    let layout = vec![root(RootType::ResourceHeap, 0, 0)];
    let l = lower_with(&words, layout, Vec::new()).unwrap();
    validate(&l.dxil);
    assert!(
        l.handle_props.contains(&(0x500B, 0)),
        "expected buffer_load_dword glc to emit 0x500B, got {:?}",
        l.handle_props
    );
    assert!(
        l.handle_props.contains(&(0x100B, 0)),
        "expected buffer_store_dword to emit 0x100B, got {:?}",
        l.handle_props
    );
    assert!(
        l.handle_props.contains(&(13, 0x1_0000)),
        "expected s_buffer_load_dword glc without store/atomic to remain CBV (13, 0x10000), got {:?}",
        l.handle_props
    );
}

#[test]
fn sixty_four_bit_image_atomics_lower_and_validate() {
    let words = [
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF40C_0501,
        0xFA00_0000, // s_load_dwordx8 s[20:27], s[2:3], null
        0xF05C_130A,
        0x0005_060D,
        0x0000_0004, // image_atomic_umax v[6:7], [v13,v4], s[20:27] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm
        S_ENDPGM,
    ];
    let layout = vec![root(RootType::Table, 0, 0)];
    let used = vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 8,
        user_data: 0,
        slot: 0,
    }];
    let l = lower_with(&words, layout, used).unwrap();
    validate(&l.dxil);
}

#[test]
fn spilled_inactive_mask_does_not_poison_other_lanes() {
    let words = [
        0xBE80_0385, // s_mov_b32 s0, 5
        0x8902_C17E, // s_xor_b32 s2, exec_lo, -1
        0xD761_0000,
        0x0001_0002, // v_writelane_b32 v0, s2, 0
        0xD761_0000,
        0x0001_0200, // v_writelane_b32 v0, s0, 1
        0xD760_0004,
        0x0001_0300, // v_readlane_b32 s4, v0, 1
        0xBF06_8504, // s_cmp_eq_u32 s4, 5
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE85_0380, // s_mov_b32 s5, 0
        S_ENDPGM,
    ];
    lowers(&words).unwrap();
}

#[test]
fn spilled_derived_lane_mask_restored_for_select_lowers_and_validates() {
    let cnd = 258 | (259 << 9) | (3 << 18); // v_cndmask_b32 v1, v2, v3, s3
    let words = [
        0x8902_C17E, // s_xor_b32 s2, exec_lo, -1 (inactive mask)
        0xD761_0000,
        0x0001_0002, // v_writelane_b32 v0, s2, 0
        0xD760_0003,
        0x0001_0100, // v_readlane_b32 s3, v0, 0
        0xD501_0001,
        cnd, // v_cndmask_b32 v1, v2, v3, s3
        S_ENDPGM,
    ];
    let l = lowers_cs(&words, false, 0).unwrap();
    validate(&l.dxil);
    assert!(!l.op_names.iter().any(|n| n == "dx.op.waveGetLaneIndex"));
}

#[test]
fn descriptor_data_handles_s_buffer_load_srv() {
    let words = [
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0001,
        0xFA00_0000, // s_load_dwordx4 s[0:3], s[2:3], null
        0xBF06_8002, // s_cmp_eq_u32 s2, 0
        0xBF85_0002, // s_cbranch_scc1 -> s_endpgm
        0xF420_0100,
        0xFA00_0000, // s_buffer_load_dword s4, s[0:3], null
        S_ENDPGM,
    ];
    let layout = vec![root(RootType::Table, 0, 0)];
    let used = vec![UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];
    let l = lower_with(&words, layout, used).unwrap();
    validate(&l.dxil);
}

#[test]
fn transitive_descriptor_phi_does_not_block_live_data_join() {
    let words = [
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0003, // s_cbranch_scc1 -> second descriptor load
        0xF408_0101,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[2:3], null
        0xBF82_0002, // s_branch -> join1
        0xF408_0101,
        0xFA00_0010, // s_load_dwordx4 s[4:7], s[2:3], 0x10
        0xBF06_8101, // join1: s_cmp_eq_u32 s1, 1
        0xBF85_0001, // s_cbranch_scc1 -> join2
        0xBE84_0381, // s_mov_b32 s4, 1
        0xBF07_8004, // join2: s_cmp_lg_u32 s4, 0
        0xBF85_0001, // s_cbranch_scc1 -> end
        0xBE88_0380, // s_mov_b32 s8, 0
        S_ENDPGM,
    ];
    let layout = vec![
        root(RootType::Table, 0, 0),
        root(RootType::Constants, 1, 0),
    ];
    let l = lower_with(&words, layout, Vec::new()).unwrap();
    validate(&l.dxil);
}

#[test]
fn transitive_lane_mask_phi_does_not_block_live_data_join() {
    let words = [
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0003, // s_cbranch_scc1 -> second v_cmp
        0xD4C2_0004,
        0x0002_0080, // v_cmp_eq_u32 s4, 0, v0
        0xBF82_0002, // s_branch -> join1
        0xD4C5_0004,
        0x0002_0080, // v_cmp_ne_u32 s4, 0, v0
        0xBF06_8101, // join1: s_cmp_eq_u32 s1, 1
        0xBF85_0001, // s_cbranch_scc1 -> join2
        0xBE84_0381, // s_mov_b32 s4, 1
        0xBF07_8004, // join2: s_cmp_lg_u32 s4, 0
        0xBF85_0001, // s_cbranch_scc1 -> end
        0xBE88_0380, // s_mov_b32 s8, 0
        S_ENDPGM,
    ];
    let l = lowers_cs(&words, false, 2).unwrap();
    validate(&l.dxil);
}

#[test]
fn m0_relative_moves_with_two_indices_translate() {
    // v1 = 5, v2 = 7, M0 = source offset 0, destination offset 1, then
    // the M0-relative instruction and an LDS store of its v1 (an effect,
    // so a value the emitter cannot produce fails the translation).
    let program = |rel: u32| {
        [
            0x7E02_0285, // v_mov_b32 v1, 5
            0x7E04_0287, // v_mov_b32 v2, 7
            0x7E06_0280, // v_mov_b32 v3, 0
            0xBEFC_03FF, // s_mov_b32 m0, literal
            0x0001_0000,
            rel,
            0xD834_0000, // ds_write_b32 v3, v1
            0x0000_0103,
            S_ENDPGM,
        ]
    };
    for rel in [
        0x7E02_D101, // v_swaprel_b32 v1, v1
        0x7E02_9101, // v_movrelsd_2_b32 v1, v1
    ] {
        lowers_lds(&program(rel)).unwrap_or_else(|e| panic!("{rel:08x}: {e}"));
    }
}

#[test]
fn tbuffer_store_format_lowers_and_validates() {
    let op = 13u32; // tbuffer_store_format_d16_xy
    let fmt = 29u32; // 16_16 float
    let words = [
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0100, // s_load_dwordx4 s[4:7], s[0:1], 0
        0xFA00_0000,
        0x7E00_0281, // v_mov_b32 v0, 1
        0x7E02_0282, // v_mov_b32 v1, 2
        0x3A << 26 | fmt << 19 | (op & 7) << 16 | 16,
        0x80 << 24 | (op >> 3 & 1) << 21 | (4 >> 2) << 16 | 1,
        S_ENDPGM,
    ];
    let layout = vec![root(RootType::Table, 0, 0)];
    let used = vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];
    let l = lower_with(&words, layout, used).unwrap();
    validate(&l.dxil);
}

#[test]
fn root_uav_subword_store_skips_get_dimensions_and_table_uav_keeps_it() {
    // buffer_store_short v1, v0, s[0:3], 0 offen -> dynamic write_subword CAS loop
    // buffer_store_short v1, off, s[0:3], 0 offset:2 -> aligned rawBufferStore.i16
    let root_words = |offen: bool, off: u32| -> Vec<u32> {
        let mut w = vec![
            0x8801_FF01,
            0x0001_0000, // s_or_b32 s1, s1, 0x10000
            0xBE82_0380, // s_mov_b32 s2, 0
            0xBE83_03C1, // s_mov_b32 s3, -1
            0x3400_0081, // v_lshlrev_b32 v0, 1, v0 (dynamic byte offset)
            0x7E02_02FF,
            0x0000_3C00, // v_mov_b32 v1, 0x3c00
        ];
        let enc0 = 0xE068_0000 | (u32::from(offen) << 12) | (off & 0xFFF);
        let enc1 = 0x8000_0100; // vaddr=v0, vdata=v1, srsrc=s[0:3], soffset=0
        w.extend([enc0, enc1, S_ENDPGM]);
        w
    };
    let root_layout = vec![RootParam {
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        sgprs: 2,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }];
    let root_used = vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 1,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];

    for (offen, off, expected_op) in [
        (true, 0u32, "dx.op.atomicCompareExchange.i32"),
        (false, 2u32, "dx.op.rawBufferStore.i16"),
    ] {
        let l = lower_with(
            &root_words(offen, off),
            root_layout.clone(),
            root_used.clone(),
        )
        .unwrap();
        validate(&l.dxil);
        assert!(
            l.op_names.iter().any(|n| n == expected_op),
            "expected {expected_op} for Root UAV subword store (offen={offen}), ops={:?}",
            l.op_names
        );
        assert!(
            !l.op_names.iter().any(|n| n == "dx.op.getDimensions"),
            "Root UAV subword store (offen={offen}) must not gate writes on dx.op.getDimensions"
        );
    }

    let table_words = [
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0100,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[0:1], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_02FF,
        0x0000_3C00, // v_mov_b32 v1, 0x3c00
        0xE068_1000,
        0x8001_0100, // buffer_store_short v1, v0, s[4:7], 0 offen
        S_ENDPGM,
    ];
    let table_layout = vec![root(RootType::Table, 0, 0)];
    let table_used = vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];
    let lt = lower_with(&table_words, table_layout, table_used).unwrap();
    validate(&lt.dxil);
    assert!(
        lt.op_names.iter().any(|n| n == "dx.op.getDimensions"),
        "Table UAV subword store should preserve bounds check via dx.op.getDimensions"
    );
}

#[test]
fn float_to_int_conversions_guard_poison_and_audit_depth_ops() {
    // Load x from UAV[0], run v_rcp_iflag_f32 + v_mul_legacy_f32 + v_fma_f32,
    // and convert via v_cvt_{u32,i32,flr_i32,rpi_i32}_f32.
    let vop1 = |vdst: u32, op: u32, src0: u32| (0x3F << 25) | (vdst << 17) | (op << 9) | src0;
    let words = [
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF408_0100,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[0:1], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E00_0280, // v_mov_b32 v0, 0
        0xE030_1000,
        0x8001_0100, // buffer_load_dword v1, v0, s[4:7], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        vop1(2, 0x2B, 0x101), // v_rcp_iflag_f32 v2, v1
        0x0804_0280 | 0x0400, // v_mul_legacy_f32 v2, v0, v2
        0xD513_0002,
        0x0402_0501 | (0x102 << 18), // v_fma_f32 v2, v1, v2, v2
        vop1(3, 0x07, 0x102), // v_cvt_u32_f32 v3, v2
        vop1(4, 0x08, 0x102), // v_cvt_i32_f32 v4, v2
        vop1(5, 0x0D, 0x102), // v_cvt_flr_i32_f32 v5, v2
        vop1(6, 0x0C, 0x102), // v_cvt_rpi_i32_f32 v6, v2
        0xE07C_1000,
        0x8001_0300, // buffer_store_dwordx4 v[3:6], v0, s[4:7], 0 offen
        S_ENDPGM,
    ];
    let layout = vec![root(RootType::Table, 0, 0)];
    let used = vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];
    let l = lower_with(&words, layout, used).unwrap();
    validate(&l.dxil);

    let cast_indices: Vec<usize> = l
        .insts
        .iter()
        .enumerate()
        .filter(|(_, s)| s.starts_with("cast:FpToUi ") || s.starts_with("cast:FpToSi "))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(cast_indices.len(), 4, "expected 4 float-to-int casts: {:?}", l.insts);
    for idx in cast_indices {
        let line = &l.insts[idx];
        let src_tok = line.split_whitespace().nth(1).unwrap();
        let src_inst: usize = src_tok
            .strip_prefix('i')
            .expect("cast input must be a guarded select instruction, not raw float/const")
            .parse()
            .unwrap();
        assert!(
            l.insts[src_inst].starts_with("select "),
            "expected guarded select before {line}, got {:?}",
            l.insts[src_inst]
        );
    }
}

mod target_shader_words {
    #[rustfmt::skip]
    pub const WORDS_4268: &[u32] = &[
        0xD58070F1, 0x03FF8BE7, 0x84BE21B9, 0xD5807D2C, 0x03FFF537, 0x61C7AA45, 0xBFA00003, 0xD7460000,
        0x0401060A, 0xD7460001, 0x0405060B, 0xBEA80305, 0xBE850380, 0xBEA90380, 0x7E080D00, 0xF40C0314,
        0xFA000060, 0xF40C0502, 0xFA000020, 0xF40C0714, 0xFA000020, 0x7E180D01, 0x0E2A0806, 0xF4080014,
        0xFA000000, 0xF4080902, 0xFA000000, 0xB0040092, 0x0E0E1807, 0x7E220315, 0xD54B0009, 0x045410F0,
        0xD52B0015, 0x000010FF, 0x40200000, 0xBE8503FF, 0x00C00000, 0x7E140307, 0xD54B0003, 0x041C12F0,
        0xD52B0011, 0x000010FF, 0x3FC00000, 0xD52B0007, 0x000012FF, 0x40200000, 0xBE860BA8, 0xD52B000A,
        0x000012FF, 0x3FC00000, 0xBE870380, 0xBF8CC07F, 0xF09C010A, 0x00271209, 0x00000003, 0xF09C0108,
        0x00270E09, 0xF09C010A, 0x00270D11, 0x0000000A, 0xF09C010A, 0x00270811, 0x00000003, 0xF09C010A,
        0x00270F15, 0x00000003, 0xF09C010A, 0x00270509, 0x00000007, 0xF09C010A, 0x00270311, 0x00000007,
        0xF09C010A, 0x00270A15, 0x0000000A, 0xF09C010A, 0x00270215, 0x00000007, 0xF4280000, 0xFA000000,
        0x7E000000, 0xF4240112, 0xFA000000, 0xBF8CC07F, 0x7E120C00, 0x7E200C02, 0x7E220C03, 0x7E160C01,
        0x7E0C0C04, 0x7E125709, 0x7E205710, 0x7E225711, 0x7E2C0C05, 0x7E16570B, 0xBE82047E, 0x0E261306,
        0x0E0C2106, 0x0E2E1716, 0x0E120913, 0x7E540506, 0x0E0C2316, 0x0E161917, 0xD54B0007, 0x00AA0913,
        0x7E5C0506, 0x7E0C0F09, 0x7E220F0B, 0x7E2602F2, 0x7E080F07, 0xD54B0010, 0x00BA1917, 0x7D880906,
        0x7E180F10, 0x7DA60906, 0xBF88002A, 0x7E280306, 0x7E2602F2, 0xD4C30000, 0x00021911, 0xBE8A047E,
        0xBE88047E, 0xBF820002, 0x87FE0A04, 0xBF800000, 0xBE9C2400, 0xBF880018, 0xF4280112, 0xFA000008,
        0x7E2A0311, 0xBE9E047E, 0xBEA0047E, 0xBF820002, 0x87FE2022, 0xBF800000, 0xBF800000, 0xBF800000,
        0xF0001108, 0x00051614, 0x4A2A2A81, 0xBEA2047E, 0xBF8C0070, 0xD54B0016, 0x40100D16, 0x0A2E2C05,
        0x7E2E5517, 0x0E2C2F16, 0x1E262716, 0x7DA81915, 0x8AA07E20, 0xBF85FFEE, 0xBEFE041C, 0x4A282881,
        0xBE84047E, 0x7DA80914, 0x8A8A7E0A, 0xBF85FFDE, 0xBEFE0408, 0xBEFE0402, 0x060E0E2A, 0xBF8C3F78,
        0x20242712, 0x7E2602F2, 0xBE88047E, 0x7E0E0F07, 0xD4C40000, 0x00020F04, 0x7DA60F04, 0xBF88002F,
        0xD4C30002, 0x00021911, 0x7E2602F2, 0x7E2A0304, 0xBE8A047E, 0xD5010014, 0x00098280, 0xBE82047E,
        0xBF820003, 0xBF800000, 0x87FE0A04, 0xBF800000, 0xBE9C047E, 0x7DA428C1, 0xBF880019, 0xBF800000,
        0xF4280112, 0xFA000008, 0x7E2C0311, 0xBE9E047E, 0xBEA0047E, 0xBF820002, 0x87FE2022, 0xBF800000,
        0xBF800000, 0xBF800000, 0xF0001108, 0x00051715, 0x4A2C2C81, 0xBEA2047E, 0xBF8C0070, 0xD54B0017,
        0x40100D17, 0x0A302E05, 0x7E305518, 0x0E2E3117, 0x1E262717, 0x7DA81916, 0x8AA07E20, 0xBF85FFEE,
        0xBEFE041C, 0x4A2A2A81, 0xBE84047E, 0x7DA80F15, 0x8A8A7E0A, 0xBF85FFDC, 0xBEFE0402, 0xBEFE0408,
        0xD52B0009, 0x0001E82A, 0xBF8C3F75, 0x20262708, 0xBE88047E, 0x0628122A, 0x7E100F09, 0x1E242712,
        0x7E2602F2, 0x7E120F14, 0xD4C40002, 0x00021308, 0x7DA61308, 0xBF88002F, 0xD4C30004, 0x00021911,
        0x7E2602F2, 0x7E2A0308, 0xBE8A047E, 0xBE9C047E, 0xD5010014, 0x00118280, 0xBF820003, 0xBF800000,
        0x87FE1C04, 0xBF800000, 0xBE9E047E, 0x7DA428C1, 0xBF880019, 0xBF800000, 0xF4280112, 0xFA000008,
        0x7E2C0311, 0xBEA0047E, 0xBEA2047E, 0xBF820002, 0x87FE222A, 0xBF800000, 0xBF800000, 0xBF800000,
        0xF0001108, 0x00051715, 0x4A2C2C81, 0xBEAA047E, 0xBF8C0070, 0xD54B0017, 0x40100D17, 0x0A302E05,
        0x7E305518, 0x0E2E3117, 0x1E262717, 0x7DA81916, 0x8AA27E22, 0xBF85FFEE, 0xBEFE041E, 0x4A2A2A81,
        0xBE84047E, 0x7DA81315, 0x8A9C7E1C, 0xBF85FFDC, 0xBEFE040A, 0xBEFE0408, 0x0622202E, 0xBF8C3F74,
        0x201E270F, 0x8988C16A, 0x7E2C0F11, 0x1E201F12, 0x7E2202F2, 0xBE8A2408, 0xBF88002A, 0x7E240306,
        0x7E2202F2, 0x7D862D0C, 0xBE9E047E, 0xBE9C047E, 0xBF820003, 0xBF800000, 0x87FE1E04, 0xBF800000,
        0xBEA0246A, 0xBF880018, 0xF4280112, 0xFA000008, 0x7E26030C, 0xBEA2047E, 0xBEAA047E, 0xBF820002,
        0x87FE2A2C, 0xBF800000, 0xBF800000, 0xBF800000, 0xF0001108, 0x00051412, 0x4A262681, 0xBEAC047E,
        0xBF8C0070, 0xD54B0014, 0x40100D14, 0x0A2A2805, 0x7E2A5515, 0x0E282B14, 0x1E222314, 0x7DA82D13,
        0x8AAA7E2A, 0xBF85FFEE, 0xBEFE0420, 0x4A242481, 0xBE84047E, 0x7DA80912, 0x8A9E7E1E, 0xBF85FFDE,
        0xBEFE041C, 0xBEFE040A, 0x201C230E, 0x7E1E02F2, 0x898AC100, 0x1E1C1D10, 0xBE80240A, 0xBF88002D,
        0x7D862D0C, 0x7E1E02F2, 0x7E240304, 0xBE9C047E, 0xBE9E047E, 0xD5010011, 0x01A98280, 0xBF820002,
        0x87FE1E04, 0xBF800000, 0xBEA0047E, 0x7DA422C1, 0xBF880019, 0xBF800000, 0xF4280112, 0xFA000008,
        0x7E26030C, 0xBEA2047E, 0xBEAA047E, 0xBF820002, 0x87FE2A2C, 0xBF800000, 0xBF800000, 0xBF800000,
        0xF0001108, 0x00051412, 0x4A262681, 0xBEAC047E, 0xBF8C0070, 0xD54B0014, 0x40100D14, 0x0A2A2805,
        0x7E2A5515, 0x0E282B14, 0x1E1E1F14, 0x7DA82D13, 0x8AAA7E2A, 0xBF85FFEE, 0xBEFE0420, 0x4A242481,
        0xBE84047E, 0x7DA80F12, 0x8A9E7E1E, 0xBF85FFDC, 0xBEFE041C, 0xBEFE0400, 0x201A1F0D, 0x7E2202F2,
        0x8984C102, 0x1E1A1B0E, 0xBE862404, 0xBF88002F, 0x7D862D0C, 0x7E2202F2, 0x7E240308, 0xBE9C047E,
        0xBE9E047E, 0xD5010010, 0x01A98280, 0xBF820002, 0x87FE1E00, 0xBF800000, 0xBEA0047E, 0x7DA420C1,
        0xBF88001B, 0xBF800000, 0xF4280012, 0xFA000008, 0x7E1E030C, 0xBEA2047E, 0xBEAA047E, 0xBF820002,
        0x87FE2A2C, 0xBF800000, 0xBF800000, 0xBF800000, 0xF000110A, 0x00051312, 0x0000000F, 0x4A1E1E81,
        0xBEAC047E, 0xBF8C0070, 0xD54B0013, 0x40000513, 0x0A282601, 0x7E285514, 0x0E262913, 0x1E222313,
        0x7DA82D0F, 0xBF800000, 0x8AAA7E2A, 0xBF85FFEC, 0xBEFE0420, 0x4A242481, 0xBE80047E, 0x7DA81312,
        0x8A9E7E1E, 0xBF85FFDA, 0xBEFE041C, 0xBEFE0406, 0xD52B000B, 0x0001E82E, 0xBF8C3F71, 0x2014230A,
        0x061C162E, 0x1E18150D, 0x7E140F0B, 0x7E160F0E, 0x7E1C02F2, 0xBE862408, 0xBF88002A, 0x7D86170A,
        0x7E1C02F2, 0xBE88047E, 0xBE9C047E, 0xBF820002, 0x87FE1C00, 0xBF800000, 0xBE9E246A, 0xBF88001A,
        0xF4280012, 0xFA000008, 0x7E22030A, 0xBEA0047E, 0xBEA2047E, 0xBF820002, 0x87FE222A, 0xBF800000,
        0xBF800000, 0xBF800000, 0xF000110A, 0x00050F06, 0x00000011, 0x4A222281, 0xBEAA047E, 0xBF8C0070,
        0xD54B000F, 0x4000050F, 0x0A201E01, 0x7E205510, 0x0E1E210F, 0x1E1C1D0F, 0x7DA81711, 0xBF800000,
        0x8AA27E22, 0xBF85FFEC, 0xBEFE041E, 0x4A0C0C81, 0xBE80047E, 0x7DA80906, 0x8A9C7E1C, 0xBF85FFDC,
        0xBEFE0408, 0xBEFE0406, 0x200A1D05, 0x7E2202F2, 0x1E0A0B0C, 0xBE86240A, 0xBF88002E, 0x7D86170A,
        0x7E2202F2, 0xBE88047E, 0xBE8A047E, 0xD501000C, 0x01A98280, 0xBF820002, 0x87FE0A00, 0xBF800000,
        0xBE9C047E, 0x7DA418C1, 0xBF88001B, 0xBF800000, 0xF4280012, 0xFA000008, 0x7E1A030A, 0xBE9E047E,
        0xBEA0047E, 0xBF820002, 0x87FE2022, 0xBF800000, 0xBF800000, 0xBF800000, 0xF000110A, 0x00050E04,
        0x0000000D, 0x4A1A1A81, 0xBEA2047E, 0xBF8C0070, 0xD54B000E, 0x4000050E, 0x0A1E1C01, 0x7E1E550F,
        0x0E1C1F0E, 0x1E22230E, 0x7DA8170D, 0xBF800000, 0x8AA07E20, 0xBF85FFEC, 0xBEFE041C, 0x4A080881,
        0xBE80047E, 0x7DA80F04, 0x8A8A7E0A, 0xBF85FFDA, 0xBEFE0408, 0xBEFE0406, 0x20062303, 0x7E0802F2,
        0x1E060705, 0xBE842404, 0xBF88002E, 0x7D86170A, 0x7E0802F2, 0xBE86047E, 0xBE88047E, 0xD5010005,
        0x01A98280, 0xBF820002, 0x87FE0800, 0xBF800000, 0xBE8A047E, 0x7DA40AC1, 0xBF88001B, 0xBF800000,
        0xF4280012, 0xFA000008, 0x7E0C030A, 0xBE9C047E, 0xBE9E047E, 0xBF820002, 0x87FE1E20, 0xBF800000,
        0xBF800000, 0xBF800000, 0xF000110A, 0x00050708, 0x00000006, 0x4A0C0C81, 0xBEA0047E, 0xBF8C0070,
        0xD54B0007, 0x40000507, 0x0A180E01, 0x7E18550C, 0x0E0E1907, 0x1E080907, 0x7DA81706, 0xBF800000,
        0x8A9E7E1E, 0xBF85FFEC, 0xBEFE040A, 0x4A101081, 0xBE80047E, 0x7DA81308, 0x8A887E08, 0xBF85FFDA,
        0xBEFE0406, 0xBEFE0404, 0xBF8C3F70, 0x20040902, 0xBE88047E, 0x1E080503, 0xD8D4201F, 0x02000004,
        0x7E0A0304, 0x7E0C0304, 0x7E0E0304, 0xF0201F08, 0x00030400, 0xBF8CC07F, 0x1E060504, 0xD7650002,
        0x000100C1, 0xD8D4041F, 0x08000003, 0xD7660002, 0x000204C1, 0x36120489, 0xBF8CC07F, 0x1E061103,
        0x7DA41280, 0xBF88000A, 0xF40C0014, 0xFA0000A0, 0x7E0C0303, 0x7E0A0303, 0x7E080303, 0x2C100281,
        0x2C0E0081, 0xBF8CC07F, 0xF0201F08, 0x00000307, 0xBEFE0408, 0xD8D4401F, 0x04000003, 0x360A049B,
        0xBE88047E, 0xBF8CC07F, 0x1E060903, 0xD8D4081F, 0x04000003, 0xBF8CC07F, 0x1E060903, 0x7DA40A80,
        0xBF88000A, 0xF40C0014, 0xFA000040, 0x7E0C0303, 0x7E0A0303, 0x7E080303, 0x2C100282, 0x2C0E0082,
        0xBF8CC07F, 0xF0201F08, 0x00000307, 0xBEFE0408, 0xD8D4101F, 0x04000003, 0xBF8CC07F, 0x1E060903,
        0xD7600008, 0x00014103, 0x7DA40480, 0xBF880001, 0xBF820001, 0xBF810000, 0xF40C0014, 0xFA000080,
        0x1E040608, 0x2C020283, 0x2C000083, 0x7E0A0302, 0x7E080302, 0x7E060302, 0xBF8CC07F, 0xF0201F08,
        0x00000200, 0xBF810000,
    ];
    #[rustfmt::skip]
    pub const WORDS_2891: &[u32] = &[
        0xD580FA81, 0x03FF2F8E, 0xCCCA8207, 0xD580E91E, 0x03FF6736, 0xB6B3BB94, 0xBFA00003, 0x8703FF01,
        0x000000FF, 0xBE820300, 0xBE8603C1, 0xF4080301, 0xFA00004C, 0xF4080201, 0xFA000000, 0xBE8303FF,
        0x11014FAC, 0xBE8203C1, 0xBE870303, 0xBF8CC07F, 0x8801FF0F, 0x00010000, 0x8805FF0D, 0x00010000,
        0x8F008D04, 0xBE84030C, 0x81080008, 0xBE80030E, 0xD7460004, 0x00210500, 0xBE9A107E, 0xE0301000,
        0x80010C04, 0xBF8C3F70, 0x361018FF, 0x00FFFFFF, 0x2C141898, 0xBE8C25C1, 0xD5010001, 0x00321480,
        0x4A0202E9, 0xDE54C101, 0x4A0202E9, 0x93601201, 0x4A0202E9, 0x00092401, 0x4A0202FA, 0xFF094001,
        0xD7780002, 0x02010101, 0x4A020302, 0xD760000E, 0x00014101, 0x4A02020E, 0xD760001B, 0x00010101,
        0xBEFE040C, 0xD7640009, 0x00010108, 0xBE8C25C1, 0xD5010001, 0x00321480, 0x7E040301, 0x4A0404FA,
        0xFF091102, 0x4A0404FA, 0xFF011202, 0x4A0404FA, 0xFF011402, 0x4A0404FA, 0xFF011802, 0xBEFE03FF,
        0xFFFF0000, 0xBEFF03FF, 0xFFFF0000, 0xD7784803, 0x03058302, 0x4A040503, 0xD760000E, 0x00013F02,
        0xBEFE0380, 0xBEFF03C1, 0x4A04040E, 0xBEFE03C1, 0xBEFF03C1, 0x4C040302, 0xBEFE040C, 0x7E080302,
        0x4A08081A, 0xD7460004, 0x00210504, 0xBE8C25C1, 0xD5010001, 0x00321280, 0x7E040301, 0x4A0404FA,
        0xFF091102, 0x4A0404FA, 0xFF011202, 0x4A0404FA, 0xFF011402, 0x4A0404FA, 0xFF011802, 0xBEFE03FF,
        0xFFFF0000, 0xBEFF03FF, 0xFFFF0000, 0xD7784803, 0x03058302, 0x4A040503, 0xD760000E, 0x00013F02,
        0xBEFE0380, 0xBEFF03C1, 0x4A04040E, 0xBEFE03C1, 0xBEFF03C1, 0x4C040302, 0xBEFE040C, 0xE0381000,
        0x80010404, 0x7E160302, 0xBE8C047E, 0x7DA818FF, 0x04000000, 0xBF88002C, 0x341E0083, 0x7E1A030A,
        0xBE9403FF, 0x7F800000, 0xBE9503FF, 0xFF800000, 0xBE90047E, 0xD746000C, 0x043D0900, 0xBE8E047E,
        0xBF820003, 0xBF800000, 0x87FE1012, 0xBF800000, 0x341C1A81, 0x7D8A1A80, 0x7E1E0214, 0x7E200215,
        0xBE92047E, 0xD746000E, 0x0439050D, 0xBF8C3F70, 0x020808C1, 0x7D8A1A81, 0xD747000E, 0x020A190E,
        0x020A0AC1, 0x7D8A1A82, 0xD8380302, 0x00100F0E, 0x7E200214, 0x020C0CC1, 0x7D8A1A83, 0x4A1A1A81,
        0xD8380100, 0x00100F0E, 0x7E1E0215, 0x7E200215, 0x020E0EC1, 0xD8380504, 0x00100F0E, 0x7DA21A83,
        0x8A907E10, 0xBF85FFE0, 0xBEFE040C, 0xBE8C047E, 0x7DAA1480, 0xBF880124, 0x341E0083, 0xBE8E047E,
        0xBE90047E, 0xBE9C0380, 0xD746000C, 0x043D0900, 0xBF820003, 0xBF800000, 0x87FE1012, 0xBF800000,
        0xBF06811C, 0x8592807E, 0xBF06821C, 0xBF8C3F70, 0xD501000D, 0x004A0B04, 0x8592807E, 0xBF06831C,
        0xD501000D, 0x004A0D0D, 0x8592807E, 0xD501000D, 0x004A0F0D, 0xBE92047E, 0x341C1A83, 0x36281A87,
        0x361A1CFF, 0xFFFFFFC0, 0x7DA42880, 0xBF880028, 0x381C1A90, 0xBF800000, 0xBF800000, 0xE0381000,
        0x8000170D, 0x381A1AB0, 0xBE94047E, 0xE0381000, 0x80001B0E, 0xE0381010, 0x8000130E, 0xE0301000,
        0x80000D0D, 0xBF8C3F72, 0xD551000E, 0x04763517, 0xBF8C3F71, 0xD551000F, 0x044E3919, 0xBF8C3F70,
        0x2C1A1A87, 0xD5510010, 0x047A3718, 0xD5540011, 0x04763517, 0xD5540012, 0x047A3718, 0xD5540013,
        0x044E3919, 0x361A1A81, 0x7DA41A81, 0xBF800000, 0x1E1C290E, 0x1E202B10, 0x1E1E2D0F, 0x20222911,
        0x20242B12, 0x20262D13, 0xBEFE0414, 0xBF800000, 0x8AFE7E12, 0xBF8800CC, 0xBE94047E, 0x7DA42881,
        0xBF880019, 0xBF800000, 0xBF800000, 0xBF800000, 0xBF800000, 0xE030100C, 0x8000110D, 0x381A1A90,
        0xE0381000, 0x8000120D, 0xE0381010, 0x8000160D, 0xBF8C3F70, 0xD551000E, 0x045E2911, 0xD5510010,
        0x04622B12, 0xD551000F, 0x04662D13, 0xD5540011, 0x045E2911, 0xD5540012, 0x04622B12, 0xD5540013,
        0x04662D13, 0xBF800000, 0x8AFE7E14, 0xBF8800AE, 0xBE96047E, 0x7DA42884, 0xBF880073, 0xBF800000,
        0x381C1A90, 0x381E1AB4, 0x38241A9C, 0x38361A98, 0x38381AB0, 0xE0281018, 0x8000100E, 0xE028101A,
        0x8000110E, 0xE0281000, 0x8000130F, 0xE0281002, 0x8000140F, 0xE0281000, 0x8000150E, 0xE0281002,
        0x8000160E, 0xE0281000, 0x80001712, 0xE0281002, 0x80001812, 0x381A1ABC, 0xE0281004, 0x8000190E,
        0xE0281006, 0x80001A0E, 0xE028101C, 0x80001D0E, 0xE028101E, 0x80000E0E, 0xE0281000, 0x80001E1B,
        0xE0281002, 0x80001B1B, 0xBF8C3F7C, 0xD7440010, 0x03FE2310, 0x01000504, 0xE0281004, 0x80001112,
        0xBF8C3F7B, 0xD7440013, 0x03FE2913, 0x01000504, 0xE0281006, 0x80001F12, 0xE0281008, 0x80000112,
        0xE0281000, 0x8000141C, 0xBF8C3F7C, 0xD7440015, 0x03FE2D15, 0x01000504, 0xBF8C3F7A, 0xD7440016,
        0x03FE3117, 0x01000504, 0xE0281006, 0x8000170F, 0xE0281000, 0x8000180D, 0xE0281004, 0x80000F0F,
        0xE0281002, 0x80001C1C, 0xE0281002, 0x80000D0D, 0xE028100A, 0x80001212, 0xCC110010, 0x18022710,
        0xCC110013, 0x18022D15, 0xCC110010, 0x18022113, 0xBF8C3F76, 0xD744000E, 0x03FE290E, 0x01000504,
        0xBF8C3F74, 0xD7440014, 0x03FE3117, 0x01000504, 0xD7440018, 0x03FE031F, 0x01000504, 0xD7440017,
        0x03FE3D1A, 0x01000504, 0xBF8C3F73, 0x741E1F1D, 0xBF8C3F71, 0x721A1B1C, 0xCC12000E, 0x1802290E,
        0xCC120014, 0x18023117, 0xD751000F, 0x043E2319, 0xBF8C3F70, 0xD754000D, 0x0436251B, 0xCC120013,
        0x18021D14, 0x7E1C1710, 0x7E1E170F, 0x7E2016F9, 0x00050610, 0x7E221713, 0x7E2416F9, 0x00050613,
        0x7E26170D, 0xBF800000, 0x8AFE7E16, 0xBF880036, 0x7E2602FF, 0xFF800000, 0x7E1E02FF, 0x7F800000,
        0xBE98047E, 0x7E240313, 0x7E220313, 0x7E20030F, 0x7E1C030F, 0x7DA42885, 0xBF88002B, 0xBF800000,
        0x383C1A90, 0x381A1AB8, 0xBF800000, 0xE0381018, 0x80000E1E, 0xE0381000, 0x8000121E, 0xE0341010,
        0x80001A1E, 0xE0341000, 0x80001C0D, 0xE0381048, 0x8000161E, 0xBF8C3F73, 0x1E1A1D12, 0x1E3E1F13,
        0x202A2315, 0x1E022114, 0xE0381030, 0x80000F1E, 0xE0341058, 0x8000131E, 0xBF8C3F73, 0x2038391A,
        0x203A3B1B, 0xE0341040, 0x80001A1E, 0xBF8C3F72, 0xD551000E, 0x045A1F0D, 0xD551000F, 0x04622301,
        0xD5540011, 0x04662515, 0xD5510010, 0x045E211F, 0xBF8C3F70, 0xD5540012, 0x044E351C, 0xD5540013,
        0x0452371D, 0xBF800000, 0xBEFE0412, 0x9312861C, 0x811C811C, 0xD747000D, 0x020A1812, 0xBE92047E,
        0xD8380302, 0x00110F0D, 0xD8380100, 0x00100E0D, 0xD8380504, 0x0013120D, 0x7DA4141C, 0xBF800000,
        0x8A907E10, 0xBF85FEE4, 0xBEFE040C, 0xBE8C047E, 0x7DA41280, 0x34140083, 0xD746000A, 0x04290900,
        0x8AFE7E0C, 0xBF880020, 0x34140083, 0x811B1A1B, 0xBE8E047E, 0x4A16161B, 0xBE90047E, 0xD746000A,
        0x04290900, 0xBE940380, 0xBF820003, 0xBF800000, 0x87FE1012, 0xBF800000, 0x4A181614, 0x7E1A7508,
        0x81148114, 0xBE92047E, 0xD746000C, 0x0021050C, 0xD51A000E, 0x0001030D, 0xD747000D, 0x020A150D,
        0xE0301000, 0x80010C0C, 0x3A10110E, 0xBF8C3F70, 0xD8340000, 0x00000C0D, 0x7DA41214, 0xBF800000,
        0x8A907E10, 0xBF85FFEA, 0xBEFE040C, 0x343C1482, 0xD7460001, 0x00210F00, 0xBF8C3F70, 0x36040886,
        0xBE84047E, 0xD8DC0100, 0x0800001E, 0xD8DC0302, 0x0A00001E, 0xD8DC0504, 0x0C00001E, 0xD8DC0706,
        0x0E00001E, 0xD8DC0908, 0x1000001E, 0xD8DC0B0A, 0x1200001E, 0xD8DC0D0C, 0x1400001E, 0xD8DC0F0E,
        0x1600001E, 0xD8DC1110, 0x1800001E, 0xD8DC1312, 0x1A00001E, 0xD8DC1514, 0x1C00001E, 0xD8DC1716,
        0x1E00001E, 0x2C000283, 0xE0781000, 0x80000401, 0xD7710000, 0x0215FF00, 0x1FFFFFF8, 0xBF8CCA7F,
        0xE0781010, 0x80000801, 0x7E100280, 0x7E120280, 0x7E140280, 0x7E160280, 0xBF8CC87F, 0xE0781020,
        0x80000C01, 0xBF8CC67F, 0xE0781030, 0x80001001, 0xBF8CC47F, 0xE0781040, 0x80001401, 0xBF8CC27F,
        0xE0781050, 0x80001801, 0xBF8CC07F, 0xE0781060, 0x80001C01, 0xE0781070, 0x80000801, 0x7DA40484,
        0xBF88000A, 0x34100883, 0x361010FF, 0xFFFFFFC0, 0x4E10100A, 0x2C101084, 0x361010FF, 0x0FFFFFFC,
        0x4A161009, 0xE0701000, 0x8000000B, 0x8AFE7E04, 0xBF880011, 0x36100887, 0xBF07800B, 0x8586807E,
        0x7D841080, 0x8786066A, 0xBE862406, 0xBF88000A, 0x34080883, 0x360808FF, 0xFFFFFFC0, 0x4E08080A,
        0x2C080884, 0x360808FF, 0x0FFFFFFC, 0x4A160809, 0xE0701000, 0x8000000B, 0xBEFE0404, 0x36080A86,
        0xBE84047E, 0x7DA40884, 0xBF88000A, 0x34080A83, 0x360808FF, 0xFFFFFFC0, 0x4E08080A, 0x2C080884,
        0x360808FF, 0x0FFFFFFC, 0x4A160809, 0xE0701000, 0x8000000B, 0x8AFE7E04, 0xBF880011, 0x36080A87,
        0xBF07800B, 0x8586807E, 0x7D840880, 0x8786066A, 0xBE862406, 0xBF88000A, 0x34080A83, 0x360808FF,
        0xFFFFFFC0, 0x4E08080A, 0x2C080884, 0x360808FF, 0x0FFFFFFC, 0x4A160809, 0xE0701000, 0x8000000B,
        0xBEFE0404, 0x36080C86, 0xBE84047E, 0x7DA40884, 0xBF88000A, 0x34080C83, 0x360808FF, 0xFFFFFFC0,
        0x4E08080A, 0x2C080884, 0x360808FF, 0x0FFFFFFC, 0x4A160809, 0xE0701000, 0x8000000B, 0x8AFE7E04,
        0xBF880011, 0x36080C87, 0xBF07800B, 0x8586807E, 0x7D840880, 0x8786066A, 0xBE862406, 0xBF88000A,
        0x34080C83, 0x360808FF, 0xFFFFFFC0, 0x4E08080A, 0x2C080884, 0x360808FF, 0x0FFFFFFC, 0x4A160809,
        0xE0701000, 0x8000000B, 0xBEFE0404, 0x36080E86, 0xBE84047E, 0x7DA40884, 0xBF88000A, 0x34080E83,
        0x360808FF, 0xFFFFFFC0, 0x4E08080A, 0x2C080884, 0x360808FF, 0x0FFFFFFC, 0x4A0E0809, 0xE0701000,
        0x80000007, 0x8AFE7E04, 0xBF880011, 0x36080E87, 0xBF07800B, 0x8584807E, 0x7D840880, 0x8784046A,
        0xBE842404, 0xBF88000A, 0x34080E83, 0x360808FF, 0xFFFFFFC0, 0x4E08080A, 0x2C080884, 0x360808FF,
        0x0FFFFFFC, 0x4A0E0809, 0xE0701000, 0x80000007, 0xBF810000,
    ];
    #[rustfmt::skip]
    pub const WORDS_CAA7: &[u32] = &[
        0xD580ADDE, 0x03FF2B97, 0xC2E73D1F, 0xD580707A, 0x03FF8F79, 0xB5C66F90, 0xBFA00003, 0x8703FF01,
        0x000000FF, 0xBE820300, 0xBEA203C1, 0xF4080201, 0xFA00004C, 0xF4080701, 0xFA000000, 0xBE9A03C1,
        0xBF8CC07F, 0x8819FF0B, 0x00010000, 0x8821FF09, 0x00010000, 0x8F008C04, 0xBEA00308, 0x811C001C,
        0xBE8003FF, 0x11014FAC, 0xD7460004, 0x00710500, 0xBEA30300, 0xBE98030A, 0xBE9B0300, 0xBEA8107E,
        0xE0301000, 0x80080C04, 0xBF8C3F70, 0x361018FF, 0x00FFFFFF, 0x2C141898, 0xBE8025C1, 0xD5010001,
        0x00021480, 0x4A0202E9, 0xDE54C101, 0x4A0202E9, 0x93601201, 0x4A0202E9, 0x00092401, 0x4A0202FA,
        0xFF094001, 0xD7780002, 0x02010101, 0x4A020302, 0xD7600002, 0x00014101, 0x4A020202, 0xD7600029,
        0x00010101, 0xBEFE0400, 0xD7640009, 0x00010108, 0xBE8025C1, 0xD5010001, 0x00021480, 0x7E040301,
        0x4A0404FA, 0xFF091102, 0x4A0404FA, 0xFF011202, 0x4A0404FA, 0xFF011402, 0x4A0404FA, 0xFF011802,
        0xBEFE03FF, 0xFFFF0000, 0xBEFF03FF, 0xFFFF0000, 0xD7784803, 0x03058302, 0x4A040503, 0xD7600002,
        0x00013F02, 0xBEFE0380, 0xBEFF03C1, 0x4A040402, 0xBEFE03C1, 0xBEFF03C1, 0x4C040302, 0xBEFE0400,
        0x7E080302, 0x4A080828, 0xD7460004, 0x00710504, 0xBE8025C1, 0xD5010001, 0x00021280, 0x7E040301,
        0x4A0404FA, 0xFF091102, 0x4A0404FA, 0xFF011202, 0x4A0404FA, 0xFF011402, 0x4A0404FA, 0xFF011802,
        0xBEFE03FF, 0xFFFF0000, 0xBEFF03FF, 0xFFFF0000, 0xD7784803, 0x03058302, 0x4A040503, 0xD7600002,
        0x00013F02, 0xBEFE0380, 0xBEFF03C1, 0x4A040402, 0xBEFE03C1, 0xBEFF03C1, 0x4C040302, 0xBEFE0400,
        0xE0381000, 0x80080404, 0x7E160302, 0xBE86047E, 0x7DA818FF, 0x04000000, 0xBF88002D, 0x341E0083,
        0x7E1A030A, 0xBE8A047E, 0xBE88047E, 0xD746000C, 0x043D0900, 0xBF820002, 0x87FE0A00, 0xBF800000,
        0x341C1A81, 0xD4C50004, 0x00021A83, 0xD4C50002, 0x00021A82, 0xD4C50000, 0x00021A81, 0x7D8A1A80,
        0xD746000E, 0x0439050D, 0x7E1E02FF, 0xFC00FC00, 0x7E2002FF, 0x7C007C00, 0xBF8C3F70, 0xD5010005,
        0x00020AC1, 0x7E2202FF, 0xFC007C00, 0xD747000E, 0x0206190E, 0x4A1A1A81, 0xD5010007, 0x00120EC1,
        0xD5010006, 0x000A0CC1, 0x020808C1, 0xBE80047E, 0xD8380100, 0x0011100E, 0xD8340008, 0x00000F0E,
        0x7DA21A83, 0xBF800000, 0x8A8A7E0A, 0xBF85FFDA, 0xBEFE0406, 0xBE96047E, 0x7DAA1480, 0xBF880194,
        0x341E0083, 0xBEA4047E, 0xBEA6047E, 0xBEAA0380, 0xD746000C, 0x043D0900, 0xBF820003, 0xBF800000,
        0x87FE2600, 0xBF800000, 0xBF06812A, 0x8580807E, 0xBF06822A, 0xBF8C3F70, 0xD501000D, 0x00020B04,
        0x8580807E, 0xBF06832A, 0xD501000D, 0x00020D0D, 0x8580807E, 0xD501000D, 0x00020F0D, 0xBE80047E,
        0x341C1A83, 0x36281A87, 0x36201CFF, 0xFFFFFFC0, 0x7DA42880, 0xBF880028, 0x381A2090, 0xBF800000,
        0xBF800000, 0xE0381000, 0x80061710, 0xBE82047E, 0xE0381000, 0x80061B0D, 0xE0381010, 0x8006130D,
        0x381A20B0, 0xE0301000, 0x8006100D, 0xBF8C3F72, 0xD551000D, 0x04763517, 0xBF8C3F71, 0xD551000E,
        0x044E3919, 0xD551000F, 0x047A3718, 0xD5540011, 0x04763517, 0xD5540012, 0x047A3718, 0xD5540013,
        0x044E3919, 0xBF8C3F70, 0x2C202087, 0x36202081, 0x7DA42081, 0xBF800000, 0x1E1A290D, 0x1E1E2B0F,
        0x1E1C2D0E, 0x20222911, 0x20242B12, 0x20262D13, 0xBEFE0402, 0xBF800000, 0x8AFE7E00, 0xBF8800CA,
        0xBE82047E, 0x7DA42881, 0xBF880017, 0xBF800000, 0x381A2090, 0xBF800000, 0xBF800000, 0xE030100C,
        0x80061110, 0xE0381000, 0x8006120D, 0xE0381010, 0x8006160D, 0xBF8C3F70, 0xD551000D, 0x045E2911,
        0xD551000F, 0x04622B12, 0xD551000E, 0x04662D13, 0xD5540011, 0x045E2911, 0xD5540012, 0x04622B12,
        0xD5540013, 0x04662D13, 0x8AFE7E02, 0xBF8800AE, 0xBE84047E, 0x7DA42884, 0xBF880073, 0xBF800000,
        0x381A2090, 0x381C20B4, 0x3824209C, 0x38362098, 0x383820B0, 0xE0281018, 0x80060F0D, 0xE028101A,
        0x8006110D, 0xE0281000, 0x8006130E, 0xE0281002, 0x8006140E, 0xE0281000, 0x8006150D, 0xE0281002,
        0x8006160D, 0xE0281000, 0x80061712, 0xE0281002, 0x80061812, 0x382020BC, 0xE0281004, 0x8006190D,
        0xE0281006, 0x80061A0D, 0xE028101C, 0x80061D0D, 0xE028101E, 0x80060D0D, 0xE0281000, 0x80061E1B,
        0xE0281002, 0x80061B1B, 0xBF8C3F7C, 0xD744000F, 0x03FE230F, 0x01000504, 0xE0281004, 0x80061112,
        0xBF8C3F7B, 0xD7440013, 0x03FE2913, 0x01000504, 0xE0281006, 0x80061F12, 0xE0281008, 0x80060112,
        0xE0281000, 0x8006141C, 0xBF8C3F7C, 0xD7440015, 0x03FE2D15, 0x01000504, 0xBF8C3F7A, 0xD7440016,
        0x03FE3117, 0x01000504, 0xE0281006, 0x8006170E, 0xE0281000, 0x80061810, 0xE0281004, 0x80060E0E,
        0xE0281002, 0x80061C1C, 0xE0281002, 0x80061010, 0xE028100A, 0x80061212, 0xCC11000F, 0x1802270F,
        0xCC110016, 0x18022D15, 0xCC11000F, 0x18021F16, 0xBF8C3F76, 0xD744000D, 0x03FE290D, 0x01000504,
        0xBF8C3F74, 0xD7440014, 0x03FE3117, 0x01000504, 0xD7440018, 0x03FE031F, 0x01000504, 0xD7440017,
        0x03FE3D1A, 0x01000504, 0xBF8C3F73, 0x741C1D1D, 0xBF8C3F71, 0x7220211C, 0xCC12000D, 0x1802290D,
        0xCC120014, 0x18023117, 0xD751000E, 0x043A2319, 0xBF8C3F70, 0xD7540010, 0x0442251B, 0xCC120013,
        0x18021B14, 0x7E1A170F, 0x7E1C170E, 0x7E1E16F9, 0x0005060F, 0x7E221713, 0x7E2416F9, 0x00050613,
        0x7E261710, 0xBF800000, 0x8AFE7E04, 0xBF880036, 0x7E2602FF, 0xFF800000, 0x7E1C02FF, 0x7F800000,
        0xBE86047E, 0x7E240313, 0x7E220313, 0x7E1E030E, 0x7E1A030E, 0x7DA42885, 0xBF88002B, 0xBF800000,
        0x381A2090, 0x381C20B8, 0xBF800000, 0xE0381018, 0x8006110D, 0xE0381000, 0x8006150D, 0xE0341010,
        0x80061D0D, 0xE0341000, 0x80061F0E, 0xE0381048, 0x8006190D, 0xBF8C3F73, 0x1E2C2516, 0x1E2E2717,
        0x20302918, 0x1E022315, 0xE0381030, 0x80060E0D, 0xE0341040, 0x8006140D, 0xE0341058, 0x8006120D,
        0xBF8C3F74, 0x203A3F1D, 0x203C411E, 0xBF8C3F72, 0xD551000D, 0x04661D01, 0xD551000F, 0x046A1F16,
        0xD551000E, 0x046E2117, 0xD5540011, 0x04722318, 0xBF8C3F70, 0xD5540012, 0x044A291D, 0xD5540013,
        0x044E2B1E, 0xBF800000, 0xBEFE0400, 0xD4010008, 0x00022680, 0xD4010006, 0x00022480, 0xD4010004,
        0x00022280, 0xD4040002, 0x00021C80, 0xD4040000, 0x00021E80, 0x7C081A80, 0xD5010017, 0x0019FE80,
        0x00001FFF, 0xD5010016, 0x0011FE80, 0x00001FFF, 0xD5010015, 0x0009FE80, 0x00001FFF, 0xD5010014,
        0x0001FE80, 0x00001FFF, 0xD5010010, 0x01A9FE80, 0x00001FFF, 0xD5010018, 0x0021FE80, 0x00001FFF,
        0x4A242517, 0x4A1C1D15, 0x4A1E1F14, 0x4A1A1B10, 0x4A222316, 0x4A262718, 0x361C1CFF, 0xFFFFE000,
        0x361E1EFF, 0xFFFFE000, 0x361A1AFF, 0xFFFFE000, 0x362022FF, 0xFFFFE000, 0x362224FF, 0xFFFFE000,
        0x362426FF, 0xFFFFE000, 0xD4030010, 0x00021CFF, 0x38800000, 0xD4030012, 0x00021EFF, 0x38800000,
        0xD406000E, 0x000220FF, 0xB8800000, 0xD406000C, 0x000222FF, 0xB8800000, 0xD406000A, 0x000224FF,
        0xB8800000, 0xD4030014, 0x00021AFF, 0x38800000, 0xD5010014, 0x004A1E80, 0x1E1E1EFF, 0xB8800000,
        0xD5010015, 0x00421C80, 0x1E1C1CFF, 0xB8800000, 0xD5010013, 0x00521A80, 0x1E1A1AFF, 0xB8800000,
        0xD5010016, 0x003A2080, 0x202020FF, 0x38800000, 0xD5010017, 0x00322280, 0x202222FF, 0x38800000,
        0xD5010018, 0x002A2480, 0x202424FF, 0x38800000, 0xD5010010, 0x00122116, 0xD501000E, 0x000A1D15,
        0xD5010011, 0x001A2317, 0xD501000F, 0x00021F14, 0xD5010012, 0x00222518, 0x021A1B13, 0x9300862A,
        0xBA800881, 0x00000000, 0x7E1C150E, 0xD7470013, 0x02061800, 0x7E2C150F, 0x7E1A150D, 0x7E201510,
        0x7E221511, 0x7E241512, 0x812A812A, 0xBE80047E, 0xD87C000A, 0x00001213, 0xD87C0008, 0x00001113,
        0xD87C0006, 0x00001013, 0xD87C0004, 0x00000E13, 0xD87C0002, 0x00001613, 0xD87C0000, 0x00000D13,
        0x7DA4142A, 0xBF800000, 0x8AA67E26, 0xBF85FE74, 0xBEFE0416, 0xBE80047E, 0x7DA41280, 0x34140083,
        0xD746000A, 0x04290900, 0x8AFE7E00, 0xBF880020, 0x34140083, 0x81292829, 0xBE890380, 0x8F088229,
        0xBE84047E, 0xD746000A, 0x04290900, 0x81081C08, 0xBE82047E, 0xBF820002, 0x87FE0406, 0xBF800000,
        0x4A181609, 0x7E1A7508, 0x81098109, 0xBE86047E, 0xD746000C, 0x0021030C, 0xD51A000E, 0x0001030D,
        0xD747000D, 0x0206150D, 0xE0281000, 0x80080C0C, 0x3A10110E, 0xBF8C3F70, 0xD87C0000, 0x00000C0D,
        0x7DA41209, 0xBF800000, 0x8A847E04, 0xBF85FFEA, 0xBEFE0400, 0x34241481, 0xD7460014, 0x00710D00,
        0xBE80047E, 0xD8DC0100, 0x08000012, 0xD8DC0302, 0x0A000012, 0xD8DC0504, 0x0C000012, 0xD8DC0706,
        0x0E000012, 0xD8DC0908, 0x10000012, 0xD8DC0B0A, 0x12000012, 0xBF8CC57F, 0x2C001090, 0x2C2A1290,
        0xBF8CC47F, 0x2C2C1490, 0x2C2E1690, 0xBF8CC37F, 0x2C301890, 0xE0681010, 0x80060814, 0xE0681014,
        0x80060914, 0xE0681018, 0x80060A14, 0xE068101C, 0x80060B14, 0xBF8C3F70, 0xE0781000, 0x80060414,
        0xE0681012, 0x80060014, 0xE0681016, 0x80061514, 0xE068101A, 0x80061614, 0xE068101E, 0x80061714,
        0xE0681020, 0x80060C14, 0xE0681022, 0x80061814, 0xE0681024, 0x80060D14, 0xBF8CC27F, 0xE0681028,
        0x80060E14, 0xE068102C, 0x80060F14, 0xBF8CC17F, 0xE0681030, 0x80061014, 0xE0681034, 0x80061114,
        0x2C002883, 0xBF8CC07F, 0x2C3C2490, 0x2C3E2690, 0x2C321A90, 0x2C341C90, 0x2C361E90, 0x2C382090,
        0x2C3A2290, 0xD7710000, 0x0211FF00, 0x1FFFFFF8, 0x36100886, 0xE0681026, 0x80061914, 0xE068102A,
        0x80061A14, 0xE068102E, 0x80061B14, 0xE0681032, 0x80061C14, 0xE0681038, 0x80061214, 0xE068103A,
        0x80061E14, 0xE0681036, 0x80061D14, 0xE068103C, 0x80061314, 0xE068103E, 0x80061F14, 0x7DA41084,
        0xBF88000A, 0x34100883, 0x361010FF, 0xFFFFFFC0, 0x4E10101E, 0x2C101084, 0x361010FF, 0x0FFFFFFC,
        0x4A16101D, 0xE0701000, 0x8006000B, 0x8AFE7E00, 0xBF880011, 0x36100887, 0xBF07801F, 0x8582807E,
        0x7D841080, 0x8782026A, 0xBE822402, 0xBF88000A, 0x34080883, 0x360808FF, 0xFFFFFFC0, 0x4E08081E,
        0x2C080884, 0x360808FF, 0x0FFFFFFC, 0x4A16081D, 0xE0701000, 0x8006000B, 0xBEFE0400, 0x36080A86,
        0xBE80047E, 0x7DA40884, 0xBF88000A, 0x34080A83, 0x360808FF, 0xFFFFFFC0, 0x4E08081E, 0x2C080884,
        0x360808FF, 0x0FFFFFFC, 0x4A16081D, 0xE0701000, 0x8006000B, 0x8AFE7E00, 0xBF880011, 0x36080A87,
        0xBF07801F, 0x8582807E, 0x7D840880, 0x8782026A, 0xBE822402, 0xBF88000A, 0x34080A83, 0x360808FF,
        0xFFFFFFC0, 0x4E08081E, 0x2C080884, 0x360808FF, 0x0FFFFFFC, 0x4A16081D, 0xE0701000, 0x8006000B,
        0xBEFE0400, 0x36080C86, 0xBE80047E, 0x7DA40884, 0xBF88000A, 0x34080C83, 0x360808FF, 0xFFFFFFC0,
        0x4E08081E, 0x2C080884, 0x360808FF, 0x0FFFFFFC, 0x4A16081D, 0xE0701000, 0x8006000B, 0x8AFE7E00,
        0xBF880011, 0x36080C87, 0xBF07801F, 0x8582807E, 0x7D840880, 0x8782026A, 0xBE822402, 0xBF88000A,
        0x34080C83, 0x360808FF, 0xFFFFFFC0, 0x4E08081E, 0x2C080884, 0x360808FF, 0x0FFFFFFC, 0x4A16081D,
        0xE0701000, 0x8006000B, 0xBEFE0400, 0x36080E86, 0xBE80047E, 0x7DA40884, 0xBF88000A, 0x34080E83,
        0x360808FF, 0xFFFFFFC0, 0x4E08081E, 0x2C080884, 0x360808FF, 0x0FFFFFFC, 0x4A0E081D, 0xE0701000,
        0x80060007, 0x8AFE7E00, 0xBF880011, 0x36080E87, 0xBF07801F, 0x8580807E, 0x7D840880, 0x8780006A,
        0xBE802400, 0xBF88000A, 0x34080E83, 0x360808FF, 0xFFFFFFC0, 0x4E08081E, 0x2C080884, 0x360808FF,
        0x0FFFFFFC, 0x4A0E081D, 0xE0701000, 0x80060007, 0xBF810000,
    ];
}

#[test]
fn bvh_28918b7150a7aad5_and_caa78b8ca62bc3d6_lower_and_validate() {
    let layout = vec![
        RootParam { root: RootType::Constants, table_offset_dwords: 0, sgprs: 19, base_register: 0, user_data: 16, slot: 0 },
        RootParam { root: RootType::Descriptor, table_offset_dwords: 0, sgprs: 2, base_register: 0, user_data: 35, slot: 1 },
        RootParam { root: RootType::Descriptor, table_offset_dwords: 0, sgprs: 2, base_register: 1, user_data: 37, slot: 2 },
        RootParam { root: RootType::Descriptor, table_offset_dwords: 0, sgprs: 2, base_register: 0, user_data: 39, slot: 3 },
    ];
    let used = vec![
        UsedResource { kind: ResourceType::Cbv, root: RootType::Constants, table_offset_dwords: 0, lower_bound: 0, stride: 0, dwords: 4, user_data: 16, slot: 0 },
        UsedResource { kind: ResourceType::Uav, root: RootType::Descriptor, table_offset_dwords: 0, lower_bound: 0, stride: 1, dwords: 4, user_data: 35, slot: 1 },
        UsedResource { kind: ResourceType::Uav, root: RootType::Descriptor, table_offset_dwords: 0, lower_bound: 1, stride: 1, dwords: 4, user_data: 37, slot: 2 },
    ];
    let dxbc = crate::dxbc::build(&[]);
    let c = Container::parse(&dxbc).unwrap();
    for (name, words, lds_granules) in [
        ("28918b7150a7aad5", target_shader_words::WORDS_2891, 12u32),
        ("caa78b8ca62bc3d6", target_shader_words::WORDS_CAA7, 6u32),
    ] {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 256,
            layout: layout.clone(),
            used: used.clone(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: (4 << 1) | (1 << 7) | (lds_granules << 15),
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build_with(&native.patched_code(), true, 256, true).unwrap();
        let res = resource::track(&p, &native.layout, native.stage).unwrap();
        let io_res = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();

        assert!(
            io_res.inputs.iter().any(|(_, inp)| *inp == crate::io::Input::GroupId(0)),
            "{name}: missing GroupId(0)"
        );
        assert_eq!(res.params.len(), 8, "{name}: expected cb0[0..3] + u0 + u1 spilled params");
        let secs = super::super::wave::sections(&p, 0);
        assert_eq!(secs.len(), 3, "{name}: expected 3 wave64 sections in block 0");

        for mode in [
            super::Wave64Mode::Wave32,
            super::Wave64Mode::Emulate,
            super::Wave64Mode::Native,
        ] {
            let l = lower(&Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io_res,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
                options: super::TranslateOptions { wave64: mode },
            })
            .unwrap_or_else(|e| panic!("{name} {mode:?} lower failed: {e}"));
            assert_eq!(l.root_uavs, vec![1, 2], "{name} {mode:?}: unexpected root_uavs");
            assert_eq!(l.user_data.len(), 4, "{name} {mode:?}: unexpected spilled cb0 user_data");
            assert!(
                !l.op_names.iter().any(|n| n == "dx.op.getDimensions"),
                "{name} {mode:?}: Root UAV stores must not emit dx.op.getDimensions"
            );
            if name == "caa78b8ca62bc3d6" {
                assert!(
                    l.op_names.iter().any(|n| n == "dx.op.atomicCompareExchange.i32"),
                    "{name} {mode:?}: expected buffer_store_short subword CAS on Root UAV"
                );
            }
            let hs = emit::emit_with(
                &c,
                &native,
                &p,
                &st,
                &io_res,
                &res,
                super::TranslateOptions { wave64: mode },
            )
            .unwrap();
            validate(&hs.dxil);
        }
    }
}

#[test]
fn depth_pyramid_42683eebec081520_lowers_and_validates() {
    let mut layout = Vec::new();
    for (slot, base) in [
        (0, 6), (1, 0), (2, 8), (3, 11), (4, 5), (5, 2), (6, 3),
        (7, 12), (8, 13), (9, 9), (10, 10), (11, 4), (12, 1), (13, 7),
    ] {
        layout.push(RootParam {
            root: RootType::ImmediateSampler,
            table_offset_dwords: 0,
            sgprs: 0,
            base_register: base,
            user_data: 0,
            slot,
        });
    }
    layout.extend([
        RootParam { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 71, user_data: 4, slot: 0 },
        RootParam { root: RootType::Table, table_offset_dwords: 8, sgprs: 1, base_register: 74, user_data: 4, slot: 0 },
        RootParam { root: RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 72, user_data: 5, slot: 1 },
        RootParam { root: RootType::Table, table_offset_dwords: 8, sgprs: 1, base_register: 73, user_data: 5, slot: 1 },
        RootParam { root: RootType::Table, table_offset_dwords: 16, sgprs: 1, base_register: 8, user_data: 5, slot: 1 },
        RootParam { root: RootType::Table, table_offset_dwords: 24, sgprs: 1, base_register: 6, user_data: 5, slot: 1 },
        RootParam { root: RootType::Table, table_offset_dwords: 32, sgprs: 1, base_register: 9, user_data: 5, slot: 1 },
        RootParam { root: RootType::Table, table_offset_dwords: 40, sgprs: 1, base_register: 7, user_data: 5, slot: 1 },
        RootParam { root: RootType::Constants, table_offset_dwords: 0, sgprs: 4, base_register: 500, user_data: 6, slot: 2 },
    ]);
    let used = vec![
        UsedResource { kind: ResourceType::Sampler, root: RootType::ImmediateSampler, table_offset_dwords: 0, lower_bound: 0, stride: 0, dwords: 4, user_data: 0, slot: 1 },
        UsedResource { kind: ResourceType::Srv, root: RootType::Table, table_offset_dwords: 0, lower_bound: 71, stride: 20, dwords: 4, user_data: 4, slot: 0 },
        UsedResource { kind: ResourceType::Srv, root: RootType::Table, table_offset_dwords: 8, lower_bound: 74, stride: 4, dwords: 8, user_data: 4, slot: 0 },
        UsedResource { kind: ResourceType::Srv, root: RootType::Table, table_offset_dwords: 0, lower_bound: 72, stride: 16, dwords: 4, user_data: 5, slot: 1 },
        UsedResource { kind: ResourceType::Srv, root: RootType::Table, table_offset_dwords: 8, lower_bound: 73, stride: 4, dwords: 8, user_data: 5, slot: 1 },
        UsedResource { kind: ResourceType::Uav, root: RootType::Table, table_offset_dwords: 16, lower_bound: 8, stride: 4, dwords: 8, user_data: 5, slot: 1 },
        UsedResource { kind: ResourceType::Uav, root: RootType::Table, table_offset_dwords: 24, lower_bound: 6, stride: 4, dwords: 8, user_data: 5, slot: 1 },
        UsedResource { kind: ResourceType::Uav, root: RootType::Table, table_offset_dwords: 32, lower_bound: 9, stride: 4, dwords: 8, user_data: 5, slot: 1 },
        UsedResource { kind: ResourceType::Uav, root: RootType::Table, table_offset_dwords: 40, lower_bound: 7, stride: 4, dwords: 8, user_data: 5, slot: 1 },
        UsedResource { kind: ResourceType::Cbv, root: RootType::Constants, table_offset_dwords: 0, lower_bound: 500, stride: 0, dwords: 4, user_data: 6, slot: 2 },
    ];
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: (10 << 1) | (1 << 7) | (1 << 8) | (1 << 11),
            threads: [8, 8, 1],
        }),
        code: target_shader_words::WORDS_4268
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
    };
    let p = ir::build_with(&native.patched_code(), true, 256, true).unwrap();
    let res = resource::track(&p, &native.layout, native.stage).unwrap();
    let io_res = io::raise(&p, &native).unwrap();
    let st = structure::structure(&p).unwrap();
    let dxbc = crate::dxbc::build(&[]);
    let c = Container::parse(&dxbc).unwrap();
    for mode in [
        super::Wave64Mode::Wave32,
        super::Wave64Mode::Emulate,
        super::Wave64Mode::Native,
    ] {
        let l = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io_res,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            options: super::TranslateOptions { wave64: mode },
        })
        .unwrap();
        let fp_to_ui: Vec<_> = l
            .insts
            .iter()
            .filter(|s| s.starts_with("cast:FpToUi "))
            .collect();
        assert!(!fp_to_ui.is_empty(), "4268 {mode:?}: expected v_cvt_u32_f32 casts");
        for line in fp_to_ui {
            let src_idx: usize = line
                .split_whitespace()
                .nth(1)
                .unwrap()
                .strip_prefix('i')
                .unwrap()
                .parse()
                .unwrap();
            assert!(
                l.insts[src_idx].starts_with("select "),
                "4268 {mode:?}: FpToUi must be guarded by select, got {:?}",
                l.insts[src_idx]
            );
        }
        let hs = emit::emit_with(
            &c,
            &native,
            &p,
            &st,
            &io_res,
            &res,
            super::TranslateOptions { wave64: mode },
        )
        .unwrap();
        validate(&hs.dxil);
    }
}

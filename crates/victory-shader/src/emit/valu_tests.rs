//! End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated
//! by `dxil.dll` when present, and executed on the host GPU and WARP.
//!
//! Covers RDNA2 hardware semantics that differ from naive DXIL lowering:
//! `v_med3_f32`/`v_med3_f16` with NaN operands, `v_fract_f32`/`v_fract_f16`
//! near `1.0`, turn-range reduction for `v_sin_*`/`v_cos_*`, and denormal
//! operands of `v_mul_legacy_f32`.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    vop3_mod(op, vdst, src0, src1, src2, 0, 0, 0, false)
}

/// VOP3 `op vdst, src0, src1, src2` with explicit `abs`, `neg`, `omod`, `clamp`.
#[allow(clippy::too_many_arguments)]
const fn vop3_mod(
    op: u16,
    vdst: u8,
    src0: u16,
    src1: u16,
    src2: u16,
    abs: u8,
    neg: u8,
    omod: u8,
    clamp: bool,
) -> [u32; 2] {
    [
        0x35 << 26
            | (op as u32 & 0x3FF) << 16
            | (clamp as u32) << 15
            | (abs as u32 & 7) << 8
            | vdst as u32,
        (neg as u32 & 7) << 29
            | (omod as u32 & 3) << 27
            | (src0 as u32 & 0x1FF)
            | (src1 as u32 & 0x1FF) << 9
            | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

/// MUBUF `buffer_store_short` / `buffer_store_short_d16_hi`.
const fn buffer_store_short(vdata: u8, srsrc: u8, byte_off: u16, high: bool) -> [u32; 2] {
    [
        0x38 << 26 | (if high { 0x1B } else { 0x1A }) << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in
/// `v0`), and stores `v0` to dword 0 of the raw UAV at heap index 0.
fn build_valu_cs(body: &[u32]) -> HostShader {
    build_cs(body, 0)
}

/// [`build_valu_cs`], with the lowering's refusal as an error.
fn try_build_valu_cs(body: &[u32]) -> Result<HostShader, String> {
    lower_valu_cs(body, 5 << 1, [1, 1, 1])
}

/// As [`build_valu_cs`], with `lds_blocks` × 512 bytes of LDS allocated.
fn build_cs(body: &[u32], lds_blocks: u32) -> HostShader {
    build_cs_with_store(body, lds_blocks, buffer_store_dword(0, 8, 0))
}

/// Builds a synthetic compute shader with a caller-selected final MUBUF store.
fn build_valu_cs_with_store(body: &[u32], store: [u32; 2]) -> HostShader {
    build_cs_with_store(body, 0, store)
}

fn build_cs_with_store(body: &[u32], lds_blocks: u32, store: [u32; 2]) -> HostShader {
    lower_cs_lds(&valu_cs_words(body, store), lds_blocks)
}

fn valu_cs_words(body: &[u32], store: [u32; 2]) -> Vec<u32> {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on
    // VGPRs just as they do in real shaders.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&store);
    words.push(S_ENDPGM);
    words
}

/// [`build_valu_cs`] with the program's `COMPUTE_PGM_RSRC2` and thread counts
/// given; the lowering error is returned.
fn lower_valu_cs(body: &[u32], rsrc2: u32, threads: [u32; 3]) -> Result<HostShader, String> {
    let words = valu_cs_words(body, buffer_store_dword(0, 8, 0));
    lower_cs_cfg(&words, rsrc2, threads)
}

/// Lowers `words` as a 1-thread compute program with [`build_valu_cs`]'s
/// layout: 4 root constants in `s0..s3`, the address of a table holding a
/// UAV in `s4`.
fn lower_cs(words: &[u32]) -> HostShader {
    lower_cs_lds(words, 0)
}

fn lower_cs_lds(words: &[u32], lds_blocks: u32) -> HostShader {
    lower_cs_cfg(words, 5 << 1 | lds_blocks << 15, [1, 1, 1]).expect("lower")
}

fn lower_cs_cfg(words: &[u32], rsrc2: u32, threads: [u32; 3]) -> Result<HostShader, String> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo { rsrc2, threads }),
        code,
    };

    try_lower_compute(&native)
}

fn lower_compute(native: &NativeProgram) -> HostShader {
    try_lower_compute(native).expect("lower")
}

fn try_lower_compute(native: &NativeProgram) -> Result<HostShader, String> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower(&lower::Parts {
        native,
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
    .map_err(|e| format!("{e:?}"))?;

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        stride_table: lowered.stride_table,
        root_uavs: lowered.root_uavs,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    Ok(shader)
}

/// One GPU test at a time. With these tests on the hardware adapter and WARP
/// from several test threads at once, now and then none of their dispatches
/// ever completes and the test binary hangs; and two GPU tests at once on
/// WARP can end the test process with an access violation.
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    run_cases_on(shader, cases, &[("hardware", false), ("warp", true)], check);
}

/// As [`run_cases`], on the given backends only (programs that need a wave
/// size WARP lacks: a one-wave group with LDS takes the console's).
fn run_cases_on(
    shader: &HostShader,
    cases: &[[u32; 4]],
    backends: &[(&str, bool)],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for &(label, warp) in backends {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    (root, _) => super::host::guard(root, 1, 0)
                        .unwrap_or_else(|| panic!("unexpected user data {:?}", (root, u.slot))),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device of {backends:?} available");
}

#[test]
fn scalar_buffer_load_uses_byte_offsets_with_a_raw_host_view() {
    let input = s_load_dwordx4(4, 0);
    let mut output = s_load_dwordx4(8, 0);
    output[1] |= 32; // second heap slot
    let store = buffer_store_dword(0, 8, 0);
    let words = [
        s_mov_b32(1, 0x80),
        input[0],
        input[1],
        output[0],
        output[1],
        // s_buffer_load_dword s12, s[4:7], 20
        0x3D << 26 | 0x08 << 18 | 12 << 6 | 2,
        125 << 25 | 20,
        vop1(0x001, 0, sgpr(12)),
        store[0],
        store[1],
        S_ENDPGM,
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
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                stride: 16,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Uav,
                stride: 0,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let shader = lower_compute(&native);
    let data = [81, 82, 83, 84, 85, 86, 87, 88];
    let host = Host::new(false)
        .or_else(|_| Host::new(true))
        .expect("SM 6.6 device");
    let (root, pipeline) = host.compute_pipeline(&shader).unwrap();
    let mut ud = vec![0; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let Some(g) = super::host::guard(u.root, 2, 0) {
            ud[u.index as usize] = g;
        }
    }
    let out = host
        .dispatch(
            &root,
            &pipeline,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[
                    View::Buffer {
                        data: &data,
                        stride: Stride::Raw,
                    },
                    View::RawUav(1),
                ],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .expect("dispatch scalar load");
    assert_eq!(out[0], [86], "byte offset 20 in the raw host view");
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

#[test]
fn packed_add_selects_halves_of_inline_constants() {
    // Add the inline constant 1/(2*pi) to v1 (zero) to read back its two halves.
    // The same GFX10 source field (248) has a different bit pattern for a
    // packed floating-point and a packed integer instruction.
    let cases = [
        // v_pk_add_f16 v0, 0x3118, v1
        ([0xcc0f4000, 0x1a0202f8], 0x0000_3118),
        // v_pk_add_f16 v0, 0x3118, v1 op_sel_hi:[0,1] (use low twice)
        ([0xcc0f4000, 0x120202f8], 0x3118_3118),
        // v_pk_add_u16 v0, 0x3e22f983, v1
        ([0xcc0a4000, 0x1a0202f8], 0x3e22_f983),
    ];
    for (words, expected) in cases {
        let shader = build_valu_cs(&words);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, expected, "{backend}: {words:08x?}");
        });
    }
}

#[test]
fn v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;

    let cases: &[([u32; 4], Option<f32>)] = &[
        // Normal medians (all orderings).
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        // Two NaNs: returns the sole non-NaN operand.
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        // Three NaNs: returns NaN.
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert_eq!(
                got_f,
                want,
                "{backend}: v_med3_f32({:?}, {:?}, {:?}) = {got_f} ({got:#010x}), want {want}",
                f32::from_bits(inp[0]),
                f32::from_bits(inp[1]),
                f32::from_bits(inp[2]),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_med3_f32(NaN, NaN, NaN) = {got_f} ({got:#010x}), want NaN",
            ),
        }
    });
}

/// RDNA2 orders -0 below +0 in its maximums and minimums; DXIL's FMax and
/// FMin compare and select, which can give either zero.
#[test]
fn max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    // Each case has a -0 and a +0 among its first two operands.
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f32", 0x110, pz),
        ("v_min_f32", 0x10F, nz),
        ("v_max3_f32", 0x154, pz),
        ("v_min3_f32", 0x151, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    }
}

#[test]
fn v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let h0 = 0x0000u32; // 0.0
    let h1 = 0x3C00u32; // 1.0
    let h2 = 0x4000u32; // 2.0
    let h3 = 0x4200u32; // 3.0
    let h5 = 0x4500u32; // 5.0
    let hn2 = 0xC000u32; // -2.0
    let nan = 0x7E00u32;

    let cases: &[([u32; 4], Option<u16>)] = &[
        ([h1, h2, h3, 0], Some(h2 as u16)),
        ([hn2, h0, h3, 0], Some(h0 as u16)),
        ([h1, h2, nan, 0], Some(h1 as u16)),
        ([nan, h1, h0, 0], Some(h0 as u16)),
        ([nan, h0, h1, 0], Some(h0 as u16)),
        ([h2, nan, h1, 0], Some(h1 as u16)),
        ([nan, h5, nan, 0], Some(h5 as u16)),
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_med3_f16(NaN, NaN, NaN) = {got_h:#06x}, want NaN",
            ),
        }
    });
}

/// `v_cmp_lg_*` is "less or greater": ordered and not equal, so false when
/// either operand is a NaN (RDNA2 ISA; the same on gfx1036 hardware).
#[test]
fn v_cmp_lg_is_false_for_nan() {
    // v_cmp_lg_{f32,f16}_e64 s20, v1, v2 (VOPC 0x005 / 0x0cd in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    let nan = f32::NAN.to_bits();
    let (inf, ninf) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
    let f = f32::to_bits;
    let f32_cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(2.0), 0, 0], 1),
        ([f(2.0), f(1.0), 0, 0], 1),
        ([f(1.0), f(1.0), 0, 0], 0),
        ([f(0.0), f(-0.0), 0, 0], 0),
        ([inf, ninf, 0, 0], 1),
        ([inf, inf, 0, 0], 0),
        ([nan, f(1.0), 0, 0], 0),
        ([f(1.0), nan, 0, 0], 0),
        ([nan, nan, 0, 0], 0),
        ([0x7F80_0001, f(0.0), 0, 0], 0), // a signalling NaN
    ];
    let f16_cases: &[([u32; 4], u32)] = &[
        ([0x3C00, 0x4000, 0, 0], 1),
        ([0x4000, 0x3C00, 0, 0], 1),
        ([0x3C00, 0x3C00, 0, 0], 0),
        ([0x0000, 0x8000, 0, 0], 0),
        ([0x7C00, 0xFC00, 0, 0], 1),
        ([0x7E00, 0x3C00, 0, 0], 0),
        ([0x3C00, 0x7E00, 0, 0], 0),
        ([0x7E00, 0x7E00, 0, 0], 0),
        ([0x7C01, 0x0000, 0, 0], 0),           // a signalling NaN
        ([0xABCD_3C00, 0x1234_7E00, 0, 0], 0), // only the low halves are read
    ];
    for (op, what, cases) in [
        (0x005u16, "v_cmp_lg_f32", f32_cases),
        (0x0cd, "v_cmp_lg_f16", f16_cases),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

#[test]
fn v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    let shader = build_valu_cs(&[vop1(0x020, 0, vgpr(1))]);
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;

    let cases: &[(u32, Option<u32>)] = &[
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        // Normal values.
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        // Non-finite inputs yield NaN.
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        match expect {
            Some(want) => assert_eq!(
                got,
                want,
                "{backend}: v_fract_f32({:?} / {:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
                f32::from_bits(inp[0]),
                inp[0],
                f32::from_bits(got),
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_fract_f32({:?}) = {got:#010x}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });
}

#[test]
fn v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    let shader = build_valu_cs(&[vop1(0x05f, 0, vgpr(1))]);
    let below_one = 0x3BFFu16; // 1.0 - 2^-11

    assert_eq!(f16_to_f32(0x9400), -2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x9000), -2.0f32.powi(-11));
    assert_eq!(f16_to_f32(0x8C00), -2.0f32.powi(-12));

    let cases: &[(u16, Option<u16>)] = &[
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        (0x8C00, Some(below_one)), // -2^-12
        (0x8400, Some(below_one)), // -2^-14
        (0x8001, Some(below_one)), // -2^-24 (smallest negative f16 subnormal)
        (0x9000, Some(below_one)), // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        (0x9400, Some(0x3BFE)),    // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        (0x0000, Some(0x0000)), // 0.0 -> 0.0
        (0x3400, Some(0x3400)), // 0.25 -> 0.25
        (0xB400, Some(0x3A00)), // -0.25 -> 0.75
        (0x4300, Some(0x3800)), // 3.5 -> 0.5
        (0x3C00, Some(0x0000)), // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        (0x7E00, None), // NaN
        (0x7C00, None), // +inf
        (0xFC00, None), // -inf
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

#[test]
fn v_sin_and_v_cos_f32_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x035, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x036, 0, vgpr(1))]);
    let f = f32::to_bits;
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let sin_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(0.0)),
        (0.25, Some(1.0)),
        (255.0, Some(0.0)),
        (255.25, Some(1.0)),
        (255.5, Some(0.0)),
        (-255.25, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_sin_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_sin_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    let cos_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(1.0)),
        (0.25, Some(0.0)),
        (255.0, Some(1.0)),
        (255.25, Some(0.0)),
        (255.5, Some(-1.0)),
        (-255.5, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_cos_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_cos_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    // Main's `trig_turns` preserves sin(-0.0) = -0.0 and produces exact +0.0 at
    // half-turns (sin) and quarter-turns (cos).
    let exact_sin: &[(u32, u32)] = &[
        (0x8000_0000, 0x8000_0000), // sin(-0.0) = -0.0
        (f(255.0), 0),              // sin(255.0) = +0.0
        (f(255.5), 0),              // sin(255.5) = +0.0
    ];
    let exact_sin_in: Vec<[u32; 4]> = exact_sin.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&sin_shader, &exact_sin_in, |backend, i, inp, got| {
        let want = exact_sin[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_sin_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let exact_cos: &[(u32, u32)] = &[
        (f(0.25), 0),   // cos(0.25) = +0.0
        (f(255.25), 0), // cos(255.25) = +0.0
    ];
    let exact_cos_in: Vec<[u32; 4]> = exact_cos.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&cos_shader, &exact_cos_in, |backend, i, inp, got| {
        let want = exact_cos[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cos_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_sin_and_v_cos_f16_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x060, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x061, 0, vgpr(1))]);

    // Sanity-check the f16 bit encodings of large turns.
    assert_eq!(f16_to_f32(0x5BF8), 255.0);
    assert_eq!(f16_to_f32(0x5BFA), 255.25);
    assert_eq!(f16_to_f32(0x5BFC), 255.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);

    let sin_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(0.0)),  // 255.0
        (0x5BFA, Some(1.0)),  // 255.25
        (0x5BFC, Some(0.0)),  // 255.5
        (0xDBFA, Some(-1.0)), // -255.25
        (0x7BFF, Some(0.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_sin_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_sin_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });

    let cos_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(1.0)),  // 255.0
        (0x5BFA, Some(0.0)),  // 255.25
        (0x5BFC, Some(-1.0)), // 255.5
        (0x7BFF, Some(1.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_cos_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_cos_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// A 32-bit literal in a signed 64-bit operand is sign-extended (an
/// unsigned one's is zero-extended): -100 compared with `v[1:2]`, and
/// shifted by `s_ashr_i64`.
#[test]
fn signed_64_bit_operands_sign_extend_their_literal() {
    const MINUS_100: u32 = -100_i32 as u32;
    let check = |name: &str, body: &[u32], cases: &[([u32; 4], u32)]| {
        let shader = build_valu_cs(body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    };
    // v_cmp_*_i64 s6, v[1:2], -100 (VOP3, literal src1), then
    // v_cndmask_b32 v0, 0, 1, s6.
    let compare = |op: u16| {
        let [a, b] = vop3(op, 6, vgpr(1), 255, 0);
        let [c, d] = vop3(0x101, 0, 0x80, 0x81, sgpr(6));
        [a, b, MINUS_100, c, d]
    };
    // An i64 as the two input dwords of `v[1:2]`.
    let pair = |n: i64| [n as u32, (n >> 32) as u32, 0, 0];
    let eq = [(pair(-100), 1), ([MINUS_100, 0, 0, 0], 0), (pair(0), 0)];
    check("v_cmp_eq_i64", &compare(0x0A2), &eq);
    let lt = [(pair(-200), 1), (pair(-50), 0), (pair(0), 0)];
    check("v_cmp_lt_i64", &compare(0x0A1), &lt);
    // s_ashr_i64 s[6:7], -100, 4; v_mov_b32 v0, s7: the high dword.
    let ashr = [sop2(0x23, 6, 255, 0x84), MINUS_100, vop1(0x001, 0, sgpr(7))];
    check("s_ashr_i64", &ashr, &[([0; 4], u32::MAX)]);
}

#[test]
fn bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    let v_signed = build_valu_cs(&vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    let v_unsigned = build_valu_cs(&vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_signed = build_valu_cs(&[sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);

    let signed_cases: &[([u32; 4], u32)] = &[
        // Field crosses bit 31: extends from bit 31.
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        // Normal fields inside 32 bits.
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0),
        // while 31 extracts all remaining bits from offset 4.
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
    ];
    let v_in: Vec<[u32; 4]> = signed_cases.iter().map(|c| c.0).collect();
    run_cases(&v_signed, &v_in, |backend, i, inp, got| {
        let want = signed_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_i32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
    // v_bfe_i32 width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    run_cases(
        &v_signed,
        &[[0x8000_0000, 4, 32, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0, "{backend}: v_bfe_i32 width 32 should mask to 0");
        },
    );

    // s_bfe_i32 width is 7-bit (src1[22:16]), so width=32 keeps all 32 bits.
    let mut s_cases = signed_cases.to_vec();
    s_cases.push(([0x8000_0000, 4, 32, 0], 0xf800_0000));
    let s_in: Vec<[u32; 4]> = s_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_signed, &s_in, |backend, i, inp, got| {
        let want = s_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    let unsigned_cases: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0x0000_000d),
        ([0x8000_0000, 31, 4, 0], 0x0000_0001),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0x0800_0000),
    ];
    let u_in: Vec<[u32; 4]> = unsigned_cases.iter().map(|c| c.0).collect();
    run_cases(&v_unsigned, &u_in, |backend, i, inp, got| {
        let want = unsigned_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_u32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

/// `s_and_b32 s7, s6, 63; v_mov_b32 v0, s7` on a program with 6 user SGPRs
/// (s0..s5) and `TG_SIZE_EN`, which puts the thread-group info in s6.
const TG_INFO_RSRC2: u32 = 6 << 1 | 1 << 10;

#[test]
fn thread_group_info_wave_count_matches_the_group() {
    // (threads in X, waves of 32): the info SGPR holds the wave count in 5:0.
    for (threads, waves) in [(1, 1), (32, 1), (33, 2), (64, 2), (96, 3), (256, 8)] {
        let body = [sop2(0x0E, 7, 6, 128 + 63), vop1(0x001, 0, sgpr(7))];
        let shader = lower_valu_cs(&body, TG_INFO_RSRC2, [threads, 1, 1]).expect("lower");
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, waves, "{backend}: {threads} threads");
        });
        // s_bfe_u32 s7, s6, (width 6 << 16 | offset 0): the same field.
        let body = [sop2(0x27, 7, 6, 0xFF), 6 << 16, vop1(0x001, 0, sgpr(7))];
        let shader = lower_valu_cs(&body, TG_INFO_RSRC2, [threads, 1, 1]).expect("lower");
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, waves, "{backend}: {threads} threads (bfe)");
        });
    }
}

#[test]
fn thread_group_info_bits_above_the_wave_count_are_refused() {
    // The wave id / ordered-append field is not derived.
    let body = [sop2(0x0E, 7, 6, 128 + 63), vop1(0x001, 0, sgpr(6))];
    assert!(lower_valu_cs(&body, TG_INFO_RSRC2, [64, 1, 1]).is_err());
    // s_and_b32 with a wider mask (a literal: 0xFFFF).
    let body = [sop2(0x0E, 7, 6, 0xFF), 0xFFFF, vop1(0x001, 0, sgpr(7))];
    assert!(lower_valu_cs(&body, TG_INFO_RSRC2, [64, 1, 1]).is_err());
}

/// Indexed vector loads from a CBV, stored to a raw UAV for comparison with host data.
fn build_indexed_cbv_cs(byte_offset: u16, dynamic: bool) -> HostShader {
    let mut words = vec![s_mov_b32(1, 0x80), s_mov_b32(3, 0x80)];
    words.extend(s_load_dwordx4(8, 0));
    words.extend(s_load_dwordx4(12, 2));
    // v0/v1 hold the input row and optional byte offset, v2 the output offset.
    words.push(vop1(1, 1, sgpr(4)));
    words.extend([
        0x3404_0084,         // v_lshlrev_b32 v2, 4, v0
        vop1(1, 3, sgpr(5)), // v_mov_b32 v3, s5; group ID
        0x3406_0689,         // v_lshlrev_b32 v3, 9, v3; output group base
        0x4a04_0503,         // v_add_nc_u32 v2, v3, v2
        0xE038_2000 | (u32::from(dynamic) << 12) | u32::from(byte_offset),
        0x8002_0400, // buffer_load_dwordx4 v[4:7], v0 or v[0:1], s[8:11]
        0xE078_1000,
        0x8003_0402, // buffer_store_dwordx4 v[4:7], v2, s[12:15], 0 offen
        S_ENDPGM,
    ]);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 2,
                slot: 1,
            },
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 2,
            },
        ],
        used: [(ResourceType::Cbv, 0, 0), (ResourceType::Uav, 2, 1)]
            .into_iter()
            .map(|(kind, user_data, slot)| UsedResource {
                kind,
                user_data,
                slot,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
            })
            .collect(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1 | 1 << 7,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    lower_compute(&native)
}

#[test]
fn indexed_cbv_loads_match_their_byte_addresses() {
    let input: Vec<u32> = (0u32..192)
        .map(|i| (i + 1).wrapping_mul(0x91de_2f71).rotate_left(i % 32))
        .collect();
    let variants = [(0, false), (16, false), (12, false), (0, true)]
        .map(|(offset, dynamic)| (offset, dynamic, build_indexed_cbv_cs(offset, dynamic)));
    let mut ran = false;

    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        ran = true;
        for (offset, dynamic, shader) in &variants {
            let (root, pso) = host.compute_pipeline(shader).expect("compute pipeline");
            let byte_offsets: &[u32] = if *dynamic { &[0, 4, 12, 16] } else { &[0] };
            for &dynamic_offset in byte_offsets {
                let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
                for u in &shader.user_data {
                    ud[u.index as usize] = match (u.root, u.slot) {
                        (RootType::Table, 0) => 0,
                        (RootType::Table, 1) => 32,
                        (RootType::Constants, 2) => dynamic_offset,
                        (root, _) => super::host::guard(root, 2, 0)
                            .unwrap_or_else(|| panic!("unexpected user data {:?}", (root, u.slot))),
                    };
                }
                let actual = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(ud))],
                            views: &[View::Cbv(&input), View::RawUav(256)],
                            samplers: &[],
                            groups: [2, 1, 1],
                        },
                    )
                    .expect("compute dispatch");
                let dword = (usize::from(*offset) + dynamic_offset as usize) / 4;
                assert_eq!(actual[0].len(), 256);
                for group in actual[0].as_chunks::<128>().0 {
                    assert_eq!(
                        group.as_slice(),
                        &input[dword..dword + 128],
                        "{label}: offset {dword}"
                    );
                }
            }
        }
    }
    assert!(ran, "neither hardware nor WARP SM 6.6 device available");
}

/// `s_cselect_b32 s13, 1, 0`.
fn scc_after(op: u8) -> HostShader {
    build_valu_cs(&[
        sop2(op, 12, 0, 1),
        sop2(0x0A, 13, 129, 128), // s_cselect_b32 s13, 1, 0
        vop1(0x001, 0, sgpr(13)), // v_mov_b32 v0, s13
    ])
}

/// Checks the SCC that `op` leaves for each `(s0, s1, scc)`.
fn check_scc(name: &str, op: u8, cases: &[(u32, u32, u32)]) {
    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(a, b, _)| [a, b, 0, 0]).collect();
    run_cases(&scc_after(op), &inputs, |backend, i, inp, got| {
        let want = cases[i].2;
        assert_eq!(
            got, want,
            "{backend}: SCC after {name}({:#010x}, {:#010x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn s_min_and_s_max_set_scc_when_src0_is_chosen() {
    let m1 = (-1i32) as u32;
    check_scc("s_min_i32", 0x06, &[(m1, 2, 1), (2, m1, 0), (4, 4, 0)]);
    check_scc("s_min_u32", 0x07, &[(3, 5, 1), (5, 3, 0), (4, 4, 0)]);
    check_scc("s_max_i32", 0x08, &[(2, m1, 1), (m1, 2, 0), (4, 4, 0)]);
    check_scc("s_max_u32", 0x09, &[(5, 3, 1), (3, 5, 0), (4, 4, 0)]);
}

#[test]
fn s_lshl_add_sets_scc_on_unsigned_overflow() {
    // (s0 << n) + s1 past 32 bits, through the shift or through the add.
    check_scc(
        "s_lshl1_add_u32",
        0x2E,
        &[
            (1, 2, 0),
            (1, 0xFFFF_FFFD, 0),
            (1, 0xFFFF_FFFE, 1),
            (0x8000_0000, 0, 1),
        ],
    );
    check_scc(
        "s_lshl4_add_u32",
        0x31,
        &[
            (0x0FFF_FFFF, 0xF, 0),
            (0x0FFF_FFFF, 0x10, 1),
            (0x1000_0000, 0, 1),
        ],
    );
}

#[test]
fn v_cmp_f32_nan_matches_rdna2() {
    let nan = f32::NAN.to_bits();
    let one = 1.0f32.to_bits();
    let two = 2.0f32.to_bits();
    let sel = vop3(0x101, 0, 0x80, 0x81, sgpr(6));

    // Self-comparisons (x, x): ordered predicates must be false on NaN and true
    // on finite values; unordered predicates must be true on NaN and false on
    // finite values (Intel Xe folds plain `fcmp` self-compares to constants).
    for (name, opc, want_nan, want_one) in [
        ("v_cmp_eq_f32", 0x002u16, 0u32, 1u32),
        ("v_cmp_le_f32", 0x003u16, 0u32, 1u32),
        ("v_cmp_ge_f32", 0x006u16, 0u32, 1u32),
        ("v_cmp_o_f32", 0x007u16, 0u32, 1u32),
        ("v_cmp_u_f32", 0x008u16, 1u32, 0u32),
        ("v_cmp_nge_f32", 0x009u16, 1u32, 0u32),
        ("v_cmp_nle_f32", 0x00cu16, 1u32, 0u32),
        ("v_cmp_neq_f32", 0x00du16, 1u32, 0u32),
    ] {
        let cmp = vop3(opc, 6, vgpr(1), vgpr(1), 0);
        let sh = build_valu_cs(&[cmp[0], cmp[1], sel[0], sel[1]]);
        run_cases(&sh, &[[nan, 0, 0, 0], [one, 0, 0, 0]], |backend, i, _, got| {
            let want = if i == 0 { want_nan } else { want_one };
            assert_eq!(
                got, want,
                "{backend}: {name}(same reg, case {i}) = {got}, want {want}",
            );
        });
    }

    // Distinct-register ordered/unordered comparisons: Intel Xe also folds
    // `fcmp ord`/`fcmp uno` on distinct registers when either operand is NaN.
    for (name, opc, want_nan, want_finite) in [
        ("v_cmp_o_f32", 0x007u16, 0u32, 1u32),
        ("v_cmp_u_f32", 0x008u16, 1u32, 0u32),
    ] {
        let cmp = vop3(opc, 6, vgpr(1), vgpr(2), 0);
        let sh = build_valu_cs(&[cmp[0], cmp[1], sel[0], sel[1]]);
        let inputs = [
            [nan, one, 0, 0],
            [one, nan, 0, 0],
            [nan, nan, 0, 0],
            [one, two, 0, 0],
        ];
        run_cases(&sh, &inputs, |backend, i, _, got| {
            let want = if i == 3 { want_finite } else { want_nan };
            assert_eq!(
                got, want,
                "{backend}: {name}(diff reg, case {i}) = {got}, want {want}",
            );
        });
    }
}

#[test]
fn root_uav_validates_and_dispatches_on_gpu_and_warp() {
    let s_load_dword = |dst: u8, base: u8, byte_off: u32| -> [u32; 2] {
        [
            0x3D << 26 | (dst as u32) << 6 | (base as u32 >> 1),
            125 << 25 | (byte_off & 0x1F_FFFF),
        ]
    };
    let buffer_load_dword = |vdst: u8, srsrc: u8, byte_off: u16| -> [u32; 2] {
        [
            0x38 << 26 | 0x0C << 18 | (byte_off as u32 & 0xFFF),
            0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdst as u32) << 8,
        ]
    };
    let buffer_atomic_add_rtn = |vdata: u8, srsrc: u8, byte_off: u16| -> [u32; 2] {
        [
            0x38 << 26 | 0x32 << 18 | (1 << 14) | (byte_off as u32 & 0xFFF),
            0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
        ]
    };
    let global_store_dword = |vaddr: u8, vdata: u8, saddr: u8, byte_off: u16| -> [u32; 2] {
        [
            0x37 << 26 | 0x1C << 18 | (2 << 14) | (byte_off as u32 & 0xFFF),
            (saddr as u32) << 16 | (vdata as u32) << 8 | vaddr as u32,
        ]
    };

    let mut words = Vec::new();
    // s[0:1] = Root CBV (slot 0), s[2:3] = Root SRV (slot 1), s[4:5] = Root UAV (slot 2).
    words.extend_from_slice(&s_load_dword(6, 0, 8)); // s6 = cbv[2]
    words.extend_from_slice(&s_load_dword(7, 2, 4)); // s7 = srv[1]
    // Build a raw buffer descriptor around Root UAV s[4:5] in s[8:11].
    words.push(s_mov_b32(8, 4));
    words.push(s_mov_b32(9, 5));
    words.push(s_mov_b32(10, 0xC1)); // -1
    words.extend_from_slice(&[s_mov_b32(11, 0xFF), 0x3101_4FAC]);
    // uav[0] = s6 via global_store_dword.
    words.push(vop1(0x001, 0, 0x80)); // v0 = 0
    words.push(vop1(0x001, 1, sgpr(6))); // v1 = s6
    words.extend_from_slice(&global_store_dword(0, 1, 4, 0));
    // uav[1] = s7 via buffer_store_dword, then atomic_add 17 (old value -> v3).
    words.push(vop1(0x001, 2, sgpr(7))); // v2 = s7
    words.extend_from_slice(&buffer_store_dword(2, 8, 4));
    words.push(vop1(0x001, 3, 128 + 17)); // v3 = 17
    words.extend_from_slice(&buffer_atomic_add_rtn(3, 8, 4));
    // v4 = uav[0] via buffer_load_dword; uav[2] = v3 ^ v4.
    words.extend_from_slice(&buffer_load_dword(4, 8, 0));
    let vop2 = |op: u8, vdst: u8, src0: u16, vsrc1: u8| -> u32 {
        (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
    };
    words.push(vop2(0x1D, 5, vgpr(3), 4)); // v_xor_b32 v5, v3, v4
    words.extend_from_slice(&buffer_store_dword(5, 8, 8));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
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
        used: vec![
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
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 6 << 1,
            threads: [1, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let container = crate::dxbc::build(&[]);
    let c = crate::Container::parse(&container).expect("container");
    let shader = super::emit(&c, &native, &p, &st, &io, &res).expect("emit");

    assert_eq!(shader.root_cbvs, vec![0]);
    assert_eq!(shader.root_srvs, vec![1]);
    assert_eq!(shader.root_uavs, vec![2]);

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        // Root parameter 0 is user data; sorted root views follow at 1 (CBV 0), 2 (SRV 1), 3 (UAV 2).
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(vec![0; 4])),
                        (
                            1,
                            Root::View {
                                srv: false,
                                data: vec![0x1111_1111, 0x2222_2222, 0xCAFE_BEEF, 0],
                            },
                        ),
                        (
                            2,
                            Root::View {
                                srv: true,
                                data: vec![0, 0x1000, 0, 0],
                            },
                        ),
                        (3, Root::Uav(4)),
                    ],
                    views: &[],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(
            &out[0][..3],
            &[0xCAFE_BEEF, 0x1011, 0x1000 ^ 0xCAFE_BEEF],
            "{label}: unexpected Root UAV readback"
        );
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn root_uav_graphics_stages_validate_and_build_pso() {
    use super::bitcode::{Attr, Term};
    use super::container::SigElement;
    use super::shader::Shader;

    let make_stage = |pixel: bool, slots: &[u8]| -> HostShader {
        let mut s = Shader::new(pixel);
        let b = s.entry;
        let iu = s.m.undef(s.t.i32);
        for (k, &slot) in slots.iter().enumerate() {
            let h = s.root_uav_handle(slot);
            let args = [
                h,
                s.m.i32((k * 4) as u32),
                iu,
                s.m.i32(0x1234_0000 + u32::from(slot)),
                iu,
                iu,
                iu,
                s.m.i8(1),
                s.m.i32(4),
            ];
            s.op(
                b,
                "rawBufferStore.i32",
                Attr::NoUnwind,
                s.t.void,
                140,
                &args,
            );
        }
        let out = SigElement {
            name: if pixel { "SV_Target" } else { "SV_Position" }.into(),
            index: 0,
            kind: if pixel { 16 } else { 3 },
            comp: 9,
            interp: if pixel { 0 } else { 4 },
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        };
        for col in 0..4u8 {
            let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), s.m.f32(0)];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.outputs.push(out);
        if pixel {
            s.inputs.push(SigElement {
                name: "SV_Position".into(),
                index: 0,
                kind: 3,
                comp: 9,
                interp: 4,
                row: Some(0),
                col: 0,
                cols: 4,
                usage: 0,
                half: false,
            });
        }
        s.m.terminate(b, Term::Ret);
        HostShader {
            stage: if pixel { HwStage::Ps } else { HwStage::Vs },
            dxil: s.finish(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            stride_table: false,
            root_uavs: slots.to_vec(),
        }
    };

    let vs = make_stage(false, &[2]);
    let ps = make_stage(true, &[2, 3]);

    // Vertex shaders writing UAVs must stamp SFI0 UAVsAtEveryStage (0x4).
    let vs_c = crate::Container::parse(&vs.dxil).unwrap();
    let sfi0 = u64::from_le_bytes(
        vs_c.part(crate::FourCC::SFI0)
            .unwrap()
            .data
            .try_into()
            .unwrap(),
    );
    assert_ne!(
        sfi0 & 0x4,
        0,
        "VS with Root UAV must set SFI0 UAVsAtEveryStage"
    );

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        for (name, sh) in [("vs", &vs), ("ps", &ps)] {
            let mut copy = sh.dxil.clone();
            let ours = copy[4..20].to_vec();
            v.validate(&mut copy)
                .unwrap_or_else(|e| panic!("{name}: DXIL validation failed: {e}"));
            assert_eq!(copy[4..20], ours[..], "{name}: container hash mismatch");
        }
    }

    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        host.pipeline(&vs, &ps).unwrap_or_else(|e| {
            panic!("{label}: graphics pipeline with shared Root UAV failed: {e}")
        });
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

fn validate_dxil(dxil: &[u8]) {
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = dxil.to_vec();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }
}

/// Builds a synthetic compute shader with explicit thread-group size, optional
/// 512-byte LDS allocation, custom `loop_limit`, and an `out_dwords` raw UAV
/// at heap index 0 (`s[8:11]` loaded from table slot 1).
fn build_custom_cs(threads: u32, use_lds: bool, loop_limit: u32, body: &[u32]) -> HostShader {
    let mut words = Vec::new();
    words.push(s_mov_b32(5, 0x80));
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.extend_from_slice(body);
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let rsrc2 = (5 << 1) | if use_lds { 1 << 15 } else { 0 };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2,
            threads: [threads, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower_with_limit(
        &lower::Parts {
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
        },
        loop_limit,
    )
    .expect("lower_with_limit");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        stride_table: lowered.stride_table,
        root_uavs: lowered.root_uavs,
    };
    validate_dxil(&shader.dxil);
    shader
}

const fn vopc(op: u8, src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | (op as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn sopp(op: u8, imm: i16) -> u32 {
    0xBF80_0000 | (op as u32) << 16 | (imm as u16 as u32)
}

const fn ds_op(op: u8, vdst: u8, vaddr: u8, vdata0: u8, offset: u16) -> [u32; 2] {
    [
        0x36 << 26 | (op as u32) << 18 | (offset as u32),
        (vdst as u32) << 24 | (vdata0 as u32) << 8 | (vaddr as u32),
    ]
}

const fn buffer_store_dword_offen(vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (1 << 12),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | (vaddr as u32),
    ]
}

/// A minimal full-screen-triangle vertex shader (`float2 POSITION0 -> SV_Position`)
/// for running pixel shaders with [`Host::draw`].
fn fullscreen_vs() -> HostShader {
    use super::bitcode::{Attr, Term};
    use super::container::SigElement;
    use super::shader::Shader;

    let mut s = Shader::new(false);
    s.inputs.push(SigElement {
        name: "POSITION".into(),
        index: 0,
        kind: 0,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 2,
        usage: 0b0011,
        half: false,
    });
    s.outputs.push(SigElement {
        name: "SV_Position".into(),
        index: 0,
        kind: 3,
        comp: 9,
        interp: 4,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0b1111,
        half: false,
    });
    let b = s.entry;
    let undef = s.m.undef(s.t.i32);
    let (z0, c0, c1) = (s.m.i32(0), s.m.i8(0), s.m.i8(1));
    let x = s.op(
        b,
        "loadInput.f32",
        Attr::ReadNone,
        s.t.f32,
        4,
        &[z0, z0, c0, undef],
    );
    let y = s.op(
        b,
        "loadInput.f32",
        Attr::ReadNone,
        s.t.f32,
        4,
        &[z0, z0, c1, undef],
    );
    let z = s.m.f32(0.5f32.to_bits());
    let w = s.m.f32(1.0f32.to_bits());
    for (col, val) in [x, y, z, w].into_iter().enumerate() {
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col as u8), val];
        s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
    }
    s.m.terminate(b, Term::Ret);
    let dxil = s.finish();
    validate_dxil(&dxil);
    HostShader {
        stage: HwStage::Vs,
        dxil,
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        stride_table: false,
        root_uavs: Vec::new(),
    }
}

#[test]
fn s_wqm_pixel_loop_and_skip_run_per_quad() {
    use super::container::SigElement;
    use super::host::Draw;

    // Pixel shader over a 4x2 target (`v0 = SV_Position.x`):
    // - Only `x == 0` starts with counter `v3 = 3` (all other pixels start at `0`).
    // - `s_wqm_b32 exec_lo, vcc_lo` + `s_cbranch_execz` skips quads where no
    //   lane has `x == 0` (the right 2x2 quad `x = 2..3` must output `0.0`).
    // - Inside the left 2x2 quad (`x = 0..1`), all 4 pixels stay in the loop
    //   via `s_wqm_b32` + `s_cbranch_execnz` while `x == 0` counts down `3..0`,
    //   exiting cleanly at `3.0` instead of spinning to `loop_limit = 16`.
    let words: &[u32] = &[
        vop1(0x007, 1, vgpr(0)), //  0: v_cvt_u32_f32 v1, v0
        vop1(0x001, 2, 0x80),    //  4: v_mov_b32 v2, 0
        vopc(0xC2, 0x80, 1),     //  8: v_cmp_eq_u32 vcc_lo, 0, v1
        vop1(0x001, 3, 0x83),    // 12: v_mov_b32 v3, 3
        vop2(0x01, 3, 0x80, 3),  // 16: v_cndmask_b32 v3, 0, v3, vcc_lo
        0xBE82_037E,             // 20: s_mov_b32 s2, exec_lo
        0xBEFE_096A,             // 24: s_wqm_b32 exec_lo, vcc_lo
        sopp(0x08, 7),           // 28: s_cbranch_execz -> 60
        // loop (offset 32):
        vop2(0x25, 2, 0x81, 2), // 32: v_add_nc_u32 v2, 1, v2
        vopc(0xC1, 0x80, 3),    // 36: v_cmp_lt_u32 vcc_lo, 0, v3
        vop2(0x25, 4, 0xC1, 3), // 40: v_add_nc_u32 v4, -1, v3
        vop2(0x01, 3, 0x80, 4), // 44: v_cndmask_b32 v3, 0, v4, vcc_lo
        vopc(0xC1, 0x80, 3),    // 48: v_cmp_lt_u32 vcc_lo, 0, v3
        0xBEFE_096A,            // 52: s_wqm_b32 exec_lo, vcc_lo
        sopp(0x09, -7),         // 56: s_cbranch_execnz -> 32
        // done (offset 60):
        0xBEFE_0302,             // 60: s_mov_b32 exec_lo, s2
        vop1(0x006, 2, vgpr(2)), // 64: v_cvt_f32_u32 v2, v2
        0xF800_180F,             // 68: exp mrt0 v2, v2, v2, v2 done vm
        0x0202_0202,
        S_ENDPGM, // 76: s_endpgm
    ];

    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 1 << 8, // Position.x in v0
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower_with_limit(
        &lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![SigElement {
                name: "SV_Position".into(),
                index: 0,
                kind: 3,
                comp: 9,
                interp: 4,
                row: Some(0),
                col: 0,
                cols: 4,
                usage: 0,
                half: false,
            }],
            outputs: vec![SigElement {
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
            }],
            prims: Vec::new(),
            mesh_stride: 0,
            options: super::TranslateOptions::default(),
        },
        16,
    )
    .expect("lower_with_limit");
    validate_dxil(&lowered.dxil);

    let ps = HostShader {
        stage: HwStage::Ps,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        stride_table: false,
        root_uavs: Vec::new(),
    };
    let vs = fullscreen_vs();
    let vertices = [-1.0f32, -1.0, 3.0, -1.0, -1.0, 3.0];

    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host.pipeline(&vs, &ps).expect("graphics pipeline");
        let pixels = host
            .draw(
                &root,
                &pso,
                &Draw {
                    roots: vec![(0, Root::Constants(vec![0])), (1, Root::Constants(vec![0]))],
                    textures: &[],
                    samplers: &[],
                    vertices: &vertices,
                    stride: 8,
                    size: (4, 2),
                    mesh_groups: 0,
                },
            )
            .expect("draw");
        assert_eq!(pixels.len(), 8);
        for y in 0..2usize {
            for x in 0..4usize {
                let want = if x < 2 { 3.0 } else { 0.0 };
                let got = pixels[y * 4 + x][0];
                assert_eq!(got, want, "{label}: pixel ({x}, {y}) = {got}, want {want}");
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

#[test]
fn oob_lds_reads_writes_and_atomics_clamp_and_drop() {
    // 64 threads, 512 bytes (128 dwords) of LDS (`s.wave_size == None`, so both
    // hardware and WARP run it):
    // 1. Write `lds[tid] = tid + 100` for `tid < 64`, then `s_barrier`.
    // 2. Under a `tid == 0` guard (`v_cmpx`), lanes `1..63` hold an OOB byte
    //    address (`1024`, dword `256 >= 128`) while lane 0 reads `lds[0]`:
    //    inactive lanes keep their sentinel `77`, lane 0 reads `100`.
    // 3. Every lane performs an active OOB `ds_write_b32`, OOB `ds_add_rtn_u32`,
    //    and OOB `ds_read_b32` at byte offset `512` (dword `128 >= 128`):
    //    both OOB reads/atomics return `0`, and `lds[0..64]` stays intact.
    let mut body = vec![
        vop2(0x1A, 1, 0x82, 0),    // v1 = tid * 4 (in-bounds byte offset 0..252)
        vop2(0x25, 2, sgpr(0), 0), // v2 = tid + 100 (`s0 = 100` from root constants)
    ];
    body.extend_from_slice(&ds_op(0x0D, 0, 1, 2, 0)); // ds_write_b32 v1, v2
    body.extend([
        sopp(0x0A, 0),             // s_barrier
        vop1(0x001, 3, sgpr(1)),   // v3 = 77 (`s1 = 77`)
        vop1(0x001, 4, sgpr(2)),   // v4 = 1024 (`s2 = 1024`, OOB)
        vopc(0xC2, 0x80, 0),       // v_cmp_eq_u32 vcc_lo, 0, v0
        vop1(0x001, 5, 0x80),      // v_mov_b32 v5, 0
        vop2(0x01, 4, sgpr(2), 5), // v_cndmask_b32 v4, s2, v5, vcc_lo (`tid == 0 ? 0 : 1024`)
        0xBE86_3C6A,               // s_and_saveexec_b32 s6, vcc_lo
    ]);
    body.extend_from_slice(&ds_op(0x36, 3, 4, 0, 0)); // ds_read_b32 v3, v4
    body.extend([
        0x887E_067E,             // s_or_b32 exec_lo, exec_lo, s6
        vop1(0x001, 6, sgpr(3)), // v6 = s3 (999)
    ]);
    // Active OOB write, OOB atomic_add_rtn, and OOB read at offset 512 (dword 128 == lds_dwords).
    body.extend_from_slice(&ds_op(0x0D, 0, 5, 6, 512)); // ds_write_b32 v5(0), v6 offset:512
    body.extend_from_slice(&ds_op(0x20, 7, 5, 6, 516)); // ds_add_rtn_u32 v7, v5(0), v6 offset:516
    body.extend_from_slice(&ds_op(0x36, 8, 5, 0, 512)); // ds_read_b32 v8, v5(0) offset:512
    body.push(sopp(0x0A, 0)); // s_barrier

    // Verify in-bounds `lds[tid]` was not corrupted: `v9 = ds_read_b32(v1)`.
    body.extend_from_slice(&ds_op(0x36, 9, 1, 0, 0));
    // Pack `(v3 << 16) | v9` + `v7` + `v8` into `uav[tid]`:
    body.extend([
        vop2(0x1A, 3, 0x90, 3),    // v_lshlrev_b32 v3, 16, v3
        vop2(0x1C, 9, vgpr(3), 9), // v_or_b32 v9, v3, v9
        vop2(0x25, 9, vgpr(7), 9), // v_add_nc_u32 v9, v7, v9
        vop2(0x25, 9, vgpr(8), 9), // v_add_nc_u32 v9, v8, v9
    ]);
    body.extend_from_slice(&buffer_store_dword_offen(9, 1, 8));

    let shader = build_custom_cs(64, true, 1024, &body);
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host.compute_pipeline(&shader).expect("compute_pipeline");
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let inputs = [100u32, 77, 1024, 999];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Constants, 0) => inputs[u.dword as usize],
                (RootType::Table, 1) => 0,
                (RootType::ResourceHeap, _) => 1,
                (RootType::SamplerHeap, _) => 0,
                other => panic!("unexpected {other:?}"),
            };
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch");
        for tid in 0..64u32 {
            let want_guarded = if tid == 0 { 100 } else { 77 };
            let want = (want_guarded << 16) | (100 + tid);
            assert_eq!(
                out[0][tid as usize], want,
                "{label}: lane {tid} got {:#010x}, want {want:#010x}",
                out[0][tid as usize]
            );
        }
        ran += 1;
    }
    assert!(ran > 0);
}

#[test]
fn nested_loops_share_invocation_step_budget() {
    // Two nested loops whose conditions always hold (`s0 == 0` -> `scc1` taken):
    // with per-loop reset counters and `limit = 25`, the inner body would run
    // `25 * 25 = 625` times; with a function-wide step budget of `25`, total
    // inner-body executions are capped at `25`.
    let body: &[u32] = &[
        vop1(0x001, 0, 0x80), //  0: v_mov_b32 v0, 0
        // outer (offset 4):
        // inner (offset 4):
        vop2(0x25, 0, 0x81, 0), //  4: v_add_nc_u32 v0, 1, v0
        0xBF06_8000,            //  8: s_cmp_eq_u32 s0, 0
        sopp(0x05, -3),         // 12: s_cbranch_scc1 -> 4 (inner back-edge)
        0xBF06_8000,            // 16: s_cmp_eq_u32 s0, 0
        sopp(0x05, -5),         // 20: s_cbranch_scc1 -> 4 (outer back-edge)
        buffer_store_dword(0, 8, 0)[0],
        buffer_store_dword(0, 8, 0)[1],
    ];
    let shader = build_custom_cs(1, false, 25, body);
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host.compute_pipeline(&shader).expect("compute_pipeline");
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = 0;
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch");
        assert!(
            out[0][0] <= 25 && out[0][0] >= 24,
            "{label}: nested loops ran {} iterations, expected ~25 (not 625)",
            out[0][0]
        );
        ran += 1;
    }
    assert!(ran > 0);
}

#[test]
fn divergent_loop_and_barrier_branches_do_not_deadlock() {
    // Part 1 (64 threads -> multi-wave on both Wave32 hardware and Wave4 WARP,
    // `loop_limit = 40`):
    // - First, a per-lane divergent loop runs `tid / 2` steps (`0..31` steps),
    //   leaving every lane/wave with a different `steps[0]` count.
    // - Next, an infinite loop containing `s_barrier` runs until `loop_limit`:
    //   because barrier loops use their own group-uniform `steps[1]` budget,
    //   all waves exit on the exact same iteration (`40`) without deadlocking!
    let mut multi_wave = vec![
        vop2(0x16, 1, 0x81, 0), // v1 = tid >> 1 (0..31)
        vop1(0x001, 2, 0x80),   // v2 = 0 (divergent counter)
        0xBE86_037E,            // s6 = saved full exec_lo
        vopc(0xC1, vgpr(2), 1), // v_cmp_lt_u32 vcc_lo, v2, v1
        0xBEFE_036A,            // s_mov_b32 exec_lo, vcc_lo
        sopp(0x08, 4),          // s_cbranch_execz -> after_div
        // div_loop:
        vop2(0x25, 2, 0x81, 2), // v2 += 1
        vopc(0xC1, vgpr(2), 1), // v_cmp_lt_u32 vcc_lo, v2, v1
        0x877E_6A7E,            // s_and_b32 exec_lo, exec_lo, vcc_lo
        sopp(0x09, -4),         // s_cbranch_execnz -> div_loop
        // after_div:
        0xBEFE_0306,          // s_mov_b32 exec_lo, s6
        vop1(0x001, 3, 0x80), // v3 = 0 (barrier loop counter)
        // bar_loop:
        vop2(0x25, 3, 0x81, 3), // v3 += 1
        sopp(0x0A, 0),          // s_barrier
        0xBF06_8000,            // s_cmp_eq_u32 s0, 0 (always true)
        sopp(0x05, -4),         // s_cbranch_scc1 -> bar_loop
        // Write `(v2 << 16) | v3` to `uav[tid]`:
        vop2(0x1A, 4, 0x82, 0),    // v4 = tid * 4
        vop2(0x1A, 2, 0x90, 2),    // v2 <<= 16
        vop2(0x1C, 3, vgpr(2), 3), // v3 |= v2
    ];
    multi_wave.extend_from_slice(&buffer_store_dword_offen(3, 4, 8));

    let mw_shader = build_custom_cs(64, true, 40, &multi_wave);
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host.compute_pipeline(&mw_shader).expect("compute_pipeline");
        let ud = vec![0u32; mw_shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch");
        for tid in 0..64u32 {
            let want = ((tid >> 1) << 16) | 40;
            assert_eq!(
                out[0][tid as usize], want,
                "{label}: lane {tid} got {:#010x}, want {want:#010x}",
                out[0][tid as usize]
            );
        }
        ran += 1;
    }
    assert!(ran > 0);

    // Part 2 (single 32-lane wave): `if (tid == 0)` via `s_and_saveexec_b32` +
    // `s_cbranch_execz` enclosing both a `ds_write_b32` and an `s_barrier`.
    // Validates with `dxil.dll` and executes on Wave32-capable hardware without
    // deadlocking `dx.op.barrier` or letting lanes 1..31 clobber `lds[0]`.
    let mut single_wave = vec![
        vop1(0x001, 1, 0x80),      // v1 = 0 (byte offset 0)
        vop2(0x25, 2, sgpr(0), 0), // v2 = 777 + tid (lane 0 has 777, others != 777)
        vopc(0xC2, 0x80, 0),       // v_cmp_eq_u32 vcc_lo, 0, v0
        0xBE86_3C6A,               // s_and_saveexec_b32 s6, vcc_lo
        sopp(0x08, 3),             // s_cbranch_execz -> join (ds_write[2] + s_barrier[1])
    ];
    single_wave.extend_from_slice(&ds_op(0x0D, 0, 1, 2, 0)); // ds_write_b32 v1, v2
    single_wave.extend([
        sopp(0x0A, 0), // s_barrier
        0x887E_067E,   // join: s_or_b32 exec_lo, exec_lo, s6
    ]);
    single_wave.extend_from_slice(&ds_op(0x36, 3, 1, 0, 0)); // v3 = ds_read_b32(0)
    single_wave.push(vop2(0x1A, 4, 0x82, 0)); // v4 = tid * 4
    single_wave.extend_from_slice(&buffer_store_dword_offen(3, 4, 8));

    let sw_shader = build_custom_cs(32, true, 40, &single_wave);
    if let Ok(host) = Host::new(false)
        && let Ok((root, pso)) = host.compute_pipeline(&sw_shader)
    {
        let mut ud = vec![0u32; sw_shader.user_data_dwords().max(1) as usize];
        for u in &sw_shader.user_data {
            if (u.root, u.slot) == (RootType::Constants, 0) {
                ud[u.index as usize] = 777;
            }
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(32)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("single-wave barrier dispatch");
        for (tid, &got) in out[0][..32].iter().enumerate() {
            assert_eq!(got, 777, "lane {tid} read {got} instead of lane 0's 777");
        }
    }
}

fn build_wave32_cs(body: &[u32]) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.extend_from_slice(body);
    // v2 = v0 * 4 (per-lane byte offset into the UAV), store v1 at out[v0].
    words.push((0x01A << 25) | (2 << 17) | 0x82); // v_lshlrev_b32 v2, 2, v0
    words.extend_from_slice(&[
        0x38 << 26 | 0x1C << 18 | (1 << 12), // buffer_store_dword ... offen
        0x80 << 24 | ((8 >> 2) << 16) | (1 << 8) | 2,
    ]);
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [32, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower(&lower::Parts {
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
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        stride_table: lowered.stride_table,
        root_uavs: lowered.root_uavs,
    };

    validate_dxil(&shader.dxil);

    shader
}

fn dispatch_wave32(
    name: &str,
    shader: &HostShader,
    fixed_wave: bool,
    cases: &[([u32; 4], Vec<u32>)],
) {
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("{name}: no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(p) => p,
            // WARP does not accept every fixed WaveSize requirement.
            Err(e) if warp && fixed_wave => {
                eprintln!("{name}: skipped on {label} ({e})");
                continue;
            }
            Err(e) => panic!("{name} ({label}): compute_pipeline failed: {e}"),
        };
        for (i, (inputs, want)) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    (RootType::ResourceHeap, _) => 1,
                    (RootType::SamplerHeap, _) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(32)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{name} ({label}): dispatch failed: {e}"));
            assert_eq!(&out[0], want, "{name} ({label}): case {i} ({inputs:?})");
        }
        ran += 1;
    }
    assert!(
        ran > 0,
        "{name}: neither hardware nor WARP SM 6.6 device available"
    );
}

#[test]
fn lane_mask_phi_joins_execute_correctly() {
    // Branch join: s0 == 0 sets s2 = 0; otherwise s2 = (v0 < s1).
    let branch = build_wave32_cs(&[
        0xBF06_8000, // 0: s_cmp_eq_u32 s0, 0
        0xBF84_0002, // 4: s_cbranch_scc0 -> 16 (otherwise s2 = (v0 < s1))
        0xBE82_0380, // 8: s_mov_b32 s2, 0
        0xBF82_0002, // 12: s_branch -> 24
        0xD4C4_0002,
        u32::from(sgpr(1)) | (u32::from(vgpr(0)) << 9), // 16: v_cmp_gt_u32 s2, s1, v0 (v0 < s1)
        0xD501_0001,
        (0x8A) | (0x94 << 9) | (2 << 18), // 24: v_cndmask_b32 v1, 10, 20, s2
    ]);
    dispatch_wave32(
        "branch_zero_join",
        &branch,
        false,
        &[
            ([0, 16, 0, 0], vec![10; 32]),
            (
                [1, 16, 0, 0],
                (0..32).map(|t| if t < 16 { 20 } else { 10 }).collect(),
            ),
        ],
    );

    // Branch join with a partial constant mask: s0 == 0 sets s2 = 0xF; otherwise s2 = (v0 < s1).
    let partial_constant = build_wave32_cs(&[
        0xBF06_8000, // 0: s_cmp_eq_u32 s0, 0
        0xBF84_0002, // 4: s_cbranch_scc0 -> 16 (otherwise s2 = (v0 < s1))
        0xBE82_038F, // 8: s_mov_b32 s2, 15
        0xBF82_0002, // 12: s_branch -> 24
        0xD4C4_0002,
        u32::from(sgpr(1)) | (u32::from(vgpr(0)) << 9), // 16: v_cmp_gt_u32 s2, s1, v0 (v0 < s1)
        0xD501_0001,
        (0x8A) | (0x94 << 9) | (2 << 18), // 24: v_cndmask_b32 v1, 10, 20, s2
    ]);
    dispatch_wave32(
        "partial_constant_join",
        &partial_constant,
        true,
        &[
            (
                [0, 16, 0, 0],
                (0..32).map(|t| if t < 4 { 20 } else { 10 }).collect(),
            ),
            (
                [1, 16, 0, 0],
                (0..32).map(|t| if t < 16 { 20 } else { 10 }).collect(),
            ),
        ],
    );

    // Loop accumulator from 0: iterates i = 0..s0, reading phi s2 inside the
    // loop, accumulating `s2 |= (v0 == i)`, then reading final s2 after exit.
    // v1 = (inside last iter: 20 if v0 < s0 - 1 else 10) + (after loop: 30 if v0 < s0 else 10).
    let loop_shader = build_wave32_cs(&[
        0xBE82_0380, // 0: s_mov_b32 s2, 0
        0xBE86_0380, // 4: s_mov_b32 s6, 0
        0xD501_0001,
        (0x8A) | (0x94 << 9) | (2 << 18), // 8: v_cndmask_b32 v1, 10, 20, s2
        0x7D84_0006,                      // 16: v_cmp_eq_u32 vcc, s6, v0
        0x8802_026A,                      // 20: s_or_b32 s2, vcc_lo, s2
        0x8006_8106,                      // 24: s_add_u32 s6, s6, 1
        0xBF0A_0006,                      // 28: s_cmp_lt_u32 s6, s0
        0xBF85_FFF9,                      // 32: s_cbranch_scc1 -> 8
        0xD501_0003,
        (0x8A) | (0x9E << 9) | (2 << 18), // 36: v_cndmask_b32 v3, 10, 30, s2
        (0x025 << 25) | (1 << 17) | (3 << 9) | 257, // 44: v_add_nc_u32 v1, v1, v3
    ]);
    dispatch_wave32(
        "loop_accumulator",
        &loop_shader,
        false,
        &[(
            [5, 0, 0, 0],
            (0..32)
                .map(|t| {
                    if t < 4 {
                        50
                    } else if t == 4 {
                        40
                    } else {
                        20
                    }
                })
                .collect(),
        )],
    );
}

/// DS `op` on LDS address `addr` (+ `offset` bytes) with `data0`, result in
/// `vdst`.
const fn ds(op: u8, vdst: u8, addr: u8, data0: u8, offset: u16) -> [u32; 2] {
    [
        0x36 << 26 | (op as u32) << 18 | offset as u32,
        (vdst as u32) << 24 | (data0 as u32) << 8 | addr as u32,
    ]
}

/// LDS programs here are one wave: wave 32, which WARP cannot run.
const HARDWARE: &[(&str, bool)] = &[("hardware", false)];

const DS_WRITE_B32: u8 = 0x0d;
const DS_WRITE_B8: u8 = 0x1e;
const DS_WRITE_B16: u8 = 0x1f;
const DS_READ_B32: u8 = 0x36;
const DS_READ_I8: u8 = 0x39;
const DS_READ_U8: u8 = 0x3a;
const DS_READ_I16: u8 = 0x3b;
const DS_READ_U16: u8 = 0x3c;
const DS_WRITE_B8_D16_HI: u8 = 0xa0;
const DS_READ_U16_D16: u8 = 0xa6;
const DS_READ_U16_D16_HI: u8 = 0xa7;

/// `v5 = 0` (the LDS address), then `body`.
fn lds_cs(body: &[u32]) -> HostShader {
    let mut words = vec![vop1(0x001, 5, 128)]; // v_mov_b32 v5, 0
    words.extend_from_slice(body);
    build_cs(&words, 1)
}

#[test]
fn lds_byte_and_half_writes_keep_the_rest_of_the_dword() {
    // [0] = s0; byte 1 = s1; half 2 = s2; byte 0 = high half's byte of s3.
    let shader = lds_cs(
        &[
            ds(DS_WRITE_B32, 0, 5, 1, 0),
            ds(DS_WRITE_B8, 0, 5, 2, 1),
            ds(DS_WRITE_B16, 0, 5, 3, 2),
            ds(DS_WRITE_B8_D16_HI, 0, 5, 4, 0),
            ds(DS_READ_B32, 0, 5, 0, 0),
        ]
        .concat(),
    );
    let cases = [
        [0x1122_3344, 0xAAAA_AA55, 0xBBBB_6677, 0x00EE_0000],
        [0xFFFF_FFFF, 0, 0, 0],
        [0, 0xFF, 0xFFFF, 0x00FF_0000],
    ];
    run_cases_on(
        &shader,
        &cases,
        HARDWARE,
        |backend, _, [s0, s1, s2, s3], got| {
            // Every byte of s0 is overwritten.
            let want = (s3 >> 16 & 0xFF) | (s1 & 0xFF) << 8 | (s2 & 0xFFFF) << 16;
            assert_eq!(got, want, "{backend}: {s0:#x} {s1:#x} {s2:#x} {s3:#x}");
        },
    );
}

#[test]
fn lds_byte_and_half_reads_extend_as_the_console() {
    // [0] = s0, then each read of it.
    let reads: [(u8, u16, fn(u32) -> u32); 6] = [
        (DS_READ_U8, 1, |w| w >> 8 & 0xFF),
        (DS_READ_I8, 1, |w| (w >> 8) as u8 as i8 as i32 as u32),
        (DS_READ_U16, 2, |w| w >> 16),
        (DS_READ_I16, 2, |w| (w >> 16) as u16 as i16 as i32 as u32),
        // d16: v0 starts as v1 (s0); the other half is kept.
        (DS_READ_U16_D16, 2, |w| w & 0xFFFF_0000 | w >> 16),
        (DS_READ_U16_D16_HI, 0, |w| w & 0xFFFF | w << 16),
    ];
    for (op, offset, want) in reads {
        let mut body = ds(DS_WRITE_B32, 0, 5, 1, 0).to_vec();
        body.push(vop1(0x001, 0, vgpr(1))); // v_mov_b32 v0, v1
        body.extend(ds(op, 0, 5, 0, offset));
        let shader = lds_cs(&body);
        let cases = [[0x8081_7F42, 0, 0, 0], [0x1234_F00D, 0, 0, 0], [0, 0, 0, 0]];
        run_cases_on(&shader, &cases, HARDWARE, |backend, _, [s0, ..], got| {
            assert_eq!(got, want(s0), "{backend}: op {op:#x} of {s0:#010x}");
        });
    }
}

#[test]
fn buffer_store_short_writes_only_the_low_halfword() {
    let body = [vop1(0x001, 0, vgpr(1))]; // v0 = s0
    let shader = build_valu_cs_with_store(&body, buffer_store_short(0, 8, 0, false));
    let cases = &[
        [0x1234_5678, 0, 0, 0],
        [0xdead_abcd, 0, 0, 0],
        [0xffff_0001, 0, 0, 0],
    ];
    run_cases(&shader, cases, |backend, i, inputs, result| {
        assert_eq!(
            result & 0xffff,
            inputs[0] & 0xffff,
            "{backend} case {i}: buffer_store_short must write the low 16 bits"
        );
    });
}

#[test]
fn buffer_store_short_d16_hi_writes_the_high_halfword() {
    let body = [vop1(0x001, 0, vgpr(1))]; // v0 = s0
    // Write the high half of v0 to byte offset 2; the readback checks the
    // upper halfword of the first output dword.
    let shader = build_valu_cs_with_store(&body, buffer_store_short(0, 8, 2, true));
    let cases = &[
        [0x1234_5678, 0, 0, 0],
        [0xdead_abcd, 0, 0, 0],
        [0xffff_0001, 0, 0, 0],
    ];
    run_cases(&shader, cases, |backend, i, inputs, result| {
        assert_eq!(
            (result >> 16) & 0xffff,
            inputs[0] >> 16,
            "{backend} case {i}: buffer_store_short_d16_hi must write the high 16 bits"
        );
    });
}

/// VOP3P `op vdst, src0, src1, src2` with its packed modifiers: `op_sel` and
/// `op_sel_hi` pick each source's half for the low and high lanes of the
/// operation (bit k for source k), `neg_lo`/`neg_hi` negate them.
#[allow(clippy::too_many_arguments)]
const fn vop3p_mods(
    op: u8,
    vdst: u8,
    [src0, src1, src2]: [u16; 3],
    op_sel: u8,
    op_sel_hi: u8,
    neg_lo: u8,
    neg_hi: u8,
    clamp: bool,
) -> [u32; 2] {
    [
        0x33 << 26
            | (op as u32 & 0x7F) << 16
            | (clamp as u32) << 15
            | (op_sel_hi as u32 >> 2 & 1) << 14
            | (op_sel as u32 & 7) << 11
            | (neg_hi as u32 & 7) << 8
            | vdst as u32,
        (src0 as u32 & 0x1FF)
            | (src1 as u32 & 0x1FF) << 9
            | (src2 as u32 & 0x1FF) << 18
            | (op_sel_hi as u32 & 3) << 27
            | (neg_lo as u32 & 7) << 29,
    ]
}

/// Inputs `[src0, src1, src2, _]` and the f32 bits RDNA2 gives for
/// `src0.f16[0] * src1.f16[0] + src0.f16[1] * src1.f16[1] + src2.f32`: f16
/// products are exact in f32, so 65504 * 65504 does not overflow as it would
/// in f16.
const DOT2_F32_F16_CASES: &[([u32; 4], u32)] = &[
    // (1, 2) . (3, 4) + 0.5 = 11.5
    ([0x4000_3c00, 0x4400_4200, 0x3f00_0000, 0], 0x4138_0000),
    // (-1.5, 0.25) . (2, -8) - 1 = -6
    ([0x3400_be00, 0xc800_4000, 0xbf80_0000, 0], 0xc0c0_0000),
    // (65504, 65504) . (65504, 0) + 0 = 4290774016
    ([0x7bff_7bff, 0x0000_7bff, 0, 0], 0x4f7f_c004),
    // (inf, 0) . (1, 0) + 0 = inf
    ([0x0000_7c00, 0x0000_3c00, 0, 0], 0x7f80_0000),
];

#[test]
fn v_dot2_f32_f16_sums_f16_products_in_f32() {
    // v0 = v_dot2_f32_f16 v1, v2, v3 (no modifiers: op_sel_hi = 0b111)
    let body = vop3p_mods(0x13, 0, [vgpr(1), vgpr(2), vgpr(3)], 0, 0b111, 0, 0, false);
    let shader = build_valu_cs(&body);
    let inputs: Vec<[u32; 4]> = DOT2_F32_F16_CASES.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = DOT2_F32_F16_CASES[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_dot2_f32_f16({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_dot2_f32_f16_applies_op_sel_and_negation_per_half() {
    // src0's halves swapped (op_sel bit 0 set, op_sel_hi bit 0 clear), src1's
    // low half negated (neg_lo bit 1), src0's high-lane half negated (neg_hi
    // bit 0): (1, 2) . (3, 4) + 0.5 becomes 2 * -3 + -1 * 4 + 0.5 = -9.5.
    let body = vop3p_mods(
        0x13,
        0,
        [vgpr(1), vgpr(2), vgpr(3)],
        0b001,
        0b110,
        0b010,
        0b001,
        false,
    );
    let shader = build_valu_cs(&body);
    let cases: &[([u32; 4], u32)] = &[
        ([0x4000_3c00, 0x4400_4200, 0x3f00_0000, 0], 0xc118_0000),
        // (-1.5, 0.25) . (2, -8) - 1 with the same modifiers:
        // 0.25 * -2 + 1.5 * -8 - 1 = -13.5
        ([0x3400_be00, 0xc800_4000, 0xbf80_0000, 0], 0xc158_0000),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_dot2_f32_f16 with modifiers({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_dot2_f32_f16_clamps_to_zero_one() {
    let body = vop3p_mods(0x13, 0, [vgpr(1), vgpr(2), vgpr(3)], 0, 0b111, 0, 0, true);
    let shader = build_valu_cs(&body);
    let cases: &[([u32; 4], u32)] = &[
        // 11.5 clamps to 1.0, -6 to 0.0
        ([0x4000_3c00, 0x4400_4200, 0x3f00_0000, 0], 0x3f80_0000),
        ([0x3400_be00, 0xc800_4000, 0xbf80_0000, 0], 0),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_dot2_f32_f16 clamp({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_dot2c_f32_f16_accumulates_into_its_destination() {
    // v3 = v_dot2c_f32_f16 v1, v2 (v3 += v1 . v2), then v0 = v3.
    let body = [vop2(0x02, 3, vgpr(1), 2), vop1(0x001, 0, vgpr(3))];
    let shader = build_valu_cs(&body);
    let inputs: Vec<[u32; 4]> = DOT2_F32_F16_CASES.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = DOT2_F32_F16_CASES[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_dot2c_f32_f16({:#010x}, {:#010x}) + {:#010x} = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

/// A descriptor at a heap address the program's data computes (the table
/// address plus an index, as bindless code adds one it loads). Stale data
/// gives any index; one past the host heap is undefined: NVIDIA wraps it
/// onto another slot (without the guard, index 2^25 writes through slot 0
/// here), and a descriptor of another kind there hangs the GPU. Such an
/// index reads the heap's guard slot (null) instead: the store is dropped.
/// In range, the store lands.
#[test]
fn a_heap_index_past_the_heap_reads_the_guard_slot() {
    let words = [
        s_mov_b32(5, 0x80),     // s5 = 0
        sop2(0x1E, 6, 0, 0x85), // s_lshl_b32 s6, s0, 5: the index from data
        // s_load_dwordx4 s[8:11], s[4:5], s6
        s_load_dwordx4(8, 4)[0],
        6 << 25,
        vop1(0x001, 0, 0x87), // v_mov_b32 v0, 7
        buffer_store_dword(0, 8, 0)[0],
        buffer_store_dword(0, 8, 0)[1],
        S_ENDPGM,
    ];
    let shader = lower_cs(&words);
    // The guard is a user-data dword of its own, past the console's.
    let guard: Vec<_> = shader
        .user_data
        .iter()
        .filter(|u| u.root == RootType::ResourceHeap)
        .collect();
    assert_eq!(guard.len(), 1, "{:?}", shader.user_data);
    assert_eq!(guard[0].index, 5, "after s0..s4");
    let cases = [
        [0, 0, 0, 0],           // heap index 0: the UAV
        [1, 0, 0, 0],           // the guard slot itself
        [1 << 25, 0, 0, 0],     // far past the heap
        [0x07FF_FFFF, 0, 0, 0], // the last index an address gives
    ];
    run_cases(&shader, &cases, |label, i, k, got| {
        let want = if i == 0 { 7 } else { 0 };
        assert_eq!(got, want, "{label}: index {:#x}", k[0]);
    });
}

/// MUBUF `buffer_load_dword vdata, vaddr, s[srsrc:srsrc+3], 0 [idxen] [offen]`.
const fn buffer_load_mubuf(
    op: u32,
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    idxen: bool,
    offen: bool,
) -> [u32; 2] {
    [
        0x38 << 26 | op << 18 | (idxen as u32) << 13 | (offen as u32) << 12,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

fn build_oob_buffer_cs(
    stride: u32,
    load_op: u32,
    idxen: bool,
    offen: bool,
    dwords: u8,
) -> HostShader {
    let mut words = Vec::new();
    // s0 = root constant (index or byte offset); s[2:3] = descriptor table base.
    words.push(s_mov_b32(3, 0x80)); // s3 = 0
    let input = s_load_dwordx4(4, 2); // heap slot 0 -> s[4:7]
    let mut output = s_load_dwordx4(8, 2); // heap slot 1 -> s[8:11]
    output[1] |= 32;
    words.extend_from_slice(&input);
    words.extend_from_slice(&output);
    words.push(vop1(0x001, 1, sgpr(0))); // v1 = s0
    words.extend_from_slice(&buffer_load_mubuf(load_op, 4, 1, 4, idxen, offen));
    for k in 0..dwords {
        words.extend_from_slice(&buffer_store_dword(4 + k, 8, u16::from(k) * 4));
    }
    words.push(S_ENDPGM);

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 12,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 2,
                slot: 1,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride,
                dwords: 4,
                user_data: 2,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 2,
                slot: 1,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1, // 3 user SGPRs: s0..s2
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    lower_compute(&native)
}

fn dispatch_oob_case(
    shader: &HostShader,
    view_stride: Stride,
    idx_or_off: u32,
    out_dwords: u32,
) -> Vec<u32> {
    let data = [
        0xCAFE_BABE,
        0x1111_2222,
        0x3333_4444,
        0x5555_6666,
        0x7777_8888,
        0x9999_AAAA,
        0xBBBB_CCCC,
        0xDDDD_EEEE,
    ];
    let host = Host::new(false).expect("hardware SM 6.6 device");
    let (root, pso) = host.compute_pipeline(shader).expect("compute_pipeline");
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match (u.root, u.slot) {
            (RootType::Constants, 0) => idx_or_off,
            (RootType::Table, 1) => 0,
            (RootType::ResourceHeap, _) | (RootType::SamplerHeap, _) => {
                super::host::guard(u.root, 2, 0)
                    .unwrap_or_else(|| panic!("unexpected user data {:?}", (u.root, u.slot)))
            }
            other => panic!("unexpected user data {other:?}"),
        };
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[
                    View::Buffer {
                        data: &data,
                        stride: view_stride,
                    },
                    View::RawUav(out_dwords),
                ],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .expect("dispatch");
    out[0].clone()
}

#[test]
fn oob_structured_stride16_at_0x10000000() {
    let shader = build_oob_buffer_cs(16, 0x0C, true, false, 1);
    assert_eq!(
        dispatch_oob_case(&shader, Stride::Structured(16), 0, 1),
        [0xCAFE_BABE]
    );
    for &idx in &[0x1000_0000u32, 0xFFFF_FFFF, 0x8000_0000] {
        eprintln!("testing structured stride=16 idx={idx:#010x}");
        let got = dispatch_oob_case(&shader, Stride::Structured(16), idx, 1);
        assert_eq!(
            got,
            [0],
            "structured stride=16 idx={idx:#010x} must return 0"
        );
    }
}

#[test]
fn oob_structured_stride4_at_0x40000000() {
    let shader = build_oob_buffer_cs(4, 0x0C, true, false, 1);
    assert_eq!(
        dispatch_oob_case(&shader, Stride::Structured(4), 0, 1),
        [0xCAFE_BABE]
    );
    for &idx in &[0x4000_0000u32, 0xFFFF_FFFF, 0x8000_0000] {
        eprintln!("testing structured stride=4 idx={idx:#010x}");
        let got = dispatch_oob_case(&shader, Stride::Structured(4), idx, 1);
        assert_eq!(
            got,
            [0],
            "structured stride=4 idx={idx:#010x} must return 0"
        );
    }
}

#[test]
fn oob_raw_dwordx4_at_0xfffffffc() {
    let shader = build_oob_buffer_cs(0, 0x0E, false, true, 4);
    assert_eq!(
        dispatch_oob_case(&shader, Stride::Raw, 0, 4),
        [0xCAFE_BABE, 0x1111_2222, 0x3333_4444, 0x5555_6666]
    );
    for &off in &[0xFFFF_FFFCu32, 0xFFFF_FFF8, 0xFFFF_FFF4, 0xFFFF_FFF0] {
        eprintln!("testing raw dwordx4 off={off:#010x}");
        let got = dispatch_oob_case(&shader, Stride::Raw, off, 4);
        assert_eq!(
            got,
            [0, 0, 0, 0],
            "raw dwordx4 off={off:#010x} must return 0"
        );
    }
}

#[test]
fn oob_structured_stride32_at_0x08000000() {
    let shader = build_oob_buffer_cs(32, 0x0C, true, false, 1);
    assert_eq!(
        dispatch_oob_case(&shader, Stride::Structured(32), 0, 1),
        [0xCAFE_BABE]
    );
    for &idx in &[0x0800_0000u32, 0x1000_0000, 0xFFFF_FFFF, 0x8000_0000] {
        eprintln!("testing structured stride=32 idx={idx:#010x}");
        let got = dispatch_oob_case(&shader, Stride::Structured(32), idx, 1);
        assert_eq!(
            got,
            [0],
            "structured stride=32 idx={idx:#010x} must return 0"
        );
    }
}

#[test]
fn oob_raw_dword_high_offsets() {
    let shader = build_oob_buffer_cs(0, 0x0C, false, true, 1);
    assert_eq!(dispatch_oob_case(&shader, Stride::Raw, 0, 1), [0xCAFE_BABE]);
    for &off in &[
        0xFFFF_FFFCu32,
        0xFFFF_FFFF,
        0x8000_0000,
        0x4000_0000,
        0x1000_0000,
    ] {
        eprintln!("testing raw dword off={off:#010x}");
        let got = dispatch_oob_case(&shader, Stride::Raw, off, 1);
        assert_eq!(got, [0], "raw dword off={off:#010x} must return 0");
    }
}

#[test]
fn oob_raw_store_does_not_corrupt_element_zero() {
    // Store 0x1111_2222 to offset 0, then store 0xDEAD_BEEF to an OOB offset `s0`.
    // Element 0 must remain 0x1111_2222 (not overwritten by OOB clamping to 0).
    let mut words = Vec::new();
    words.push(s_mov_b32(3, 0x80)); // s3 = 0
    let output = s_load_dwordx4(8, 2); // heap slot 0 -> s[8:11]
    words.extend_from_slice(&output);
    // v4 = 0x1111_2222 via literal
    words.push(vop1(0x001, 4, 0xFF));
    words.push(0x1111_2222);
    // store v4 at offset 0
    words.extend_from_slice(&buffer_store_dword(4, 8, 0));
    // v1 = s0 (OOB byte offset), v5 = 0xDEAD_BEEF
    words.push(vop1(0x001, 1, sgpr(0)));
    words.push(vop1(0x001, 5, 0xFF));
    words.push(0xDEAD_BEEF);
    // buffer_store_dword v5, v1, s[8:11], 0 offen
    words.extend_from_slice(&[
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | ((8u32 >> 2) & 0x1F) << 16 | (5u32 << 8) | 1,
    ]);
    words.push(S_ENDPGM);

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 2,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 2,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let shader = lower_compute(&native);
    let host = Host::new(false).expect("hardware SM 6.6 device");
    let (root, pso) = host.compute_pipeline(&shader).expect("compute_pipeline");
    for &off in &[0xFFFF_FFFCu32, 0x8000_0000, 0xFFFF_FFF0] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Constants, 0) => off,
                (RootType::Table, 1) => 0,
                (RootType::ResourceHeap, _) | (RootType::SamplerHeap, _) => {
                    super::host::guard(u.root, 1, 0)
                        .unwrap_or_else(|| panic!("unexpected user data {:?}", (u.root, u.slot)))
                }
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(4)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch");
        assert_eq!(
            out[0][0], 0x1111_2222,
            "OOB raw store at {off:#010x} must not overwrite element 0"
        );
    }
}

/// SOP1 `op sdst, src0`.
const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | src0 as u32
}

/// DS `ds_bpermute_b32 vdst, vaddr, vdata0 offset:off`.
const fn ds_bpermute_b32(vdst: u8, vaddr: u8, vdata0: u8, offset: u16) -> [u32; 2] {
    [
        0xD800_0000 | (0x0B3 << 18) | offset as u32,
        (vdst as u32) << 24 | (vdata0 as u32) << 8 | vaddr as u32,
    ]
}

/// Builds, validates and runs a 32-thread Wave32 compute shader (with zero LDS
/// allocated) where `body` leaves each lane's result in `v3`, stored to dword
/// `threadIdInGroup.x` of a 32-dword raw UAV.
fn run_wave32_cs(body: &[u32], mut check: impl FnMut(&str, &[u32])) {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.extend_from_slice(body);
    // v7 = v0 << 2 (per-lane byte offset in the output UAV).
    words.push(vop2(0x1A, 7, 0x80 + 2, 0)); // v_lshlrev_b32 v7, 2, v0
    words.extend_from_slice(&buffer_store_dword_offen(3, 7, 8));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 12,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs, lds_bytes = 0
            threads: [32, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower(&lower::Parts {
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
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        stride_table: lowered.stride_table,
        root_uavs: lowered.root_uavs,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }

    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            // WARP does not support fixed WaveSize(32).
            Err(e) if warp => {
                eprintln!("{label}: skipped fixed wave size 32 ({e})");
                continue;
            }
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(32)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        check(label, &out[0]);
        ran += 1;
    }
    if ran == 0 {
        eprintln!("no SM 6.6 Wave32 device available; DXIL validation only");
    }
}

#[test]
fn ds_bpermute_b32_gathers_with_offset_and_wrap() {
    // Lane t (v0 = 0..31):
    //   v1 = 0x1000 + t          (source payload)
    //   v2 = ((31 - t) << 2) | (t & 3)  (reversed byte addr + low sub-dword bits)
    //   ds_bpermute_b32 v3, v2, v1 offset:12  (adds +3 lanes and wraps mod 32)
    let mut body = vec![
        vop2(0x25, 1, 0xFF, 0),      // v_add_nc_u32 v1, 0x1000, v0
        0x1000,                      // literal
        vop2(0x26, 2, 0x80 + 31, 0), // v_sub_nc_u32 v2, 31, v0
        vop2(0x1A, 2, 0x80 + 2, 2),  // v_lshlrev_b32 v2, 2, v2
        vop2(0x1B, 4, 0x80 + 3, 0),  // v_and_b32 v4, 3, v0
        vop2(0x1C, 2, vgpr(4), 2),   // v_or_b32 v2, v4, v2
    ];
    body.extend_from_slice(&ds_bpermute_b32(3, 2, 1, 12));

    run_wave32_cs(&body, |backend, got| {
        for (t, &val) in got.iter().enumerate() {
            let src = ((31 - t as u32) + 3) & 31;
            let want = 0x1000 + src;
            assert_eq!(
                val, want,
                "{backend}: lane {t} got {val:#x}, want {want:#x} (from lane {src})",
            );
        }
    });
}

#[test]
fn ds_bpermute_b32_zeroes_inactive_source_and_preserves_inactive_dst() {
    // Lane t (v0 = 0..31):
    //   v1 = 0x2000 + t          (source payload)
    //   v3 = 0xDEAD_0000 + t     (pre-existing destination value)
    //   v2 = (t + 8) << 2        (pulls from lane (t + 8) & 31)
    // Narrow EXEC to t < 16:
    //   - lanes 0..7  pull from active lanes 8..15   -> 0x2000 + t + 8
    //   - lanes 8..15 pull from inactive lanes 16..23 -> 0
    //   - lanes 16..31 are inactive during bpermute  -> preserve 0xDEAD_0000 + t
    let mut body = vec![
        vop2(0x25, 1, 0xFF, 0),     // v_add_nc_u32 v1, 0x2000, v0
        0x2000,                     // literal
        vop2(0x25, 3, 0xFF, 0),     // v_add_nc_u32 v3, 0xDEAD_0000, v0
        0xDEAD_0000,                // literal
        vop2(0x25, 2, 0x80 + 8, 0), // v_add_nc_u32 v2, 8, v0
        vop2(0x1A, 2, 0x80 + 2, 2), // v_lshlrev_b32 v2, 2, v2
        vop1(0x001, 5, 0x80 + 16),  // v_mov_b32 v5, 16
        vopc(0xC1, vgpr(0), 5),     // v_cmp_lt_u32 vcc, v0, v5
        sop1(0x3C, 6, 0x6A),        // s_and_saveexec_b32 s6, vcc_lo
    ];
    body.extend_from_slice(&ds_bpermute_b32(3, 2, 1, 0));
    body.push(sop1(0x03, 0x7E, 6)); // s_mov_b32 exec_lo, s6

    run_wave32_cs(&body, |backend, got| {
        for (t, &val) in got.iter().enumerate() {
            let t = t as u32;
            let want = if t < 8 {
                0x2000 + t + 8
            } else if t < 16 {
                0
            } else {
                0xDEAD_0000 + t
            };
            assert_eq!(
                val, want,
                "{backend}: lane {t} got {val:#010x}, want {want:#010x}",
            );
        }
    });
}

#[test]
fn ds_bpermute_b32_divergent_branch_zeroes_skipped_lanes() {
    // Same mask as above, but with `s_cbranch_execz` around `ds_bpermute_b32`
    // so structurisation emits a per-lane CFG branch rather than a guard.
    let mut body = vec![
        vop2(0x25, 1, 0xFF, 0),     // v_add_nc_u32 v1, 0x3000, v0
        0x3000,                     // literal
        vop2(0x25, 3, 0xFF, 0),     // v_add_nc_u32 v3, 0xBEEF_0000, v0
        0xBEEF_0000,                // literal
        vop2(0x25, 2, 0x80 + 8, 0), // v_add_nc_u32 v2, 8, v0
        vop2(0x1A, 2, 0x80 + 2, 2), // v_lshlrev_b32 v2, 2, v2
        vop1(0x001, 5, 0x80 + 16),  // v_mov_b32 v5, 16
        vopc(0xC1, vgpr(0), 5),     // v_cmp_lt_u32 vcc, v0, v5
        sop1(0x3C, 6, 0x6A),        // s_and_saveexec_b32 s6, vcc_lo
        0xBF88_0002,                // s_cbranch_execz +2 (skip the 2-dword DS op)
    ];
    body.extend_from_slice(&ds_bpermute_b32(3, 2, 1, 0));
    body.push(sop1(0x03, 0x7E, 6)); // s_mov_b32 exec_lo, s6

    run_wave32_cs(&body, |backend, got| {
        for (t, &val) in got.iter().enumerate() {
            let t = t as u32;
            let want = if t < 8 {
                0x3000 + t + 8
            } else if t < 16 {
                0
            } else {
                0xBEEF_0000 + t
            };
            assert_eq!(
                val, want,
                "{backend}: lane {t} got {val:#010x}, want {want:#010x}",
            );
        }
    });
}

fn run_wave64_cs(
    threads: [u32; 3],
    rsrc2: u32,
    body: &[u32],
    mode: super::Wave64Mode,
    mut check: impl FnMut(&str, &[u32]),
) {
    let mut words = Vec::new();
    words.push(s_mov_b32(5, 0x80)); // s[4:5] = descriptor table address
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.extend_from_slice(body);
    words.push(vop2(0x1A, 7, 0x80 + 2, 0)); // v7 = v0 << 2
    words.extend_from_slice(&buffer_store_dword_offen(3, 7, 8));
    words.push(S_ENDPGM);

    let total = threads.iter().product::<u32>();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 12,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo { rsrc2, threads }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        options: super::TranslateOptions { wave64: mode },
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        stride_table: lowered.stride_table,
        root_uavs: lowered.root_uavs,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }

    let host = Host::new(false).expect("hardware SM 6.6 device");
    let (root, pso) = host.compute_pipeline(&shader).expect("compute_pipeline");
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(total)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .expect("dispatch");
    check("hardware", &out[0]);
}

#[test]
fn ds_bpermute_b32_wave64_emulated_gathers_across_halves() {
    // 64-thread Wave64 dispatch: each lane pulls from `t ^ 32`, crossing
    // between the two 32-lane host halves under Wave32 and Emulate.
    let mut body = vec![
        vop2(0x25, 1, 0xFF, 0),      // v1 = 0x4000 + v0
        0x4000,
        vop2(0x1D, 2, 0x80 + 32, 0), // v2 = v0 ^ 32
        vop2(0x1A, 2, 0x80 + 2, 2),  // v2 = (v0 ^ 32) << 2
    ];
    body.extend_from_slice(&ds_bpermute_b32(3, 2, 1, 0));

    for mode in [super::Wave64Mode::Wave32, super::Wave64Mode::Emulate] {
        run_wave64_cs([64, 1, 1], 5 << 1, &body, mode, |backend, got| {
            for (t, &val) in got.iter().enumerate() {
                let want = 0x4000 + (t as u32 ^ 32);
                assert_eq!(val, want, "{backend} ({mode:?}): lane {t}");
            }
        });
    }
}

#[test]
fn waveid_and_group_info_match_virtual_wave64_in_wave32_and_emulate() {
    // 128-thread Wave64 dispatch with 6 user SGPRs (s0..s5; s[4:5] is the UAV
    // table) + TG_SIZE_EN (s6 = GroupInfo; s[8:11] holds the UAV descriptor):
    //   s7  = s_get_waveid_in_workgroup
    //   s12 = (GroupInfo >> 6) & 0x3f (wave index)
    //   s13 = GroupInfo & 0x3f        (wave count = 2)
    //   v3  = (s13 << 16) | (s12 << 8) | s7
    let body = [
        0xF4A8_01C0u32,
        0xFA00_0000,                  // s_get_waveid_in_workgroup s7
        sop2(0x27, 12, 6, 0xFF),      // s_bfe_u32 s12, s6, (6 << 16 | 6)
        6 << 16 | 6,
        sop2(0x0E, 13, 6, 128 + 63),  // s_and_b32 s13, s6, 63
        sop2(0x1E, 12, 12, 128 + 8),  // s_lshl_b32 s12, s12, 8
        sop2(0x1E, 13, 13, 128 + 16), // s_lshl_b32 s13, s13, 16
        sop2(0x10, 7, 7, 12),         // s_or_b32 s7, s7, s12
        sop2(0x10, 7, 7, 13),         // s_or_b32 s7, s7, s13
        vop1(0x001, 3, sgpr(7)),      // v_mov_b32 v3, s7
    ];

    for mode in [super::Wave64Mode::Wave32, super::Wave64Mode::Emulate] {
        run_wave64_cs(
            [128, 1, 1],
            TG_INFO_RSRC2,
            &body,
            mode,
            |backend, got| {
                for (t, &val) in got.iter().enumerate() {
                    let w = (t as u32) / 64;
                    let want = (2 << 16) | (w << 8) | w;
                    assert_eq!(val, want, "{backend} ({mode:?}): lane {t}");
                }
            },
        );
    }
}

/// DS `ds_swizzle_b32 vdst, vsrc offset:offset`.
const fn ds_swizzle_b32(vdst: u8, vsrc: u8, offset: u16) -> [u32; 2] {
    [
        0x36 << 26 | 0x35 << 18 | offset as u32,
        (vdst as u32) << 24 | vsrc as u32,
    ]
}

/// Lowers a synthetic RDNA2 compute shader of `threads` threads in one row
/// (thread id in `v0`): 4 dwords of root constants in `s0..s3`, the
/// descriptor table address in `s4`, a raw UAV at heap index 0.
fn build_cs_threads(words: &[u32], threads: u32) -> HostShader {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 4,
                user_data: 4,
                slot: 0,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code,
    };
    let shader = try_lower_compute(&native).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    shader
}

/// One wave of 32 lanes holding 1..=32 (the input of the ISA's examples)
/// through `ds_swizzle_b32` with `offset`: what each lane reads.
fn swizzle_32(offset: u16) -> Vec<Vec<u32>> {
    let mut words = vec![s_mov_b32(5, 0x80)];
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.push(vop2(0x25, 1, 129, 0)); // v_add_nc_u32 v1, 1, v0
    words.extend_from_slice(&ds_swizzle_b32(2, 1, offset));
    words.push(vop2(0x1A, 3, 130, 0)); // v_lshlrev_b32 v3, 2, v0
    words.extend_from_slice(&buffer_store_dword_offen(2, 3, 8));
    words.push(S_ENDPGM);
    let shader = build_cs_threads(&words, 32);

    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            // A host without 32-lane waves cannot run it.
            Err(e) if warp => {
                eprintln!("{label}: no pipeline ({e}); skipped");
                continue;
            }
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let uavs = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(32)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push(uavs[0].clone());
    }
    if out.is_empty() {
        eprintln!("no SM 6.6 device with 32-lane waves; skipped");
    }
    out
}

/// The basic modes: a quad permutation and a 32-lane and/or/xor mapping.
#[test]
fn ds_swizzle_basic_modes() {
    // offset[15]: lanes of each quad read lanes 3, 2, 1, 0 of it.
    let reversed: Vec<u32> = (0..32).map(|i| (i ^ 3) + 1).collect();
    // and 0x1f, or 0, xor 0x10: the halves swap.
    let swapped: Vec<u32> = (0..32).map(|i| (i ^ 16) + 1).collect();
    for (offset, want) in [(0x801B, reversed), (0x401F, swapped)] {
        for got in swizzle_32(offset) {
            assert_eq!(got, want, "offset {offset:#06x}");
        }
    }
}

/// Offsets from 0xE000: the FFT orders, as the ISA lists them for inputs
/// 1..=0x20 (RDNA2 ISA, 12.13.1).
#[test]
fn ds_swizzle_fft_mode() {
    let cases: [(u16, [u32; 32]); 3] = [
        (
            0xE000,
            [
                0x01, 0x11, 0x09, 0x19, 0x05, 0x15, 0x0d, 0x1d, 0x03, 0x13, 0x0b, 0x1b, 0x07, 0x17,
                0x0f, 0x1f, 0x02, 0x12, 0x0a, 0x1a, 0x06, 0x16, 0x0e, 0x1e, 0x04, 0x14, 0x0c, 0x1c,
                0x08, 0x18, 0x10, 0x20,
            ],
        ),
        (
            0xE010,
            [
                0x01, 0x09, 0x05, 0x0d, 0x03, 0x0b, 0x07, 0x0f, 0x02, 0x0a, 0x06, 0x0e, 0x04, 0x0c,
                0x08, 0x10, 0x11, 0x19, 0x15, 0x1d, 0x13, 0x1b, 0x17, 0x1f, 0x12, 0x1a, 0x16, 0x1e,
                0x14, 0x1c, 0x18, 0x20,
            ],
        ),
        // 0x1f: no swizzle.
        (0xE01F, std::array::from_fn(|i| i as u32 + 1)),
    ];
    for (offset, want) in cases {
        for got in swizzle_32(offset) {
            assert_eq!(got, want, "offset {offset:#06x}");
        }
    }
    // The other bits of an FFT offset do not matter.
    for got in swizzle_32(0xFFF0) {
        assert_eq!(got, cases[1].1, "offset 0xfff0");
    }
}

/// Offsets from 0xC000 to 0xDFFF: the wave's 32 lanes rotate left by
/// offset[9:5], or right with offset[10] (RDNA2 ISA, 12.13.1).
#[test]
fn ds_swizzle_rotate_mode() {
    let rotated = |by: u32| -> Vec<u32> { (0..32).map(|i| (i + by) % 32 + 1).collect() };
    for (offset, by) in [
        (0xC000 | 1 << 5, 1),            // the ISA's example: {2, 3, ... 0x20, 1}
        (0xC000 | 1 << 10 | 1 << 5, 31), // and {0x20, 1, ... 0x1f}
        (0xC000, 0),
        (0xC000 | 5 << 5, 5),
        (0xC000 | 1 << 10 | 5 << 5, 27),
        (0xD800 | 31 << 5, 31), // bits 11 and 12 are not read
    ] {
        for got in swizzle_32(offset) {
            assert_eq!(got, rotated(by), "offset {offset:#06x}");
        }
    }
}

/// Rotations with a mask (offset[4:0]): the mask's bits of the lane index
/// stay, the others rotate, as the ISA's pseudocode has it. Its examples for
/// mask 1 list other lanes; the lanes here are the ones measured on RDNA2
/// hardware (gfx1036, by Lyrnc in Marvisak/VIctory#606, who gives the first
/// and last entries of each list and reports the pseudocode matching all 2048
/// offsets 0xC000-0xC7FF, and bits 11 and 12 being ignored).
#[test]
fn ds_swizzle_masked_rotations() {
    // The lane each lane reads.
    let cases: [(u16, [u32; 32]); 4] = [
        // Left by 1, mask 1: 0, 3, 2, 5, 4, 7, 6, 9, 8, ..., 28, 31, 30, 1.
        (
            0xC021,
            [
                0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22,
                25, 24, 27, 26, 29, 28, 31, 30, 1,
            ],
        ),
        // Right by 1, mask 1: 30, 1, 0, 3, 2, 5, 4, ...
        (
            0xC421,
            [
                30, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20,
                23, 22, 25, 24, 27, 26, 29, 28, 31,
            ],
        ),
        // Left by 1, mask 3: 0, 1, 2, 7, 4, 5, 6, 11, ...
        (
            0xC023,
            [
                0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14, 19, 16, 17, 18, 23, 20, 21, 22,
                27, 24, 25, 26, 31, 28, 29, 30, 3,
            ],
        ),
        // Bits 11 and 12 set: as 0xC021.
        (
            0xD821,
            [
                0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22,
                25, 24, 27, 26, 29, 28, 31, 30, 1,
            ],
        ),
    ];
    for (offset, lanes) in cases {
        // Lane i holds i + 1.
        let want: Vec<u32> = lanes.iter().map(|j| j + 1).collect();
        for got in swizzle_32(offset) {
            assert_eq!(got, want, "offset {offset:#06x}");
        }
    }
}

/// Runs VOP1 `op` on each input and compares the result dword.
fn check_cvt(name: &str, op: u16, cases: &[(u32, u32)]) {
    let shader = build_valu_cs(&[vop1(op, 0, vgpr(1))]);
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [c.0, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got,
            want,
            "{backend}: {name}({:#010x} = {:e}) = {got:#010x}, want {want:#010x}",
            inp[0],
            f32::from_bits(inp[0]),
        );
    });
}

#[test]
fn v_cvt_rpi_i32_f32_rounds_to_nearest() {
    // floor(x + 0.5) of the exact sum: an f32 sum rounds to even first, so
    // it is 1 too high just below 0.5 and for odd integers from 2^23 up.
    check_cvt(
        "v_cvt_rpi_i32_f32",
        0x00C,
        &[
            (0x0000_0000, 0),
            (0x3EFF_FFFF, 0), // 0.5 - 2^-25
            (0x3F00_0000, 1), // 0.5: ties go up
            (0xBF00_0000, 0), // -0.5
            (0xBF00_0001, 0xFFFF_FFFF),
            (0x3FC0_0000, 2),           // 1.5
            (0x4020_0000, 3),           // 2.5
            (0xBFC0_0000, 0xFFFF_FFFF), // -1.5
            (0xC020_0000, 0xFFFF_FFFE), // -2.5
            (0x0000_0001, 0),           // denormals, flushed or not
            (0x8000_0001, 0),
            (0x4A80_0001, 0x0040_0001), // 2^22 + 0.5
            (0x4B00_0001, 0x0080_0001), // 2^23 + 1
            (0x4B7F_FFFF, 0x00FF_FFFF), // 2^24 - 1
            (0xCB00_0001, 0xFF7F_FFFF), // -(2^23 + 1)
            (0xCB7F_FFFF, 0xFF00_0001),
            (0x4EFF_FFFF, 0x7FFF_FF80),
            (0x4F00_0000, 0x7FFF_FFFF), // 2^31 and up saturate
            (0x7F80_0000, 0x7FFF_FFFF),
            (0xCF00_0000, 0x8000_0000),
            (0xCF00_0001, 0x8000_0000),
            (0xFF80_0000, 0x8000_0000),
        ],
    );
}

#[test]
fn v_cvt_flr_i32_f32_rounds_down() {
    check_cvt(
        "v_cvt_flr_i32_f32",
        0x00D,
        &[
            (0x0000_0000, 0),
            (0x8000_0000, 0),
            (0x3FC0_0000, 1),           // 1.5
            (0xBFC0_0000, 0xFFFF_FFFE), // -1.5
            (0x3F7F_FFFF, 0),
            (0xBF80_0000, 0xFFFF_FFFF), // -1
            (0xB300_0000, 0xFFFF_FFFF), // -2^-25
            (0x4B7F_FFFF, 0x00FF_FFFF),
            (0x4F00_0000, 0x7FFF_FFFF),
            (0x7F80_0000, 0x7FFF_FFFF),
            (0xCF00_0001, 0x8000_0000),
            (0xFF80_0000, 0x8000_0000),
        ],
    );
}

#[test]
fn v_cvt_flr_and_rpi_i32_f32_saturate_nan_by_sign() {
    // Unlike v_cvt_i32_f32 (NaN gives 0), these saturate whatever has an
    // exponent above 30, a NaN too: to the largest or the smallest integer
    // by its sign.
    let cases = [
        (0x7FC0_0000, 0x7FFF_FFFF),
        (0x7F80_0001, 0x7FFF_FFFF),
        (0x7FFF_FFFF, 0x7FFF_FFFF),
        (0xFFC0_0000, 0x8000_0000),
        (0xFF80_0001, 0x8000_0000),
        (0xFFFF_FFFF, 0x8000_0000),
    ];
    check_cvt("v_cvt_rpi_i32_f32", 0x00C, &cases);
    check_cvt("v_cvt_flr_i32_f32", 0x00D, &cases);
    // v_cvt_i32_f32 keeps giving 0.
    check_cvt(
        "v_cvt_i32_f32",
        0x008,
        &[(0x7FC0_0000, 0), (0xFFC0_0000, 0), (0x7F80_0001, 0)],
    );
}

#[test]
fn v_mul_legacy_f32_treats_denormals_as_zero() {
    // v_mul_legacy_f32 v0, v1, v2 (VOP2 opcode 0x007 in its VOP3 encoding)
    let shader = build_valu_cs(&vop3(0x107, 0, vgpr(1), vgpr(2), vgpr(0)));
    let f = f32::to_bits;
    let (denorm, neg_denorm, min_normal) = (0x0000_0001u32, 0x8040_0000u32, 0x0080_0000u32);

    let cases: &[(u32, u32, u32)] = &[
        // Zero or denormal times anything, non-finite included, is +0.
        (f(0.0), f(f32::INFINITY), f(0.0)),
        (f(-0.0), f(f32::NAN), f(0.0)),
        (f(f32::NAN), f(0.0), f(0.0)),
        (denorm, f(f32::INFINITY), f(0.0)),
        (f(f32::NAN), neg_denorm, f(0.0)),
        (denorm, f(2.0f32.powi(100)), f(0.0)),
        // The smallest normal is not zero.
        (min_normal, f(2.0), f(f32::MIN_POSITIVE * 2.0)),
        // Ordinary products.
        (f(1.5), f(-4.0), f(-6.0)),
        (f(3.0), f(0.5), f(1.5)),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(a, b, _)| [a, b, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].2;
        assert_eq!(
            got, want,
            "{backend}: v_mul_legacy_f32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn f16_omod_applies_before_clamp() {
    // VOP3 v_add_f16 (0x132 = 0x100 + VOP2 0x32): v0 = (v1 + v2) * omod [clamp]
    let mul2 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 1, false));
    let mul4 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 2, false));
    let div2 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 3, false));
    let div2_clamp = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 3, true));

    // f16 constants: 0.25=0x3400, 0.5=0x3800, 0.75=0x3A00, 1.0=0x3C00, 1.5=0x3E00, 2.0=0x4000, -0.5=0xB800
    run_cases(
        &mul2,
        &[[0x3400, 0x3800, 0, 0]], // (0.25 + 0.5) * 2 = 1.5 (0x3E00)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x3E00, "{backend}: v_add_f16 mul:2");
        },
    );
    run_cases(
        &mul4,
        &[[0x3400, 0x3400, 0, 0]], // (0.25 + 0.25) * 4 = 2.0 (0x4000)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x4000, "{backend}: v_add_f16 mul:4");
        },
    );
    run_cases(
        &div2,
        &[[0x3E00, 0x3E00, 0, 0]], // (1.5 + 1.5) / 2 = 1.5 (0x3E00)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x3E00, "{backend}: v_add_f16 div:2");
        },
    );
    // Ordering check: (0.75 + 0.75) = 1.5; div:2 before clamp gives 0.75 (0x3A00),
    // whereas clamp before div:2 would give 0.5 (0x3800).
    let clamp_cases: &[(u16, u16, u16)] = &[
        (0x3A00, 0x3A00, 0x3A00), // (0.75 + 0.75) / 2 = 0.75 -> 0.75
        (0x3E00, 0x3E00, 0x3C00), // (1.5 + 1.5) / 2 = 1.5 -> clamped to 1.0
        (0xB800, 0xB800, 0x0000), // (-0.5 + -0.5) / 2 = -0.5 -> clamped to 0.0
    ];
    let inputs: Vec<[u32; 4]> = clamp_cases
        .iter()
        .map(|&(a, b, _)| [u32::from(a), u32::from(b), 0, 0])
        .collect();
    run_cases(&div2_clamp, &inputs, |backend, i, _, got| {
        let want = clamp_cases[i].2;
        assert_eq!(
            got as u16, want,
            "{backend}: v_add_f16 div:2 clamp case {i}: got {:#06x}, want {want:#06x}",
            got as u16,
        );
    });
}

#[test]
fn cvt_16bit_int_from_f16_saturates_and_honors_modifiers() {
    // VOP3 v_cvt_u16_f16 = 0x1D2 (0x180 + VOP1 0x52)
    // VOP3 v_cvt_i16_f16 = 0x1D3 (0x180 + VOP1 0x53)
    let cvt_u16 = build_valu_cs(&vop3_mod(0x1D2, 0, vgpr(1), 0, 0, 0, 0, 0, false));
    let cvt_u16_abs = build_valu_cs(&vop3_mod(0x1D2, 0, vgpr(1), 0, 0, 1, 0, 0, false));
    let cvt_i16 = build_valu_cs(&vop3_mod(0x1D3, 0, vgpr(1), 0, 0, 0, 0, 0, false));
    let cvt_i16_neg = build_valu_cs(&vop3_mod(0x1D3, 0, vgpr(1), 0, 0, 0, 1, 0, false));

    let u16_cases: &[(u16, u16)] = &[
        (0xBC00, 0),      // -1.0 -> 0
        (0xFC00, 0),      // -inf -> 0
        (0x7E00, 0),      // NaN -> 0
        (0x57BC, 123),    // 123.75 -> 123
        (0x7BFF, 65504),  // 65504.0 (max finite f16) -> 65504
        (0x7C00, 0xFFFF), // +inf -> 65535 (saturated)
    ];
    let u16_in: Vec<[u32; 4]> = u16_cases.iter().map(|&(x, _)| [u32::from(x), 0, 0, 0]).collect();
    run_cases(&cvt_u16, &u16_in, |backend, i, inp, got| {
        let want = u16_cases[i].1;
        assert_eq!(
            got as u16, want,
            "{backend}: v_cvt_u16_f16({:#06x}) = {:#06x}, want {want:#06x}",
            inp[0], got as u16,
        );
    });
    run_cases(
        &cvt_u16_abs,
        &[[0xD7BC, 0, 0, 0]], // |-123.75| -> 123
        |backend, _, _, got| {
            assert_eq!(got as u16, 123, "{backend}: v_cvt_u16_f16 |src0|");
        },
    );

    let i16_cases: &[(u16, u16)] = &[
        (0xFC00, 0x8000),          // -inf -> -32768
        (0xF8E2, 0x8000),          // -40000.0 -> -32768 (saturated, not wrapped)
        (0xD7BC, (-123i16) as u16), // -123.75 -> -123
        (0x7E00, 0),               // NaN -> 0
        (0x57BC, 123),             // 123.75 -> 123
        (0x78E2, 0x7FFF),          // +40000.0 -> 32767 (saturated, not wrapped)
        (0x7C00, 0x7FFF),          // +inf -> 32767
    ];
    let i16_in: Vec<[u32; 4]> = i16_cases.iter().map(|&(x, _)| [u32::from(x), 0, 0, 0]).collect();
    run_cases(&cvt_i16, &i16_in, |backend, i, inp, got| {
        let want = i16_cases[i].1;
        assert_eq!(
            got as u16, want,
            "{backend}: v_cvt_i16_f16({:#06x}) = {:#06x}, want {want:#06x}",
            inp[0], got as u16,
        );
    });
    run_cases(
        &cvt_i16_neg,
        &[[0x57BC, 0, 0, 0], [0x78E2, 0, 0, 0]], // -(123.75) -> -123; -(40000.0) -> -32768
        |backend, i, _, got| {
            let want = [(-123i16) as u16, 0x8000][i];
            assert_eq!(got as u16, want, "{backend}: v_cvt_i16_f16 -src0 case {i}");
        },
    );
}

/// MTBUF `op vdata, v5, s[8:11], 0 format:fmt offset:16` (no offen/idxen;
/// `op` 4..7 stores x..xyzw, 12..15 their d16 forms).
const fn mtbuf_store(op: u8, fmt: u8, vdata: u8) -> [u32; 2] {
    [
        0x3A << 26 | (fmt as u32) << 19 | (op as u32 & 7) << 16 | 16,
        0x80 << 24 | (op as u32 >> 3 & 1) << 21 | (8 >> 2) << 16 | (vdata as u32) << 8 | 5,
    ]
}

/// Runs `shader` once with root constants `inputs` on every available
/// SM 6.6 host and returns the first `dwords` of its raw UAV, per device.
fn run_dwords(shader: &HostShader, inputs: [u32; 4], dwords: u32) -> Vec<(&'static str, Vec<u32>)> {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Constants, 0) => inputs[u.dword as usize],
                (RootType::Table, 1) => 0,
                (root, _) => super::host::guard(root, 1, 0)
                    .unwrap_or_else(|| panic!("unexpected user data {:?}", (root, u.slot))),
            };
        }
        let r = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(dwords)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, r[0].clone()));
    }
    assert!(
        !out.is_empty(),
        "neither hardware nor WARP SM 6.6 device available"
    );
    out
}

#[test]
fn tbuffer_store_format_writes_the_formats_layout() {
    let f = |x: f32| x.to_bits();
    // (what, op, format, data in v1..v4, dwords 4..8 afterwards)
    type Case = (&'static str, u8, u8, [u32; 4], [u32; 4]);
    let cases: &[Case] = &[
        (
            "32_32_32_32 float xyzw",
            7,
            77,
            [f(1.5), f(-0.0), 0x7FC0_1234, 0x0000_0001],
            [f(1.5), f(-0.0), 0x7FC0_1234, 0x0000_0001],
        ),
        (
            "32_32_32_32 sint xyzw",
            7,
            76,
            [u32::MAX, 0x8000_0000, 7, 0x1234_5678],
            [u32::MAX, 0x8000_0000, 7, 0x1234_5678],
        ),
        (
            "32_32_32 float xyz",
            6,
            74,
            [f(2.0), f(3.0), f(4.0), 0xDEAD_BEEF],
            [f(2.0), f(3.0), f(4.0), 0],
        ),
        (
            "32_32 uint xy",
            5,
            63,
            [0xAABB_CCDD, 9, 0xDEAD_BEEF, 0xDEAD_BEEF],
            [0xAABB_CCDD, 9, 0, 0],
        ),
        (
            "32 float x",
            4,
            22,
            [f(-7.25), 1, 2, 3],
            [f(-7.25), 0, 0, 0],
        ),
        (
            // d16: the halves are stored as they are.
            "16_16 float d16_xy",
            13,
            29,
            [0xBC00_3555, 0xDEAD_BEEF, 0, 0],
            [0xBC00_3555, 0, 0, 0],
        ),
        (
            "16_16_16_16 float d16_xyzw",
            15,
            71,
            [0x1234_5678, 0x7C00_FC00, 0, 0],
            [0x1234_5678, 0x7C00_FC00, 0, 0],
        ),
    ];
    for &(what, op, fmt, inputs, want) in cases {
        let mut body = vec![vop1(0x001, 5, 0x80)]; // v_mov_b32 v5, 0
        body.extend(mtbuf_store(op, fmt, 1));
        let shader = build_valu_cs(&body);
        for (backend, out) in run_dwords(&shader, inputs, 8) {
            assert_eq!(
                out[4..8],
                want,
                "{backend}: tbuffer_store {what}: wrote {:#010x?}, want {want:#010x?}",
                &out[4..8],
            );
        }
    }
}

#[test]
fn tbuffer_store_format_refuses_what_it_does_not_convert() {
    for (what, op, fmt, reason) in [
        ("8_8_8_8 unorm", 7, 56, "a converted or narrow format"),
        ("16_16 uint", 5, 27, "a converted or narrow format"),
        ("16 float", 4, 13, "an element under a dword"),
        (
            "32_32_32_32 float, xy only",
            5,
            77,
            "some of its format's channels",
        ),
        ("32 float d16_x", 12, 22, "a converted or narrow format"),
        // f32 to f16 waits for nearest-even narrowing (#422).
        ("16_16 float xy", 5, 29, "a converted or narrow format"),
        (
            "16_16_16_16 float xyzw",
            7,
            71,
            "a converted or narrow format",
        ),
    ] {
        let mut body = vec![vop1(0x001, 5, 0x80)];
        body.extend(mtbuf_store(op, fmt, 1));
        match try_build_valu_cs(&body) {
            Ok(_) => panic!("tbuffer_store {what} lowered; it should be refused"),
            Err(e) => assert!(e.contains(reason), "tbuffer_store {what}: refused with {e}"),
        }
    }
}

/// Contributed by @aziaconsal2026 on the pull request: the vector (`offen`)
/// and scalar (`soffset`) byte offsets add to the immediate one, and the
/// neighbouring dwords stay as they were.
#[test]
fn tbuffer_store_adds_scalar_and_vector_byte_offsets() {
    // v1 = payload, v2 = vector byte offset, s2 = scalar byte offset,
    // v4 = sentinel. Fill the window first so stray writes are observable.
    let mut body = Vec::new();
    for offset in (16..44).step_by(4) {
        body.extend(buffer_store_dword(4, 8, offset));
    }
    body.push(vop1(0x001, 5, vgpr(2))); // v_mov_b32 v5, v2
    let [opcode, operands] = mtbuf_store(4, 20, 1); // 32_UINT, immediate byte offset 16
    body.extend([opcode | (1 << 12), (operands & 0x00ff_ffff) | (2 << 24)]); // offen, soffset = s2
    let shader = build_valu_cs(&body);

    let payload = 0x1357_9bdf;
    let sentinel = 0xa5a5_a5a5;
    for (vector, scalar) in [(0, 0), (4, 0), (0, 8), (4, 8), (12, 8)] {
        let mut expected = [sentinel; 7];
        expected[((vector + scalar) / 4) as usize] = payload;
        for (backend, words) in run_dwords(&shader, [payload, vector, scalar, sentinel], 11) {
            assert_eq!(
                &words[4..],
                &expected,
                "{backend}: vector={vector} scalar={scalar}"
            );
        }
    }
}

#[test]
fn m0_relative_swaps_and_moves_index_source_and_destination_apart() {
    // v1..v4 = 10, 20, 30, 40. v_movrelsd_2_b32 and v_swaprel_b32 offset
    // their source by M0[9:0] and their destination by M0[25:16]; other M0
    // bits are ignored. Each case reads one VGPR back into v0.
    const SWAPREL: u16 = 0x068;
    const MOVRELSD_2: u16 = 0x048;
    let cases: &[(u16, u8, u32, u8, u32)] = &[
        // (opcode, vdst, M0, register read back, expected)
        // v[1 + 2] <-> v[1 + 0]
        (SWAPREL, 1, 2 << 16, 1, 30),
        (SWAPREL, 1, 2 << 16, 3, 10),
        // v[1 + 0] <-> v[1 + 3]
        (SWAPREL, 1, 3, 4, 10),
        (SWAPREL, 1, 3, 1, 40),
        // Both name v2: it keeps its value.
        (SWAPREL, 1, 1 << 16 | 1, 2, 20),
        // v[5 + 1] = v[1 + 2]
        (MOVRELSD_2, 5, 1 << 16 | 2, 6, 30),
        // The same with every ignored M0 bit set.
        (MOVRELSD_2, 5, 0xFC00_FC00 | 1 << 16 | 2, 6, 30),
    ];
    for &(op, vdst, m0, back, want) in cases {
        let shader = build_valu_cs(&[
            s_mov_b32(124, 0xFF), // s_mov_b32 m0, literal
            m0,
            vop1(op, vdst, vgpr(1)),
            vop1(0x001, 0, vgpr(back)), // v_mov_b32 v0, v[back]
        ]);
        run_cases(&shader, &[[10, 20, 30, 40]], |backend, _, _, got| {
            assert_eq!(
                got, want,
                "{backend}: op {op:#05x} v{vdst}, v1 with M0 {m0:#010x}, v{back} = {got}, want {want}",
            );
        });
    }
}

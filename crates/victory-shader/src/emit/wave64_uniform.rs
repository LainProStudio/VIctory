//! Thread-group uniformity, M0 waterfall detection, and convergent predication
//! planning for Wave64 compute lowering.

use std::collections::{HashMap, HashSet};

use super::lower::{Spill, is_lds_mem, reads_mask_as_data};
use super::wave::Section;
use crate::gfx10::{Fields, NULL, Operand};
use crate::io::Input;
use crate::ir::{EXEC, Inst, Program, Reg, Value};
use crate::resource::{Moved, Resources, Site};
use crate::structure::{Atom, Bool, LaneExit, Structured};

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

fn is_pure_lane_valu(inst: &Inst) -> bool {
    let Fields::Valu(v) = &inst.fields else {
        return false;
    };
    if v.dpp.is_some() || v.sdwa.is_some() {
        return false;
    }
    let name = inst.name();
    !name.starts_with("v_mbcnt")
        && !name.starts_with("v_permlane")
        && !name.starts_with("v_cmp_")
        && !name.starts_with("v_cmpx_")
        && !matches!(name, "v_readlane_b32" | "v_writelane_b32")
}

fn is_uniform_scalar_op(inst: &Inst) -> bool {
    let iname = inst.name();
    matches!(inst.fields, Fields::Sop { .. } | Fields::Smem { .. })
        && !iname.starts_with("s_getreg")
        && !iname.starts_with("s_bcnt")
        && !iname.starts_with("s_ff1")
        && !iname.contains("saveexec")
        && iname != "s_get_waveid_in_workgroup"
}

fn operand_is_uniform(inst: &Inst, op: &Operand, vals: &[bool]) -> bool {
    match *op {
        Operand::Vector { reg, .. } => inst
            .read(Reg::V(reg))
            .is_some_and(|v| vals.get(v.0 as usize).copied().unwrap_or(false)),
        Operand::Scalar { code, .. } if code != NULL => inst
            .read(Reg::S(code))
            .is_some_and(|v| vals.get(v.0 as usize).copied().unwrap_or(false)),
        _ => true,
    }
}

fn overwritten_dst_vreg(inst: &Inst) -> Option<u8> {
    let Fields::Valu(v) = &inst.fields else {
        return None;
    };
    let name = inst.name();
    if matches!(
        name,
        "v_fmac_f32"
            | "v_mac_f32"
            | "v_mac_legacy_f32"
            | "v_fmac_f16"
            | "v_mac_f16"
            | "v_cvt_f16_f32"
            | "v_swap_b32"
            | "v_swaprel_b32"
            | "v_movreld_b32"
            | "v_movrelsd_b32"
            | "v_movrelsd_2_b32"
    ) || name.starts_with("v_fma_mix")
        || name.starts_with("v_cmpx")
    {
        return None;
    }
    match v.dst {
        Some(Operand::Vector { reg, .. }) => Some(reg),
        _ => None,
    }
}

fn valu_reads_uniform(inst: &Inst, vals: &[bool]) -> bool {
    let skip_dst = overwritten_dst_vreg(inst);
    inst.reads.iter().all(|&(reg, v)| match reg {
        Reg::S(c) if c == EXEC || c == EXEC + 1 => true,
        Reg::V(r) if Some(r) == skip_dst => true,
        _ => vals.get(v.0 as usize).copied().unwrap_or(false),
    })
}

fn bool_is_uniform(st: &Structured, b: Bool, vals: &[bool], cmps: &[bool]) -> bool {
    st.bools.support(b).iter().all(|&a| match a {
        Atom::Uniform(v) => vals.get(v.0 as usize).copied().unwrap_or(false),
        Atom::Lane(v) => cmps.get(v.0 as usize).copied().unwrap_or(false),
        _ => false,
    })
}

fn lane_independent_reads_ok(
    st: &Structured,
    b: usize,
    ii: usize,
    inst: &Inst,
    vals: &[bool],
    cmps: &[bool],
) -> bool {
    let iname = inst.name();
    let scalar_op = is_uniform_scalar_op(inst);
    let pure_valu = is_pure_lane_valu(inst) && st.guards[b][ii].is_none();
    if !scalar_op && !pure_valu {
        return false;
    }
    if pure_valu
        && let Fields::Valu(v) = &inst.fields
        && matches!(
            iname,
            "v_cndmask_b32" | "v_add_co_ci_u32" | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32"
        )
    {
        let (Some(s0), Some(s1)) = (v.src.first(), v.src.get(1)) else {
            return false;
        };
        if !operand_is_uniform(inst, s0, vals) || !operand_is_uniform(inst, s1, vals) {
            return false;
        }
        if let Some(&Operand::Scalar { code, .. }) = v.src.get(2)
            && code != NULL
            && inst
                .read(Reg::S(code))
                .is_some_and(|m| cmps.get(m.0 as usize).copied().unwrap_or(false))
        {
            return true;
        }
        let mask_ok = if iname == "v_cndmask_b32" {
            inst.writes
                .first()
                .and_then(|&(_, dst)| st.select_masks.get(&dst).copied())
        } else {
            match v.src.get(2) {
                Some(&Operand::Scalar { code, .. }) if code == NULL => Some(Bool::FALSE),
                Some(&Operand::Scalar { code, .. }) => inst
                    .read(Reg::S(code))
                    .and_then(|m| st.masks.get(&m).copied()),
                _ => None,
            }
        };
        return mask_ok.is_some_and(|m| bool_is_uniform(st, m, vals, cmps));
    }
    if pure_valu {
        return valu_reads_uniform(inst, vals);
    }
    inst.reads
        .iter()
        .all(|&(_, v)| vals.get(v.0 as usize).copied().unwrap_or(false))
}

fn is_unguarded_lane_cmp(st: &Structured, b: usize, ii: usize, inst: &Inst) -> bool {
    st.guards[b][ii].is_none()
        && inst.name().starts_with("v_cmp_")
        && matches!(
            &inst.fields,
            Fields::Valu(v) if v.dpp.is_none() && v.sdwa.is_none()
        )
}

/// Computes which IR blocks are thread-group uniform (all active threads arrive
/// together) and which IR values hold group-uniform scalars.
pub(super) fn compute_uniformity(
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
    let mut lane_cmp_uniform = vec![false; max_v];
    for &b in &order {
        for (ii, inst) in p.blocks[b].insts.iter().enumerate() {
            if is_unguarded_lane_cmp(st, b, ii, inst) {
                for &(_, w) in &inst.writes {
                    lane_cmp_uniform[w.0 as usize] = true;
                }
            }
        }
    }

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
                if is_unguarded_lane_cmp(st, b, ii, inst) {
                    let cmp_ok = blk_ok && valu_reads_uniform(inst, &val_uniform);
                    for &(_, w) in &inst.writes {
                        if lane_cmp_uniform[w.0 as usize] && !cmp_ok {
                            lane_cmp_uniform[w.0 as usize] = false;
                            changed = true;
                        }
                    }
                }
                let reads_ok = blk_ok
                    && lane_independent_reads_ok(
                        st,
                        b,
                        ii,
                        inst,
                        &val_uniform,
                        &lane_cmp_uniform,
                    );
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

/// Propagates lane-uniformity within blocks from group-uniform `val_uniform`
/// through unguarded, lane-independent operations. Unlike `val_uniform`, this
/// propagates inside divergent blocks too (every active lane entering a block
/// computes the same result), while conservatively never promoting across phis.
pub(super) fn compute_active_uniform(
    p: &Program,
    st: &Structured,
    val_uniform: &[bool],
) -> Vec<bool> {
    let mut active = val_uniform.to_vec();
    let mut active_cmp = vec![false; val_uniform.len()];
    let order = crate::ir::reverse_postorder(&p.blocks);
    loop {
        let mut changed = false;
        for &b in &order {
            for (ii, inst) in p.blocks[b].insts.iter().enumerate() {
                if is_unguarded_lane_cmp(st, b, ii, inst) && valu_reads_uniform(inst, &active) {
                    for &(_, w) in &inst.writes {
                        if let Some(slot) = active_cmp.get_mut(w.0 as usize)
                            && !*slot
                        {
                            *slot = true;
                            changed = true;
                        }
                    }
                }
                if lane_independent_reads_ok(st, b, ii, inst, &active, &active_cmp) {
                    for &(_, w) in &inst.writes {
                        if !st.masks.contains_key(&w)
                            && !st.highs.contains_key(&w)
                            && let Some(slot) = active.get_mut(w.0 as usize)
                            && !*slot
                        {
                            *slot = true;
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    active
}

/// A divergent single-sided `if` converted to threadgroup-convergent predication
/// via `tg_any_true(enter_cond)`.
#[derive(Clone)]
pub(super) struct TgBranch {
    pub(super) merge: usize,
    pub(super) region: HashSet<usize>,
}

/// Threadgroup-convergent predication plan computed for `emulating_wave64()`.
#[derive(Default)]
pub(super) struct ConvergentPlan {
    /// Blocks reached synchronously by every thread in the thread group.
    pub(super) convergent: Vec<bool>,
    /// Branches inside predicated regions whose condition is still threadgroup-uniform.
    pub(super) tg_uniform_branches: HashSet<usize>,
    /// Single-sided divergent `if`s converted to `tg_any_true` convergent branches.
    pub(super) tg_branches: HashMap<usize, TgBranch>,
    /// Per block: enclosing `tg_branches` ordered outer-to-inner by header RPO.
    envs: Vec<Vec<usize>>,
}

impl ConvergentPlan {
    pub(super) fn pred_header_for(&self, b: usize) -> Option<usize> {
        self.envs.get(b).and_then(|e| e.last().copied())
    }

    /// Returns enclosing `tg_branch` headers (outer-to-inner) exited on edge `(from, to)`.
    pub(super) fn closed_on_edge(&self, from: usize, to: usize) -> Vec<usize> {
        let Some(from_env) = self.envs.get(from) else {
            return Vec::new();
        };
        let to_env = self.envs.get(to).map(Vec::as_slice).unwrap_or(&[]);
        from_env
            .iter()
            .copied()
            .filter(|h| !to_env.contains(h))
            .collect()
    }
}

/// Plans threadgroup-convergent predication (`tg_any_true`) for non-uniform
/// single-sided `if`s and nested threadgroup-uniform loops so cross-lane ops
/// execute with every thread reaching `lds_barrier` in lockstep.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan_convergent_regions(
    p: &Program,
    st: &Structured,
    uniform: &[bool],
    val_uniform: &[bool],
    active_uniform: &[bool],
    spills: &HashMap<(usize, usize), Moved>,
    resolved_reads: &HashMap<Site, Spill>,
    m0_waterfall_reads: &HashSet<Value>,
    res_waterfall_reads: &[Value],
    sections: &HashMap<usize, HashMap<usize, Section>>,
    cross_block_masks: &HashMap<Option<usize>, Vec<Value>>,
    def_block: &HashMap<Value, usize>,
    enabled: bool,
    need_lds_barriers: bool,
) -> ConvergentPlan {
    let n = p.blocks.len();
    if n == 0 || !enabled {
        return ConvergentPlan {
            convergent: uniform.to_vec(),
            ..ConvergentPlan::default()
        };
    }

    let mut needs_conv = vec![false; n];
    let mut has_s_barrier = vec![false; n];
    let mark_mask = |needs_conv: &mut [bool], v: Value, fallback: usize| {
        let lo = st.highs.get(&v).copied().unwrap_or(v);
        if st.masks.contains_key(&lo) {
            let target = def_block.get(&lo).copied().unwrap_or(fallback);
            if target < n {
                needs_conv[target] = true;
            }
        }
    };
    for (b, blk) in p.blocks.iter().enumerate() {
        if sections.contains_key(&b) {
            needs_conv[b] = true;
        }
        for (i, inst) in blk.insts.iter().enumerate() {
            let iname = inst.name();
            if iname == "s_barrier" {
                has_s_barrier[b] = true;
            }
            if iname == "v_readfirstlane_b32"
                && !inst.writes.first().is_some_and(|w| {
                    res_waterfall_reads.contains(&w.1) || m0_waterfall_reads.contains(&w.1)
                })
                && matches!(
                    &inst.fields,
                    Fields::Valu(v)
                        if matches!(
                            v.src.first(),
                            Some(Operand::Vector { reg, .. })
                                if inst.read(Reg::V(*reg)).is_some_and(|val| !active_uniform[val.0 as usize])
                        )
                )
            {
                needs_conv[b] = true;
            }
            if iname == "v_readlane_b32"
                && !spills.contains_key(&(b, i))
                && !resolved_reads.contains_key(&(b, i))
            {
                needs_conv[b] = true;
            }
            if iname == "ds_bpermute_b32" || iname == "v_permlanex16_b32" {
                needs_conv[b] = true;
            }
            if need_lds_barriers
                && is_lds_mem(inst)
                && iname != "ds_nop"
                && iname != "ds_swizzle_b32"
            {
                needs_conv[b] = true;
            }
            if reads_mask_as_data(st, inst) {
                for &(_, r) in &inst.reads {
                    mark_mask(&mut needs_conv, r, b);
                }
            }
            if let Some(g) = st.guards[b][i] {
                for a in st.bools.support(g) {
                    if let Atom::Opaque(v) = a {
                        mark_mask(&mut needs_conv, v, b);
                    }
                }
            }
        }
        if let LaneExit::Branch { when, .. } = st.exits[b] {
            for a in st.bools.support(when) {
                if let Atom::Opaque(v) = a {
                    mark_mask(&mut needs_conv, v, b);
                }
            }
        }
    }
    for los in cross_block_masks.values() {
        for &lo in los {
            let db = def_block.get(&lo).copied().unwrap_or(0);
            if db < n && !uniform[db] {
                needs_conv[db] = true;
            }
        }
    }

    if (0..n).all(|b| uniform[b] || !needs_conv[b]) {
        return ConvergentPlan {
            convergent: uniform.to_vec(),
            ..ConvergentPlan::default()
        };
    }

    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut rpo = vec![usize::MAX; n];
    let mut fwd = vec![false; n];
    for (idx, &b) in order.iter().enumerate() {
        rpo[b] = idx;
        fwd[b] = true;
    }
    let mut succs = vec![Vec::new(); n];
    let mut fwd_preds = vec![Vec::new(); n];
    for &b in &order {
        match st.exits[b] {
            LaneExit::Jump(t) => {
                succs[b].push(t);
                if fwd[t] {
                    fwd_preds[t].push(b);
                }
            }
            LaneExit::Branch {
                taken, not_taken, ..
            } => {
                succs[b].push(taken);
                fwd_preds[taken].push(b);
                if not_taken != taken {
                    succs[b].push(not_taken);
                    fwd_preds[not_taken].push(b);
                }
            }
            LaneExit::End => succs[b].push(n),
            LaneExit::Indirect => {}
        }
    }
    let (rev_idx, ipdom) = compute_postdoms(n, &succs, &fwd);
    let can_exit: Vec<bool> = (0..n).map(|b| fwd[b] && rev_idx[b] != usize::MAX).collect();

    // Mark blocks participating in any natural loop so unpredicated divergent
    // regions containing loops never keep their post-dominators convergent.
    let mut in_any_loop = vec![false; n];
    for &h in &order {
        let back: Vec<usize> = fwd_preds[h]
            .iter()
            .copied()
            .filter(|&q| rpo[h] <= rpo[q])
            .collect();
        if back.is_empty() {
            continue;
        }
        in_any_loop[h] = true;
        let mut stack = back;
        while let Some(u) = stack.pop() {
            if !in_any_loop[u] {
                in_any_loop[u] = true;
                stack.extend(fwd_preds[u].iter().copied());
            }
        }
    }

    // Identify single-entry, single-exit forward single-sided `if` candidates.
    let mut cand_ifs: Vec<Option<(usize, HashSet<usize>)>> = vec![None; n];
    for &h in &order {
        let LaneExit::Branch {
            taken, not_taken, ..
        } = st.exits[h]
        else {
            continue;
        };
        let merge = ipdom[h];
        if merge >= n || !can_exit[merge] || taken == not_taken {
            continue;
        }
        let body_entry = if taken == merge {
            not_taken
        } else if not_taken == merge {
            taken
        } else {
            continue;
        };
        if !fwd[body_entry] || fwd_preds[body_entry] != vec![h] {
            continue;
        }
        let mut region = HashSet::new();
        let mut stack = vec![body_entry];
        let mut valid = true;
        while let Some(u) = stack.pop() {
            if u == merge || !region.insert(u) {
                continue;
            }
            if !can_exit[u] || rpo[u] <= rpo[h] || rpo[u] >= rpo[merge] {
                valid = false;
                break;
            }
            match st.exits[u] {
                LaneExit::Jump(t) => stack.push(t),
                LaneExit::Branch {
                    taken: t,
                    not_taken: f,
                    ..
                } => {
                    stack.push(t);
                    stack.push(f);
                }
                LaneExit::End | LaneExit::Indirect => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid
            || region.is_empty()
            || region.contains(&h)
            || region.contains(&merge)
            || region.iter().any(|&u| {
                u != body_entry && fwd_preds[u].iter().any(|&q| !region.contains(&q))
            })
        {
            continue;
        }
        cand_ifs[h] = Some((merge, region));
    }

    let mut must_reach_conv = vec![false; n];
    {
        let mut stack: Vec<usize> = (0..n)
            .filter(|&b| fwd[b] && (needs_conv[b] || has_s_barrier[b]))
            .collect();
        while let Some(u) = stack.pop() {
            if must_reach_conv[u] {
                continue;
            }
            must_reach_conv[u] = true;
            for &q in &fwd_preds[u] {
                if !must_reach_conv[q] {
                    stack.push(q);
                }
            }
        }
    }

    let predicated: HashSet<usize> = cand_ifs
        .iter()
        .enumerate()
        .filter_map(|(h, c)| {
            let (merge, region) = c.as_ref()?;
            let inner_conv = region
                .iter()
                .any(|&u| !uniform[u] && (needs_conv[u] || has_s_barrier[u]));
            let loop_before_conv = must_reach_conv[*merge]
                && region.iter().any(|&u| !uniform[u] && in_any_loop[u]);
            (inner_conv || loop_before_conv).then_some(h)
        })
        .collect();

    if predicated.is_empty() {
        return ConvergentPlan {
            convergent: uniform.to_vec(),
            ..ConvergentPlan::default()
        };
    }

    let max_v = val_uniform.len();
    let mut phi_blocked = vec![false; n];
    for &h in &predicated {
        if let Some((merge, _)) = &cand_ifs[h] {
            phi_blocked[*merge] = true;
        }
    }

    let io_or_mask_false: HashSet<usize> = {
        let mut s = HashSet::new();
        for &(_, v) in &p.inputs {
            if !val_uniform[v.0 as usize] {
                s.insert(v.0 as usize);
            }
        }
        for &k in st.masks.keys().chain(st.highs.keys()) {
            s.insert(k.0 as usize);
        }
        s
    };
    let mut blocks = can_exit.clone();
    let mut tg_vals: Vec<bool> = (0..max_v).map(|v| !io_or_mask_false.contains(&v)).collect();
    let mut tg_cmps = vec![false; max_v];
    for &b in &order {
        for (ii, inst) in p.blocks[b].insts.iter().enumerate() {
            if is_unguarded_lane_cmp(st, b, ii, inst) {
                for &(_, w) in &inst.writes {
                    tg_cmps[w.0 as usize] = true;
                }
            }
        }
    }

    let mut tg_uniform_branches = HashSet::new();
    loop {
        let mut changed = false;
        for &b in &order {
            let blk_ok = blocks[b];
            for phi in &p.blocks[b].phis {
                let idx = phi.value.0 as usize;
                if !tg_vals[idx] {
                    continue;
                }
                let preds_ok = blk_ok
                    && !phi_blocked[b]
                    && fwd_preds[b].iter().all(|&q| blocks[q]);
                let args_ok = phi.args.iter().all(|a| tg_vals[a.0 as usize]);
                if !(preds_ok && args_ok) {
                    tg_vals[idx] = false;
                    changed = true;
                }
            }
            for (ii, inst) in p.blocks[b].insts.iter().enumerate() {
                if is_unguarded_lane_cmp(st, b, ii, inst) {
                    let cmp_ok = blk_ok && valu_reads_uniform(inst, &tg_vals);
                    for &(_, w) in &inst.writes {
                        if tg_cmps[w.0 as usize] && !cmp_ok {
                            tg_cmps[w.0 as usize] = false;
                            changed = true;
                        }
                    }
                }
                let reads_ok = blk_ok
                    && lane_independent_reads_ok(st, b, ii, inst, &tg_vals, &tg_cmps);
                for &(_, w) in &inst.writes {
                    if tg_vals[w.0 as usize] && !reads_ok {
                        tg_vals[w.0 as usize] = false;
                        changed = true;
                    }
                }
            }
        }

        let mut next_blocks = can_exit.clone();
        tg_uniform_branches.clear();
        for &b in &order {
            if !can_exit[b] {
                continue;
            }
            let LaneExit::Branch {
                when,
                taken,
                not_taken,
            } = st.exits[b]
            else {
                continue;
            };
            let branch_uniform = blocks[b]
                && st.bools.support(when).iter().all(|a| match *a {
                    Atom::Uniform(v) => tg_vals[v.0 as usize],
                    _ => false,
                });
            if branch_uniform {
                tg_uniform_branches.insert(b);
                continue;
            }
            if predicated.contains(&b) {
                continue;
            }
            let raw_stop = ipdom[b];
            // Check whether the divergent region between `b` and `raw_stop` is
            // strictly acyclic and loop-free; otherwise lanes can abort or
            // desynchronise across iterations, so convergence never recovers at `raw_stop`.
            let mut acyclic = raw_stop < n && !in_any_loop[b];
            let mut inner = Vec::new();
            let mut seen = HashSet::new();
            let mut stack = vec![taken, not_taken];
            while let Some(x) = stack.pop() {
                if x == raw_stop || !seen.insert(x) {
                    continue;
                }
                if x >= n || !can_exit[x] || in_any_loop[x] {
                    acyclic = false;
                    if x >= n {
                        continue;
                    }
                }
                inner.push(x);
                for &s in &succs[x] {
                    if s < n && s != raw_stop && !seen.contains(&s) {
                        stack.push(s);
                    }
                }
            }
            if acyclic {
                phi_blocked[raw_stop] = true;
                for x in inner {
                    next_blocks[x] = false;
                }
            } else {
                // Drop convergence for everything reachable from divergent loop/exit.
                let mut drop_seen = HashSet::new();
                let mut drop_stack = vec![taken, not_taken];
                while let Some(x) = drop_stack.pop() {
                    if x >= n || !drop_seen.insert(x) {
                        continue;
                    }
                    next_blocks[x] = false;
                    for &s in &succs[x] {
                        if s < n {
                            drop_stack.push(s);
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

    // Only keep `tg_branches` whose header itself stayed convergent.
    let mut tg_branches = HashMap::new();
    for h in predicated {
        if blocks[h]
            && let Some((merge, region)) = cand_ifs[h].take()
        {
            tg_branches.insert(h, TgBranch { merge, region });
        }
    }
    let mut convergent: Vec<bool> = (0..n).map(|b| uniform[b] || blocks[b]).collect();
    // Never downgrade originally `uniform` blocks.
    for (b, &u) in uniform.iter().enumerate() {
        if u {
            convergent[b] = true;
        }
    }

    let mut ordered_headers: Vec<usize> = tg_branches.keys().copied().collect();
    ordered_headers.sort_by_key(|&h| rpo[h]);
    let mut envs = vec![Vec::new(); n];
    for (b, env) in envs.iter_mut().enumerate() {
        for &h in &ordered_headers {
            if tg_branches[&h].region.contains(&b) {
                env.push(h);
            }
        }
    }

    ConvergentPlan {
        convergent,
        tg_uniform_branches,
        tg_branches,
        envs,
    }
}

/// Detects `v_readfirstlane_b32` results used solely to drive an M0-relative
/// VGPR-indexing waterfall loop (`s_mov_b32 m0, s` for `v_movrel{s,d,sd}_b32`
/// paired with `v_cmp_eq`/`v_cmpx_eq` against the same index VGPR).
pub(super) fn m0_waterfall(p: &Program) -> (HashSet<Value>, HashSet<(usize, usize)>) {
    let mut readers: HashMap<Value, Vec<(usize, usize)>> = HashMap::new();
    let mut phi_users: HashSet<Value> = HashSet::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for phi in &block.phis {
            phi_users.extend(phi.args.iter().copied());
        }
        for (i, inst) in block.insts.iter().enumerate() {
            for &(_, v) in &inst.reads {
                readers.entry(v).or_default().push((b, i));
            }
        }
    }
    let m0_only_movrel = |inst: &Inst| -> bool {
        let Some(m0_val) = inst.write(Reg::S(crate::ir::M0)) else {
            return false;
        };
        if phi_users.contains(&m0_val) {
            return false;
        }
        let Some(uses) = readers.get(&m0_val) else {
            return false;
        };
        !uses.is_empty()
            && uses.iter().all(|&(ub, ui)| {
                let u = &p.blocks[ub].insts[ui];
                matches!(
                    u.name(),
                    "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32"
                ) && u.read(Reg::S(crate::ir::M0)) == Some(m0_val)
            })
    };
    let cmp_matches = |inst: &Inst, r: Value, vsrc: Value| -> bool {
        let Fields::Valu(vf) = &inst.fields else {
            return false;
        };
        if !matches!(
            inst.name(),
            "v_cmp_eq_u32" | "v_cmp_eq_i32" | "v_cmpx_eq_u32" | "v_cmpx_eq_i32"
        ) || vf.abs != 0
            || vf.neg != 0
            || vf.sdwa.is_some()
            || vf.dpp.is_some()
        {
            return false;
        }
        let (Some(a), Some(b)) = (vf.src.first(), vf.src.get(1)) else {
            return false;
        };
        let is_pair = |first: &Operand, own: &Operand| -> bool {
            matches!(
                (first, own),
                (&Operand::Scalar { code, .. }, &Operand::Vector { reg, .. })
                    if code != NULL
                        && inst.read(Reg::S(code)) == Some(r)
                        && inst.read(Reg::V(reg)) == Some(vsrc)
            )
        };
        is_pair(a, b) || is_pair(b, a)
    };

    let mut reads = HashSet::new();
    let mut compares = HashSet::new();
    for block in &p.blocks {
        for inst in &block.insts {
            if inst.name() != "v_readfirstlane_b32" {
                continue;
            }
            let Fields::Valu(v) = &inst.fields else {
                continue;
            };
            let (Some(&Operand::Vector { reg: vreg, .. }), Some(Operand::Scalar { code, .. })) =
                (v.src.first(), v.dst)
            else {
                continue;
            };
            let (Some(r), Some(vsrc)) = (inst.write(Reg::S(code)), inst.read(Reg::V(vreg))) else {
                continue;
            };
            if phi_users.contains(&r) {
                continue;
            }
            let Some(r_uses) = readers.get(&r) else {
                continue;
            };
            let mut saw_m0 = false;
            let mut cmp_sites = Vec::new();
            let mut all_ok = true;
            for &(ub, ui) in r_uses {
                let u = &p.blocks[ub].insts[ui];
                if u.name() == "s_mov_b32"
                    && matches!(
                        &u.fields,
                        Fields::Sop { dst: Some(Operand::Scalar { code, .. }), src, .. }
                            if *code == crate::ir::M0
                                && matches!(src.first(), Some(&Operand::Scalar { code: sc, .. }) if sc != NULL && u.read(Reg::S(sc)) == Some(r))
                    )
                    && m0_only_movrel(u)
                {
                    saw_m0 = true;
                } else if cmp_matches(u, r, vsrc) {
                    cmp_sites.push((ub, ui));
                } else {
                    all_ok = false;
                    break;
                }
            }
            if all_ok && saw_m0 && !cmp_sites.is_empty() {
                reads.insert(r);
                compares.extend(cmp_sites);
            }
        }
    }
    (reads, compares)
}

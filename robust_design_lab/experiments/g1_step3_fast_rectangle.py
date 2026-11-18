# -*- coding: utf-8 -*-
"""G1 第三步(a)：矩形三场景快速层参与比的"全局场+局部截面"修正与两档验收。

架构（方案获批路线 2：全局场 + 局部截面求解）：
- 全局解（既有独立重网格存档，h=0.01/0.005 × 名义/刻蚀±）给远区积分（r>0.5μm）
  与每条金属底缘线的幅度标定带积分（r∈[5,20]μm）；
- 已验证 2D 局部模型（h=2.5nm 实体氧化层，g1_step2）给近缘窗口（|r|≤0.5μm）
  真能量 U_solid_window 与同带参考积分 I_2D_band；
- 修正：num_near = Σ_lines (I_3D_band/I_2D_band)·U_solid_window。
  幅度比 = 有效缘长（米），量纲自洽；不引入任意经验倍率。

验收：电容 ≤2%、三个参与比各 ≤5%、逐方向对数寿命漂移档位差 ≤0.005。
MA 沿用存档值（近缘权重 0.34%，登记）。输出 results/g1_step3_rectangle_2026-10-23/。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.configs import frozen                      # noqa: E402
from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

OUT = ROOT / "results" / "g1_step3_rectangle_2026-10-23"
EPS0 = 8.8541878128e-12
W_IN = 20.0          # 近缘窗口半径 μm（SA 档位不稳定区达 20μm）
B1, B2 = 20.0, 40.0  # 超额幅度标定带 μm（>20μm 段两档稳定 2%）
BG1, BG2 = 40.0, 90.0  # 背景参考带 μm（带外远尾，两档稳定）
L_H_2D = 0.0025      # 2D 局部模型加密档 μm

LEVELS = {
    "h0p01": ROOT / "results/finite_alignment_v1/rect_reference/h0p01",
    "h0p005": ROOT / "results/finite_alignment_v1/rect_reference/h0p005",
}
SCENES = ("nominal", "etch_plus", "etch_minus")


def mesh_gap_cell(h_corner_um: float, solid: bool, gap_um: float = 30.0,
                  half_len: float = 80.0, r_box: float = 100.0):
    """双金属对向 2D 元：左金属 [−half_len,−g/2]，右金属 [+g/2,+half_len]，
    跨 gap_um 隙；用于结隙向边缘的近缘修正（复用 s2 的求解与采样机械）。"""
    import gmsh
    import numpy as np
    from robust_design_lab.experiments import g1_step2_local_edge as s2

    g2 = gap_um / 2
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("gap_cell")
    try:
        occ = gmsh.model.occ
        pieces = [
            (2, occ.addRectangle(-r_box, -r_box, 0, 2 * r_box, r_box)),
            (2, occ.addRectangle(-r_box, s2.T_METAL, 0, 2 * r_box, r_box - s2.T_METAL)),
        ]
        metals = [occ.addRectangle(-half_len, 0.0, 0, half_len - g2, s2.T_METAL),
                  occ.addRectangle(g2, 0.0, 0, half_len - g2, s2.T_METAL)]
        tools = [(2, m) for m in metals]
        if solid:
            tools += [(2, occ.addRectangle(-r_box, -s2.T_OX, 0, 2 * r_box, s2.T_OX))]
        _, _map = occ.fragment(pieces, tools)
        occ.synchronize()
        tags = {t for _, t in gmsh.model.getEntities(2)}
        com = {t: occ.getCenterOfMass(2, t) for t in tags}
        metal_tags = {t for t in tags
                      if 0 < com[t][1] < s2.T_METAL - s2.TOL
                      and abs(com[t][0]) > g2 + s2.TOL}
        ox_tags = {t for t in tags if -s2.T_OX - s2.TOL < com[t][1] < -s2.TOL} if solid else set()
        sub_tags = {t for t in tags if com[t][1] < (-s2.T_OX if solid else 0.0) - s2.TOL}
        air_tags = tags - metal_tags - ox_tags - sub_tags
        if not air_tags or not sub_tags or len(metal_tags) != 2:
            raise RuntimeError("gap cell region split failed")

        f = gmsh.model.mesh.field
        corner = f.add("Box")
        for k, v in [("VIn", h_corner_um), ("VOut", 0.4), ("XMin", g2 - 1.5),
                     ("XMax", g2 + 1.5), ("YMin", -0.08), ("YMax", s2.T_METAL + 0.08),
                     ("Thickness", 0.5)]:
            f.setNumber(corner, k, v)
        corner2 = f.add("Box")
        for k, v in [("VIn", h_corner_um), ("VOut", 0.4), ("XMin", -g2 - 1.5),
                     ("XMax", -g2 + 1.5), ("YMin", -0.08), ("YMax", s2.T_METAL + 0.08),
                     ("Thickness", 0.5)]:
            f.setNumber(corner2, k, v)
        iface = f.add("Box")
        for k, v in [("VIn", min(0.02, h_corner_um * 4)), ("VOut", 0.4),
                     ("XMin", -r_box), ("XMax", r_box),
                     ("YMin", (-s2.T_OX if solid else 0.0) - 0.01), ("YMax", 0.012),
                     ("Thickness", 0.3)]:
            f.setNumber(iface, k, v)
        mn = f.add("Min")
        f.setNumbers(mn, "FieldsList", [corner, corner2, iface])
        f.setAsBackgroundMesh(mn)
        for k, v in [("Mesh.MeshSizeMax", 2.0), ("Mesh.MeshSizeMin", h_corner_um),
                     ("Mesh.MeshSizeExtendFromBoundary", 0),
                     ("Mesh.MeshSizeFromPoints", 0), ("Mesh.MeshSizeFromCurvature", 0),
                     ("Mesh.Algorithm", 6)]:
            gmsh.option.setNumber(k, v)
        gmsh.model.mesh.generate(2)

        ntags, coords, _ = gmsh.model.mesh.getNodes()
        p = coords.reshape(-1, 3)[:, :2].T.copy()
        lookup = np.full(int(ntags.max()) + 1, -1, dtype=np.int64)
        lookup[ntags.astype(np.int64)] = np.arange(len(ntags))
        tris, mats = [], []
        for tag_list, mat_id in ((air_tags, 0), (sub_tags, 1), (ox_tags, 2)):
            for tag in tag_list:
                types, _, nodes = gmsh.model.mesh.getElements(2, tag)
                arr = lookup[nodes[0].astype(np.int64)].reshape(-1, 3)
                tris.append(arr)
                mats.append(np.full(len(arr), mat_id, dtype=np.int8))
        t = np.vstack(tris).T.copy()
        mat = np.concatenate(mats)
        v = p[:, t]
        area2 = np.abs((v[0, 1] - v[0, 0]) * (v[1, 2] - v[1, 0])
                       - (v[1, 1] - v[1, 0]) * (v[0, 2] - v[0, 0]))
        good = np.isfinite(area2) & (area2 > 1e-16)
        if not np.all(good):
            t, mat = t[:, good], mat[good]
        used = np.unique(t)
        remap = np.full(p.shape[1], -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        return p[:, used], remap[t], mat
    finally:
        gmsh.model.remove()
        if own:
            gmsh.finalize()


def solve_gap_cell(p, t, mat, gap_um=30.0, half_len=80.0, r_box=100.0):
    """双金属对向元：左 +0.5 V，右 −0.5 V（反对称，对应器件模式）。"""
    from skfem import Basis, BilinearForm, ElementTriP2, MeshTri, asm, solve
    from skfem.helpers import dot, grad
    from robust_design_lab.experiments import g1_step2_local_edge as s2

    g2 = gap_um / 2
    mesh = MeshTri(p, t)
    element = ElementTriP2()
    basis = Basis(mesh, element)

    @BilinearForm
    def laplace(u, v, w):
        return dot(grad(u), grad(v))

    def sub(idx):
        return Basis(mesh, element, elements=idx)

    K = asm(laplace, sub(np.nonzero(mat == 0)[0]))
    for m, e in ((1, s2.EPS_SUB), (2, s2.EPS_OX)):
        idx = np.nonzero(mat == m)[0]
        if len(idx):
            K = K + e * asm(laplace, sub(idx))

    r, z = basis.doflocs
    on_metal = (((np.isclose(z, 0.0, atol=s2.TOL) | np.isclose(z, s2.T_METAL, atol=s2.TOL))
                 & (np.abs(r) > g2 + s2.TOL) & (np.abs(r) < half_len - s2.TOL))
                | (np.isclose(np.abs(r), g2, atol=s2.TOL)
                   & (z >= -s2.TOL) & (z <= s2.T_METAL + s2.TOL)))
    left = on_metal & (r < 0)
    right = on_metal & (r > 0)
    on_zero = ((np.isclose(np.abs(r), r_box, atol=1e-6)
                | np.isclose(np.abs(z), r_box, atol=1e-6)) & ~on_metal)
    x = np.zeros(basis.N)
    x[left], x[right] = 0.5, -0.5
    D = np.nonzero(on_metal | on_zero)[0]
    I = basis.complement_dofs(D)
    x[I] = solve(K[I][:, I], -K[I][:, D] @ x[D])
    return basis, mesh, x, K


def gap_reference(gap_um=30.0):
    """间隙元的加权密度参考量（SM=金属下面向缘侧，SA=隙内裸面侧）。

    gap_um 按场景实际间隙传入（刻蚀后 29.5/30.5μm）——整隙匹配对间隙宽度
    敏感，固定 30μm 参考会把刻蚀的几何效应错误放大（p_SA 伪跳）。
    其余同前：interface_density 加权；背景=带积分/带长；单位同前。
    """
    from robust_design_lab.experiments import g1_step2_local_edge as s2

    g2 = gap_um / 2.0
    eps_sub = frozen.EPS_R_SUB
    refs = {}
    for solid in (False, True):
        pm, tm_, mat = mesh_gap_cell(L_H_2D, solid, gap_um=gap_um)
        basis, mesh, x, K = solve_gap_cell(pm, tm_, mat, gap_um=gap_um)
        sampler = s2.Grad2D(basis, mesh, x)
        edges = s2._boundary_edge_integrals(sampler, mesh, mat)
        r_mid = edges["r_mid"]
        d = np.abs(np.abs(r_mid) - g2)          # 到最近对向缘的距离
        # 每边长度（由相邻边中点差估计；首尾用差分）
        rm = edges["r_mid"]
        dr = np.empty_like(rm)
        dr[1:-1] = (rm[2:] - rm[:-2]) / 2
        dr[0] = rm[1] - rm[0]
        dr[-1] = rm[-1] - rm[-2]

        def m_of(lo, hi, metal_side):
            m = (d >= lo) & (d <= hi)
            m &= (np.abs(rm) > g2) if metal_side else (np.abs(rm) < g2)
            return m

        row = {}
        support = {}
        for lab, metal_side in (("SM", True), ("SA", False)):
            d2 = s2.interface_density(lab, edges["et2"], edges["en2"], eps_sub)
            # 背景密度 = 带积分/带长（e2_i 已是沿边积分，勿再乘边长——旧式
            # Σ(e2·dr)/Σdr 对长边双重加权，网格划分依赖）
            m_bg = m_of(B1, B2, metal_side)
            bg = (float(np.sum(d2[m_bg]) / np.sum(dr[m_bg]))
                  if m_bg.any() and np.sum(dr[m_bg]) > 0 else 0.0)
            support[f"{lab}_bg_len_um"] = float(np.sum(dr[m_bg]))
            if lab == "SA":
                # SA 在 30μm 间隙内：到对向缘距离 d≤g2=15μm < 带起点（W≥15）——
                # 原带定义在该区域无支撑（SA_band_raw=0 → amp=0，固体近缘项被抹去，
                # 2026-10-25 复核确认）。改为**整隙匹配**：带 = 裸隙全域 d∈[0,g2]，
                # 两侧均有支撑；间隙内无远场平台，不作背景扣除。
                m_band = m_of(0.0, g2, metal_side)
            else:
                m_band = m_of(B1, B2, metal_side)
            row[f"{lab}_band_raw"] = float(np.sum(d2[m_band])) * 1e6
            support[f"{lab}_band_len_um"] = float(np.sum(dr[m_band]))
            support[f"{lab}_band_nonzero"] = bool(np.sum(dr[m_band]) > 0)
            m_win = m_of(0.0, W_IN, metal_side)
            bg_use = 0.0 if lab == "SA" else bg     # 间隙 SA 无背景平台
            row[f"{lab}_win_excess_d2"] = float(
                np.sum(d2[m_win]) - bg_use * float(np.sum(dr[m_win]))) * 1e6
            row[f"{lab}_bg"] = bg * 1e12
        row["support_audit"] = support
        if solid:
            from skfem import Basis as B2Basis, BilinearForm as BF, ElementTriP2 as E2, asm as asm2
            from skfem.helpers import dot, grad

            @BF
            def laplace(u, v, w):
                return dot(grad(u), grad(v))

            ox = np.nonzero(mat == 2)[0]
            cent = mesh.p[:, mesh.t][:, :, ox].mean(axis=1)
            d_el = np.abs(np.abs(cent[0]) - g2)
            for name, metal_side in (("SM", True), ("SA", False)):
                m_win = (d_el <= W_IN) & ((np.abs(cent[0]) > g2) if metal_side
                                          else (np.abs(cent[0]) < g2))
                m_band = (d_el >= B1) & (d_el <= B2) & ((np.abs(cent[0]) > g2) if metal_side
                                                        else (np.abs(cent[0]) < g2))
                u_win = 0.5 * EPS0 * s2.EPS_OX * float(
                    x @ (asm2(laplace, B2Basis(mesh, E2(), elements=ox[m_win])) @ x))
                u_band = 0.5 * EPS0 * s2.EPS_OX * float(
                    x @ (asm2(laplace, B2Basis(mesh, E2(), elements=ox[m_band])) @ x))
                row[f"U_solid_{name}_window"] = u_win
                if name == "SA":
                    # 间隙 SA：整隙匹配 + 无背景平台 → 窗口全量即匹配量
                    row[f"U_solid_{name}_window_excess"] = u_win
                else:
                    # SM：背景外推 = 带内每微米氧化层能量 × 窗口宽度
                    row[f"U_solid_{name}_window_excess"] = (
                        u_win - (u_band / (B2 - B1)) * W_IN)
        refs["solid" if solid else "thin"] = row
        print(f"[2D gap {'solid' if solid else 'thin '}] {row}", flush=True)
    return refs


def pad_edge_segments(rec):
    segs = []
    for poly in rec.polys:
        xs, ys = poly.exterior.coords.xy
        for i in range(len(xs) - 1):
            segs.append(((xs[i], ys[i]), (xs[i + 1], ys[i + 1])))
    return segs


def dist_to_segments(px, py, segs):
    out = np.empty((len(px), len(segs)))
    for k, ((x0, y0), (x1, y1)) in enumerate(segs):
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        t = np.clip(((px - x0) * dx + (py - y0) * dy) / L2, 0.0, 1.0)
        out[:, k] = np.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
    return out


def two_d_reference():
    """单金属外缘元的加权密度参考量（与 gap_reference 同口径）。"""
    from robust_design_lab.experiments import g1_step2_local_edge as s2

    refs = {}
    r_metal, r_box = -80.0, 100.0
    eps_sub = frozen.EPS_R_SUB
    for solid in (False, True):
        p, t, mat, _ = s2.mesh_local(L_H_2D, solid, r_metal=r_metal, r_box=r_box)
        basis, mesh, x, K, _ = s2.solve_p2(p, t, mat, r_metal=r_metal, r_box=r_box)
        sampler = s2.Grad2D(basis, mesh, x)
        edges = s2._boundary_edge_integrals(sampler, mesh, mat)
        rm = edges["r_mid"]
        dr = np.empty_like(rm)
        dr[1:-1] = (rm[2:] - rm[:-2]) / 2
        dr[0] = rm[1] - rm[0]
        dr[-1] = rm[-1] - rm[-2]

        def m_of(neg, lo, hi):
            sign = -1.0 if neg else 1.0
            return (np.sign(rm) == sign) & (np.abs(rm) >= lo) & (np.abs(rm) <= hi)

        row = {}
        for lab, neg in (("SM", True), ("SA", False)):
            d2 = s2.interface_density(lab, edges["et2"], edges["en2"], eps_sub)
            m_bg = m_of(neg, B1, B2)
            bg = (float(np.sum(d2[m_bg]) / np.sum(dr[m_bg]))
                  if m_bg.any() and np.sum(dr[m_bg]) > 0 else 0.0)
            m_band = m_of(neg, B1, B2)
            m_win = m_of(neg, 0.0, W_IN)
            row[f"{lab}_band_raw"] = float(np.sum(d2[m_band])) * 1e6
            row[f"{lab}_win_excess_d2"] = float(
                np.sum(d2[m_win]) - bg * float(np.sum(dr[m_win]))) * 1e6
            row[f"{lab}_bg"] = bg * 1e12
        if solid:
            from skfem import Basis as B2Basis, BilinearForm as BF, ElementTriP2 as E2, asm as asm2
            from skfem.helpers import dot, grad

            @BF
            def laplace(u, v, w):
                return dot(grad(u), grad(v))

            ox = np.nonzero(mat == 2)[0]
            cent = mesh.p[:, mesh.t][:, :, ox].mean(axis=1)
            for name, neg in (("SM", True), ("SA", False)):
                sign = -1.0 if neg else 1.0
                m_win = (np.sign(cent[0]) == sign) & (np.abs(cent[0]) <= W_IN)
                m_band = (np.sign(cent[0]) == sign) & (np.abs(cent[0]) >= B1) & (np.abs(cent[0]) <= B2)
                u_win = 0.5 * EPS0 * s2.EPS_OX * float(
                    x @ (asm2(laplace, B2Basis(mesh, E2(), elements=ox[m_win])) @ x))
                u_band = 0.5 * EPS0 * s2.EPS_OX * float(
                    x @ (asm2(laplace, B2Basis(mesh, E2(), elements=ox[m_band])) @ x))
                row[f"U_solid_{name}_window"] = u_win
                row[f"U_solid_{name}_window_excess"] = u_win - (u_band / (B2 - B1)) * W_IN
        refs["solid" if solid else "thin"] = row
        print(f"[2D outer {'solid' if solid else 'thin '}] {row}", flush=True)
    return refs


def corrected_participation(fs, surfaces, rec, lj_H, refs):
    """修正参与比 p_SM/p_SA/p_MA（全局远区 + 2D 近缘耦合）。

    口径（docs/reviews/progress_review_verdict_2026-10-24 §四）：
    - 分母 denominator = 2·U·CΣ/Cq，要求分子为 2U_i 约定；
      远区/背景项 = ε₀t·∫d2（无 ½，=2U）✓；固体近缘项 = 2·amp·U_solid_excess
      （U_solid 为真能量 1U，乘 2 进入约定）；
    - d2 = interface_density（SA 分切向/法向：ε_i·Et²+ε_sub²/ε_i·En²；
      SM 全分量 ε_sub²/ε_i 缩放——与 finite_film 验证口径一致）；
    - 3D 背景为面积加权均值（网格划分无关）；幅度比 = 3D/2D 带内密度积分比
      （同为加权口径），单位为有效缘长（米）；
    - p_MA 沿用薄层全量（权重 0.34%，近缘修正不改变结论，登记）。
    """
    from skfem.quadrature import get_quadrature_tri
    from qiskit_metal.analyses.fem.solver import _facet_index
    from robust_design_lab.evaluator.fields import ElectrostaticField
    from robust_design_lab.evaluator.lom import transmon_metrics
    from robust_design_lab.experiments import g1_step2_local_edge as s2

    field = ElectrostaticField(fs)
    mesh = fs.est._mesh
    segs = pad_edge_segments(rec)
    g2 = frozen.JUNCTION_GAP_UM / 2
    seg_is_gap = np.array([abs(abs((a[0] + b[0]) / 2) - g2) < 0.6
                           for a, b in segs])
    tm = transmon_metrics(fs.C_q_fF * 1e-15, lj_H)
    denominator = 2 * field.energy * tm["C_sigma_fF"] / fs.C_q_fF
    eps_sub = frozen.EPS_R_SUB

    def facet_data(label):
        facets = _facet_index(mesh, surfaces[label])
        adjacent = mesh.f2t[:, facets]
        a, b = adjacent
        if label == "MA":
            # MA：金属上面/侧面为网格边界，唯一相邻单元是空气（材料 0）
            first = np.where(fs.package_mesh.material[a] == 0,
                             a, np.where(b >= 0, b, a))
        elif label == "SA":
            # SA：内部材料界面，两侧分别为空气/基片——必须按材料选基片侧
            first = np.where(fs.package_mesh.material[a] == 1, a, b)
        else:
            # SM：金属底面为网格边界，唯一相邻单元是基片
            first = np.where(adjacent[0] >= 0, adjacent[0], adjacent[1])
        expected = 0 if label == "MA" else 1
        if np.any(fs.package_mesh.material[first] != expected):
            raise ValueError(f"{label} 侧材料不符")
        verts = mesh.p[:, mesh.facets[:, facets]]
        area = np.linalg.norm(np.cross((verts[:, 1] - verts[:, 0]).T,
                                       (verts[:, 2] - verts[:, 0]).T), axis=1) / 2.0
        uv, w = get_quadrature_tri(2)
        bary = np.vstack((1.0 - uv.sum(axis=0), uv))
        points = np.einsum("iv,nif->nvf", bary, verts).reshape(3, -1)
        wm2 = (w[:, None] / w.sum() * area[None, :]).ravel() * 1e-6
        E = field.electric_on_elements(points, np.tile(first, len(w))) * 1e3
        return E, wm2, points

    p = {}
    for label in ("SM", "SA"):
        E, wm2, points = facet_data(label)
        et2 = np.sum(E[:2] ** 2, axis=0)
        en2 = E[2] ** 2
        d2 = s2.interface_density(label, et2, en2, eps_sub)
        x_um = (points[0] - fs.origin_xy_mm[0]) * 1e3
        y_um = (points[1] - fs.origin_xy_mm[1]) * 1e3
        dist_all = dist_to_segments(x_um, y_um, segs)
        nearest = np.argmin(dist_all, axis=1)
        dist_min = dist_all.min(axis=1)
        near = dist_min <= W_IN
        ti = frozen.T_INTERFACE_NM[label] * 1e-9

        num_near_total, amps = 0.0, {}
        actual_gap = float(rec.junction.length)          # 刻蚀后实际间隙（μm）
        gap_refs = refs.get("gap_by_um")
        if gap_refs:
            gap_refs = {float(k): v for k, v in gap_refs.items()}
        for cls in ("outer", "gap"):
            if cls == "gap" and gap_refs:
                ref2d = gap_refs[min(gap_refs, key=lambda g: abs(g - actual_gap))]
            else:
                ref2d = refs[cls]
            pts_cls = seg_is_gap[nearest] == (cls == "gap")
            in_gap_sa = (label == "SA" and cls == "gap")   # 仅间隙类的 SA 走整隙匹配
            if in_gap_sa:
                # 匹配带 = 物理间隙内的 SA 点（|x|<g2；仅按最近边距离会把间隙
                # 角落外的裸面点也计入，与 2D 参考区域不对齐 → amp 高估）。
                # 2D 参考 = 整隙 d∈[0,g2]，此处同区域；无背景平台。
                in_gap_phys = (np.abs(x_um) < frozen.JUNCTION_GAP_UM / 2 + 0.5)
                m_band = pts_cls & in_gap_phys
                bg3d = 0.0
            else:
                m_bg = pts_cls & (dist_min >= BG1) & (dist_min <= BG2)
                bg3d = (float(np.dot(wm2[m_bg], d2[m_bg]) / wm2[m_bg].sum())
                        if m_bg.any() else 0.0)
                m_band = pts_cls & (dist_min >= B1) & (dist_min <= B2)
            raw3d = float(np.dot(wm2[m_band], d2[m_band]))
            raw2d = ref2d["solid"][f"{label}_band_raw"]
            amp_x = raw3d / raw2d if raw2d != 0 else 0.0
            a_win = float(wm2[near & pts_cls].sum())
            # 分子（2U 约定）：背景平铺（间隙 SA 无此项）+ 2×固体近缘超额
            num_near_total += (EPS0 * ti * bg3d * a_win
                               + 2.0 * amp_x * ref2d["solid"][f"U_solid_{label}_window_excess"])
            amps[cls] = amp_x
        num_far = EPS0 * ti * float(np.dot(wm2[~near], d2[~near]))
        p["p_" + label] = float((num_far + num_near_total) / denominator)
        p[f"near_fraction_{label}"] = float(
            num_near_total / max(num_far + num_near_total, 1e-300))
        p[f"amp_excess_{label}"] = {k: float(v) for k, v in amps.items()}

    # MA：薄层全量（ε_bulk=1 全分量缩放，finite_film 口径）
    E, wm2, _ = facet_data("MA")
    ti_ma = frozen.T_INTERFACE_NM["MA"] * 1e-9
    eps_i_ma = frozen.INTERFACE_EPS_R["MA"]
    p["p_MA"] = float(EPS0 * ti_ma / eps_i_ma
                      * np.dot(wm2, np.sum(E ** 2, axis=0)) / denominator)
    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    report = dict(provenance=provenance(), frozen_config=frozen.frozen_hash(),
                  window_um=W_IN, band_um=[B1, B2], local_2d_h_nm=L_H_2D * 1e3,
                  scenarios={}, notes=["MA 沿用存档值（权重0.34%）",
                                       "amp=有效缘长(米)，由标定带积分比给出"])
    # 按场景间隙宽度建三份间隙参考（刻蚀后 3D 间隙 29.5/30/30.5μm；
    # 单一 30μm 参考会使整隙幅度匹配对间隙宽度过敏——刻蚀 p_SA 伪跳 55%）
    g0 = frozen.JUNCTION_GAP_UM
    # 刻蚀 δ 使两岛对向缘各移动 δ → 间隙变化 2δ（±0.5μm）
    refs = dict(outer=two_d_reference(),
                gap_by_um={g: gap_reference(gap_um=g)
                           for g in (g0 - 2*frozen.ETCH_UM, g0, g0 + 2*frozen.ETCH_UM)},
                gap=gap_reference())          # 兼容键（30μm）
    report["two_d_reference"] = refs

    from robust_design_lab.experiments.field_audit import load_solution
    from robust_design_lab.evaluator.lom import transmon_metrics

    # 固定 Lj：以 h0p01 名义档配定一次，两档三场景共用（档位比较的前提）
    fs0, _, _ = load_solution(LEVELS["h0p01"] / "nominal")
    from robust_design_lab.evaluator.lom import match_lj
    lj_pinned, _info = match_lj(fs0.C_q_fF * 1e-15, 5.10)
    report["lj_pinned_nH"] = lj_pinned * 1e9
    del fs0

    for level, base in LEVELS.items():
        for scene in SCENES:
            folder = base / scene
            fs, rec, _ = load_solution(folder)
            with np.load(folder / "surfaces.npz", allow_pickle=False) as s:
                surfaces = {k: s[k] for k in s.files}
            pr = corrected_participation(fs, surfaces, rec, lj_pinned, refs)
            tm = transmon_metrics(fs.C_q_fF * 1e-15, lj_pinned)
            row = dict(folder=str(folder), C_q_fF=fs.C_q_fF,
                       p_SM_corrected=pr["p_SM"], p_SA_corrected=pr["p_SA"],
                       p_MA_corrected=pr["p_MA"],
                       near_fraction_SM=pr["near_fraction_SM"],
                       near_fraction_SA=pr["near_fraction_SA"],
                       amp_excess_SM=pr["amp_excess_SM"],
                       amp_excess_SA=pr["amp_excess_SA"],
                       f01_GHz=tm["f01_GHz"], alpha_MHz=tm["alpha_MHz"],
                       lj_nH=lj_pinned * 1e9)
            report["scenarios"].setdefault(level, {})[scene] = row
            print(f"[{level}/{scene}] C_q={fs.C_q_fF:.3f} p_SM={pr['p_SM']:.4e} "
                  f"p_SA={pr['p_SA']:.4e} p_MA={pr['p_MA']:.4e} "
                  f"nearSM={pr['near_fraction_SM']:.3f} "
                  f"ampSM_o={pr['amp_excess_SM']['outer']*1e3:.2f}mm "
                  f"ampSM_g={pr['amp_excess_SM']['gap']*1e3:.2f}mm", flush=True)

    def t1_rate(row):
        return 2 * np.pi * row["f01_GHz"] * 1e9 * (
            row["p_SM_corrected"] * frozen.TAN_DELTA["SM"]
            + row["p_SA_corrected"] * frozen.TAN_DELTA["SA"]
            + row["p_MA_corrected"] * frozen.TAN_DELTA["MA"])

    crit = {}
    for scene in SCENES:
        a = report["scenarios"]["h0p01"][scene]
        b = report["scenarios"]["h0p005"][scene]
        crit[scene] = dict(
            C_rel_change=abs(b["C_q_fF"] / a["C_q_fF"] - 1),
            p_SM_rel_change=abs(b["p_SM_corrected"] / a["p_SM_corrected"] - 1),
            p_SA_rel_change=abs(b["p_SA_corrected"] / a["p_SA_corrected"] - 1),
            p_MA_rel_change=abs(b["p_MA_corrected"] / a["p_MA_corrected"] - 1),
        )
    for level in LEVELS:
        rep = report["scenarios"][level]
        nom_rate = t1_rate(rep["nominal"])
        rep["nominal"]["T1_us_corrected"] = 1e6 / nom_rate
        for scene in ("etch_plus", "etch_minus"):
            rate = t1_rate(rep[scene])
            rep[scene]["T1_us_corrected"] = 1e6 / rate
            rep[scene]["signed_log_drift_T1"] = float(np.log(nom_rate / rate))
    drift_diff = {s: abs(report["scenarios"]["h0p005"][s]["signed_log_drift_T1"]
                         - report["scenarios"]["h0p01"][s]["signed_log_drift_T1"])
                  for s in ("etch_plus", "etch_minus")}
    report["criteria"] = dict(per_scene=crit, drift_level_diff=drift_diff,
                              thresholds=dict(C=0.02, p_each=0.05, drift=0.005))
    report["criteria_passed"] = bool(
        all(c["C_rel_change"] <= 0.02 and c["p_SM_rel_change"] <= 0.05
            and c["p_SA_rel_change"] <= 0.05 and c["p_MA_rel_change"] <= 0.05
            for c in crit.values())
        and all(d <= 0.005 for d in drift_diff.values()))
    write_json(OUT / "step3_fast_rectangle.json", report)
    print(f"criteria passed = {report['criteria_passed']} → "
          f"{OUT/'step3_fast_rectangle.json'} [{time.perf_counter()-t0:.0f}s]")


if __name__ == "__main__":
    main()

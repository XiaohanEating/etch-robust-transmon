# -*- coding: utf-8 -*-
"""G1 第二步：直金属底边局部截面模型（方案 2026-10-23 第二步）。

2D 局部坐标 (r, z)，单位 μm。金属 r∈[−R_BOX,0], z∈[0,0.2]，PEC V=1；
氧化层（实体模型）z∈[−0.003,0] 覆盖全表面（金属下=SM 段，裸露面=SA 段）；
衬底 z<−0.003（ε=11.45）；空气其余。外框接地。薄层模型 = 无氧化层网格，
用与正式评价器相同的薄层公式后处理。

产出：
1. 楔形指数：log|E|~log r 斜率 → λ−1，对照解析 0.5255946；
2. I(R)=∫|E|²dr 的 R^{2λ−1} 增长（R=0.5→5 μm）；
3. U_thin（薄层公式）与 U_solid（实体氧化层能量）的 SM/SA 分段收敛序列；
4. 能量闭合：∫ε|∇φ|² 与电极边界功一致；P2 梯度采样器对二次函数自检。
交接区归类：氧化层终止于楔点 (0,0)；不引入任何圆角参数。
输出 results/g1_step2_local_edge_2026-10-23/local_edge.json。
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

OUT = ROOT / "results" / "g1_step2_local_edge_2026-10-23"
LAMBDA_ANALYTIC = 0.5255946302684442
R_BOX = 40.0
T_METAL = 0.2
T_OX = 0.003
EPS_SUB, EPS_AIR, EPS_OX = frozen.EPS_R_SUB, 1.0, frozen.INTERFACE_EPS_R["SM"]
EPS0 = 8.8541878128e-12
TOL = 1e-9

THIN_H_UM = [0.02, 0.01, 0.005, 0.0025, 0.00125]
SOLID_H_UM = [0.01, 0.005, 0.0025]


# ── 网格 ────────────────────────────────────────────────────
def mesh_local(h_corner_um: float, solid: bool, r_metal: float = -R_BOX,
               r_box: float = R_BOX):
    """2D 局部模型。默认几何与第二步发布结果一致（金属到墙）；
    r_metal/r_box 可把金属条缩短、域扩大，避免金属触墙的伪奇异（第三步耦合用）。
    """
    import gmsh

    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("local_edge")
    try:
        occ = gmsh.model.occ
        # 显式分解（不依赖线压印切开面）：衬底条 + 空气 L 形的三个矩形 + 金属
        pieces = [
            (2, occ.addRectangle(-r_box, -r_box, 0, 2 * r_box, r_box)),        # 衬底 z<0
            (2, occ.addRectangle(-r_box, T_METAL, 0, 2 * r_box, r_box - T_METAL)),  # 金属上方
            (2, occ.addRectangle(0.0, 0.0, 0, r_box, T_METAL)),                # 金属右侧同高
        ]
        if r_box + r_metal > 2 * TOL:      # 金属不到墙时补左端条
            pieces.append((2, occ.addRectangle(-r_box, 0.0, 0, r_box + r_metal, T_METAL)))
        metal = occ.addRectangle(r_metal, 0.0, 0, -r_metal, T_METAL)
        tools = [(2, metal)]
        if solid:
            tools.insert(0, (2, occ.addRectangle(-r_box, -T_OX, 0, 2 * r_box, T_OX)))
        _, mapping = occ.fragment(pieces, tools)
        occ.synchronize()

        tags = {t for _, t in gmsh.model.getEntities(2)}
        com = {t: occ.getCenterOfMass(2, t) for t in tags}
        metal_tags = {t for t in tags
                      if 0 < com[t][1] < T_METAL - TOL
                      and r_metal + TOL < com[t][0] < -TOL}
        ox_tags = {t for t in tags if -T_OX - TOL < com[t][1] < -TOL} if solid else set()
        z_sub_top = -T_OX if solid else 0.0
        sub_tags = {t for t in tags if com[t][1] < z_sub_top - TOL}
        air_tags = tags - metal_tags - ox_tags - sub_tags
        if not air_tags or not sub_tags:
            raise RuntimeError(f"region split failed: air={len(air_tags)} "
                               f"sub={len(sub_tags)} metal={len(metal_tags)}")

        f = gmsh.model.mesh.field
        corner = f.add("Box")
        for k, v in [("VIn", h_corner_um), ("VOut", 0.4), ("XMin", -1.5),
                     ("XMax", 1.5), ("YMin", -0.08), ("YMax", T_METAL + 0.08),
                     ("Thickness", 0.5)]:
            f.setNumber(corner, k, v)
        iface = f.add("Box")
        for k, v in [("VIn", min(0.02, h_corner_um * 4)), ("VOut", 0.4),
                     ("XMin", -r_box), ("XMax", r_box),
                     ("YMin", (z_sub_top if solid else 0.0) - 0.01), ("YMax", 0.012),
                     ("Thickness", 0.3)]:
            f.setNumber(iface, k, v)
        mn = f.add("Min")
        f.setNumbers(mn, "FieldsList", [corner, iface])
        f.setAsBackgroundMesh(mn)
        for k, v in [("Mesh.MeshSizeMax", 2.0), ("Mesh.MeshSizeMin", h_corner_um),
                     ("Mesh.MeshSizeExtendFromBoundary", 0),
                     ("Mesh.MeshSizeFromPoints", 0), ("Mesh.MeshSizeFromCurvature", 0),
                     ("Mesh.Algorithm", 6)]:
            gmsh.option.setNumber(k, v)
        gmsh.model.mesh.generate(2)

        ntags, coords, _ = gmsh.model.mesh.getNodes()
        p = coords.reshape(-1, 3)[:, :2].T.copy()      # 平面模型 z≡0，gmsh 仍返回 3 列
        lookup = np.full(int(ntags.max()) + 1, -1, dtype=np.int64)
        lookup[ntags.astype(np.int64)] = np.arange(len(ntags))

        tris, mats = [], []
        n_metal_tri = 0
        for tag_list, mat_id in ((air_tags, 0), (sub_tags, 1), (ox_tags, 2), (metal_tags, 3)):
            for tag in tag_list:
                types, _, nodes = gmsh.model.mesh.getElements(2, tag)
                if len(types) != 1 or int(types[0]) != 2:
                    raise ValueError("expected linear triangles")
                arr = lookup[nodes[0].astype(np.int64)].reshape(-1, 3)
                if mat_id == 3:
                    n_metal_tri += len(arr)          # 金属区不进求解网格（PEC 内部无方程）
                    continue
                tris.append(arr)
                mats.append(np.full(len(arr), mat_id, dtype=np.int8))
        t = np.vstack(tris).T.copy()
        mat = np.concatenate(mats)
        # 丢弃退化三角形（零面积会产生 NaN 使分解崩溃）
        v = p[:, t]
        area2 = np.abs((v[0, 1] - v[0, 0]) * (v[1, 2] - v[1, 0])
                       - (v[1, 1] - v[1, 0]) * (v[0, 2] - v[0, 0]))
        good = np.isfinite(area2) & (area2 > 1e-16)
        if not np.all(good):
            t, mat = t[:, good], mat[good]
        used = np.unique(t)
        remap = np.full(p.shape[1], -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        p, t = p[:, used], remap[t]
        meta = dict(n_tri=int(t.shape[1]), h_corner_um=h_corner_um, solid=solid,
                    n_metal_tri_dropped=n_metal_tri,
                    n_by_mat={int(m): int((mat == m).sum()) for m in np.unique(mat)})
        return p, t, mat, meta
    finally:
        gmsh.model.remove()
        if own:
            gmsh.finalize()


# ── P2 求解 ─────────────────────────────────────────────────
def solve_p2(p, t, mat, r_metal: float = -R_BOX, r_box: float = R_BOX):
    from skfem import Basis, BilinearForm, ElementTriP2, MeshTri, asm, solve
    from skfem.helpers import dot, grad

    mesh = MeshTri(p, t)
    element = ElementTriP2()
    basis = Basis(mesh, element)

    @BilinearForm
    def laplace(u, v, w):
        return dot(grad(u), grad(v))

    def sub(idx):
        return Basis(mesh, element, elements=idx)

    K = asm(laplace, sub(np.nonzero(mat == 0)[0]))
    for m, e in ((1, EPS_SUB), (2, EPS_OX)):
        idx = np.nonzero(mat == m)[0]
        if len(idx):
            K = K + e * asm(laplace, sub(idx))

    r, z = basis.doflocs
    on_one = (((np.isclose(z, 0.0, atol=TOL) | np.isclose(z, T_METAL, atol=TOL)
                | np.isclose(r, 0.0, atol=TOL))
               & (r <= TOL) & (r >= r_metal + 1e-6) & (z >= -TOL) & (z <= T_METAL + TOL)))
    on_zero = ((np.isclose(np.abs(r), r_box, atol=1e-6)
                | np.isclose(np.abs(z), r_box, atol=1e-6)) & ~on_one)
    x = np.zeros(basis.N)
    x[on_one] = 1.0
    D = np.nonzero(on_one | on_zero)[0]
    I = basis.complement_dofs(D)
    x[I] = solve(K[I][:, I], -K[I][:, D] @ x[D])
    energy = float(x @ (K @ x))          # = ∫ ε|∇φ|²；Green 恒等式下应等于电极电量×V
    charge_work = float(np.sum((K @ x)[on_one]) * 1.0)   # 电极电量（V=1）
    return basis, mesh, x, K, dict(energy=energy, electrode_charge=charge_work,
                                   n_dofs=int(basis.N))


# ── 2D P2 梯度采样（含自检）─────────────────────────────────
class Grad2D:
    def __init__(self, basis, mesh, x):
        from scipy.spatial import cKDTree
        self.basis, self.mesh, self.x = basis, mesh, x
        self.p2t = mesh.p[:, mesh.t]                    # (2, 3, ne)
        cent = self.p2t.mean(axis=1).T                  # (ne, 2)
        self.tree = cKDTree(cent)
        self.nodal = basis.nodal_dofs[0]
        self.t2dof = basis.facet_dofs[0][mesh.t2f].T   # (ne,3) 每元边 dof（t2f 为 (3,ne)）
        # 局部边次序（自适应识别）：(0,1),(1,2),(2,0)
        v = self.p2t
        e = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 1], v[:, 0] - v[:, 2]], axis=1)
        self.edge_pairs = ((0, 1), (1, 2), (2, 0))
        self.grads = self._grad_basis()
        self._selftest()

    def _grad_basis(self):
        """每元每局部基函数的常梯度（线性映射）。v[k] 是第 k 行坐标（全体单元）。"""
        v = self.p2t
        v0, v1, v2 = v[:, 0], v[:, 1], v[:, 2]         # 各顶点坐标 (2, ne)
        det = (v1[0] - v0[0]) * (v2[1] - v0[1]) \
            - (v1[1] - v0[1]) * (v2[0] - v0[0])
        inv = np.stack([
            np.stack([(v2[1] - v0[1]) / det, -(v2[0] - v0[0]) / det], axis=1),
            np.stack([-(v1[1] - v0[1]) / det, (v1[0] - v0[0]) / det], axis=1),
        ], axis=1)                                     # inv[:, k] = ∇λ_{k+1}
        return np.concatenate([-inv.sum(axis=1)[:, None, :], inv], axis=1)  # (ne,3,2)

    def _selftest(self):
        """f=x²+xy 的 P2 插值（顶点/边中点直接赋值），梯度应为 (2x+y, x)——
        以此识别 skfem 的边 dof 局部次序。"""
        basis, mesh = self.basis, self.mesh

        def f(pt):
            return pt[0] ** 2 + pt[1] * pt[0]

        f_dof = np.zeros(basis.N)
        f_dof[self.nodal] = f(mesh.p)
        mids = mesh.p[:, mesh.facets].mean(axis=1)
        f_dof[basis.facet_dofs[0]] = f(mids)

        probe = np.array([[0.03, -0.011, 0.07, -0.2], [0.004, 0.05, -0.06, 0.1]])
        expect = np.stack([2 * probe[0] + probe[1], probe[0]], axis=0)
        import itertools
        base = ((0, 1), (1, 2), (2, 0))
        for order in itertools.permutations(base):
            g = self._eval(f_dof, probe, order)
            if np.allclose(g, expect, atol=1e-9):
                self.edge_order = order
                return
        raise RuntimeError("P2 边 dof 次序自检失败（六种排列均不匹配）")

    def gradient_on_elements(self, elements, points, edge_order=None):
        """在指定单元内求 P2 梯度（点必须落在该单元内；无定位）。"""
        order = self.edge_order if edge_order is None else edge_order
        pts = np.asarray(points, dtype=float)          # (2, n)
        n = pts.shape[1]
        best = np.asarray(elements, dtype=np.int64)
        v0 = self.p2t[:, 0, best]
        e1 = self.p2t[:, 1, best] - v0
        e2 = self.p2t[:, 2, best] - v0
        det = e1[0] * e2[1] - e1[1] * e2[0]
        if np.any(np.abs(det) < 1e-30):
            raise ValueError("退化单元")
        rhs = pts - v0
        lam1 = (e2[1] * rhs[0] - e2[0] * rhs[1]) / det
        lam2 = (-e1[1] * rhs[0] + e1[0] * rhs[1]) / det
        lam_best = np.stack([1.0 - lam1 - lam2, lam1, lam2])
        if lam_best.min(axis=0).max(initial=-np.inf) < -1e-6:
            raise ValueError("采样点不在指定单元内")

        grads_sel = self.grads[best]
        nodal_val = self.x[self.nodal[self.mesh.t[:, best]]]
        fac = 4.0 * lam_best - 1.0
        g = np.einsum("in,nij->nj", nodal_val * fac, grads_sel)
        for k, (a, b) in enumerate(order):
            val = self.x[self.t2dof[best, k]]
            shape = 4.0 * (lam_best[a][:, None] * grads_sel[:, b, :]
                           + lam_best[b][:, None] * grads_sel[:, a, :])
            g += val[:, None] * shape
        return g.T

    def _eval(self, dof_values, points, edge_order):
        pts = np.asarray(points, dtype=float)          # (2, n)
        n = pts.shape[1]
        _, cand = self.tree.query(pts.T, k=8)
        best = np.full(n, -1, dtype=np.int64)
        best_score = np.full(n, -np.inf)
        lam_best = np.zeros((3, n))
        for c in cand.T:                               # c: (n,) 候选单元
            v0 = self.p2t[:, 0, c]                     # (2, n)
            e1 = self.p2t[:, 1, c] - v0
            e2 = self.p2t[:, 2, c] - v0
            det = e1[0] * e2[1] - e1[1] * e2[0]
            det = np.where(np.abs(det) < 1e-30, np.nan, det)
            rhs = pts - v0
            lam1 = (e2[1] * rhs[0] - e2[0] * rhs[1]) / det
            lam2 = (-e1[1] * rhs[0] + e1[0] * rhs[1]) / det
            lam = np.stack([1.0 - lam1 - lam2, lam1, lam2])
            score = np.nan_to_num(lam.min(axis=0), nan=-np.inf)
            better = score > best_score
            best[better] = c[better]
            lam_best[:, better] = lam[:, better]
        if np.any(best < 0):
            raise ValueError("采样点未落在任何单元内")

        grads_sel = self.grads[best]                   # (n, 3, 2)
        nodal_val = dof_values[self.nodal[self.mesh.t[:, best]]]   # (3, n)
        fac = 4.0 * lam_best - 1.0                     # (3, n)
        g = np.einsum("in,nij->nj", nodal_val * fac, grads_sel)
        for k, (a, b) in enumerate(edge_order):
            val = dof_values[self.t2dof[best, k]]      # (n,)
            shape = 4.0 * (lam_best[a][:, None] * grads_sel[:, b, :]
                           + lam_best[b][:, None] * grads_sel[:, a, :])
            g += val[:, None] * shape
        return g.T                                     # ∇（梯度；电场由 electric 取负）

    def electric(self, points):
        return -self._eval(self.x, points, self.edge_order)


# ── 分析 ────────────────────────────────────────────────────
def _boundary_edge_integrals(sampler, mesh, mat):
    """z=0 线上逐边切向/法向分量的高斯积分（边界单元迹，单侧极限）。

    SM 段（r<0）：金属底面，取下方介质单元；SA 段（r>0）：内部材料界面，
    取基片侧单元。返回 dict(r_mid, et2, en2)——et2=Er²（面内切向），
    en2=Ez²（面法向），各为 ∫·ds 的边积分（μm 制）。
    """
    from numpy.polynomial.legendre import leggauss

    idx = np.nonzero(np.all(np.isclose(mesh.p[1, mesh.facets], 0.0,
                                       atol=1e-12), axis=0))[0]
    xi, w = leggauss(3)
    t = 0.5 * (xi + 1.0)
    rows_r, rows_t, rows_n = [], [], []
    for f in idx:
        nodes = mesh.facets[:, f]
        r0, r1 = mesh.p[0, nodes]
        adj = mesh.f2t[:, f]
        cands = [e for e in adj if e >= 0]
        below = [e for e in cands
                 if mesh.p[1, mesh.t[:, e]].mean() < -1e-12]
        if not below:
            continue
        elem = below[0]
        rr = r0 + (r1 - r0) * t
        E = -sampler.gradient_on_elements(np.full(len(t), elem),
                                          np.vstack([rr, np.zeros_like(rr)]))
        fac = abs(r1 - r0) / 2
        rows_r.append(0.5 * (r0 + r1))
        rows_t.append(float(np.sum(w * E[0] ** 2) * fac))
        rows_n.append(float(np.sum(w * E[1] ** 2) * fac))
    order = np.argsort(rows_r)
    return dict(r_mid=np.asarray(rows_r)[order],
                et2=np.asarray(rows_t)[order],
                en2=np.asarray(rows_n)[order])


def interface_density(label, et2, en2, eps_side):
    """薄层界面的加权重密度 d2（无 ε₀t 因子）。

    SM（金属下，基片侧，全分量按 ε_side²/ε_i 缩放——GPT 验证口径）；
    SA（裸露面，基片侧）：ε_i·Et² + ε_side²/ε_i·En²（切向/法向分开）。
    两侧等能检验：纯切向两侧同值（Et 连续）；纯法向经 Dn 连续换算后同值。
    """
    eps_i = frozen.INTERFACE_EPS_R[label]
    if label == "SM":
        return eps_side ** 2 / eps_i * (et2 + en2)
    return eps_i * et2 + eps_side ** 2 / eps_i * en2


def analyze(basis, mesh, x, K, mat, sampler: Grad2D, solid: bool, h):
    out = dict(solid=solid, h_corner_um=h)

    # 1) 楔形指数：首层单元中部采样（定位稳健）；拟合区间 [3h, 0.3μm]
    z_s = -(h / 2 if not solid else T_OX / 2)
    rs = np.geomspace(max(3 * h, 0.004), 0.3, 40)
    E = sampler.electric(np.vstack([-rs, np.full_like(rs, z_s)]))
    Emag = np.linalg.norm(E, axis=0)
    keep = Emag > 0
    slope = np.polyfit(np.log(rs[keep]), np.log(Emag[keep]), 1)[0]
    out["lambda_fit"] = float(slope + 1.0)

    # 2/3) 边界边元积分：背景扣除的近缘窗口（局部模型的远场是封装伪影，
    # 不属于边缘物理；窗口 |r|≤0.5 μm，背景取 r∈[−5,−2] μm 平台均值）
    edges = _boundary_edge_integrals(sampler, mesh, mat)
    r_mid = edges["r_mid"]
    e2 = edges["et2"] + edges["en2"]
    sm = r_mid < 0
    sa = r_mid > 0
    r0 = 0.002
    bg_mask = sm & (r_mid <= -2.0) & (r_mid >= -5.0)
    bg = float(np.mean(e2[bg_mask] / np.abs(r_mid[bg_mask]))) if bg_mask.sum() else 0.0
    out["background_e2_per_um"] = bg

    def window_sum(neg=True):
        sign = -1.0 if neg else 1.0
        m = (np.sign(r_mid) == sign) & (np.abs(r_mid) <= 0.5) & (np.abs(r_mid) >= r0)
        width = np.abs(r_mid[sm & (np.abs(r_mid) <= 5.0)])  # noqa: F841
        return float(np.sum(e2[m] - bg * np.abs(r_mid[m])), )

    i_sm = window_sum(True)
    i_sa = window_sum(False)
    out["I_edge_SM_0p5um"] = i_sm
    out["I_edge_SA_0p5um"] = i_sa
    out["U_thin_SM_per_length_near"] = EPS0 * T_OX * EPS_SUB ** 2 / EPS_OX * i_sm
    out["U_thin_SA_per_length_near"] = EPS0 * T_OX * EPS_SUB ** 2 / EPS_OX * i_sa
    inner = float(e2[sm & (r_mid < -r0) & (r_mid > -0.05)].sum())
    outer = float(e2[sm & (r_mid <= -0.05) & (r_mid > -0.5)].sum())
    out["I_growth_50nm_to_500nm"] = float((inner + outer) / max(inner, 1e-300))
    out["I_growth_analytic"] = float((0.5 / 0.05) ** (2 * LAMBDA_ANALYTIC - 1))
    out["n_boundary_edges"] = int(len(edges))

    # 4) 实体氧化层能量（分 SM/SA 段：r<0 / r>0）
    if solid:
        from skfem import Basis, BilinearForm, ElementTriP2, asm
        from skfem.helpers import dot, grad

        @BilinearForm
        def laplace(u, v, w):
            return dot(grad(u), grad(v))

        ox = np.nonzero(mat == 2)[0]
        cent = mesh.p[:, mesh.t][:, :, ox].mean(axis=1)   # (2, n_ox)
        near = np.abs(cent[0]) <= 0.5

        def sub(idx):
            return Basis(mesh, ElementTriP2(), elements=idx)

        K_ox = asm(laplace, sub(ox))
        uKu = float(x @ (K_ox @ x))
        out["U_solid_ox_total_per_length"] = 0.5 * EPS0 * EPS_OX * uKu
        K_left = asm(laplace, sub(ox[(cent[0] < 0) & near]))
        K_right = asm(laplace, sub(ox[(cent[0] > 0) & near]))
        out["U_solid_SM_per_length_near"] = 0.5 * EPS0 * EPS_OX * float(x @ (K_left @ x))
        out["U_solid_SA_per_length_near"] = 0.5 * EPS0 * EPS_OX * float(x @ (K_right @ x))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    report = dict(provenance=provenance(), frozen_config=frozen.frozen_hash(),
                  lambda_analytic=LAMBDA_ANALYTIC, series=[], units="per unit edge length")
    for solid, hs in ((False, THIN_H_UM), (True, SOLID_H_UM)):
        for h in hs:
            t1 = time.perf_counter()
            p, t, mat, meta = mesh_local(h, solid)
            basis, mesh, x, K, info = solve_p2(p, t, mat)
            sampler = Grad2D(basis, mesh, x)
            row = analyze(basis, mesh, x, K, mat, sampler, solid, h)
            row.update(mesh=meta, solver=info,
                       energy_closure_rel=abs(info["energy"] - info["electrode_charge"])
                       / abs(info["energy"]))
            report["series"].append(row)
            lam = row["lambda_fit"]
            print(f"[{'solid' if solid else 'thin '} h={h*1e3:5.1f}nm] λ_fit={lam:.5f} "
                  f"Igrow={row['I_growth_50nm_to_500nm']:.3f}"
                  f"(analytic {row['I_growth_analytic']:.3f}) "
                  f"I_edge_SM={row['I_edge_SM_0p5um']:.4e} "
                  f"U_thin_SM_near={row['U_thin_SM_per_length_near']:.4e} "
                  + (f"U_solid_SM_near={row['U_solid_SM_per_length_near']:.4e}" if solid else "")
                  + f" [{time.perf_counter()-t1:.0f}s]", flush=True)
            write_json(OUT / "local_edge.json", report)
    print(f"step-2 local edge → {OUT/'local_edge.json'} [{time.perf_counter()-t0:.0f}s]")


if __name__ == "__main__":
    main()

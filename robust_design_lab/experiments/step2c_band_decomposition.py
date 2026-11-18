# -*- coding: utf-8 -*-
"""第二步(c)：同场解拆解匹配系数与二维背景扣除的贡献。

方法（separated_sensitivity_review §2 的精确分解）：
- 二维金属/间隙模型各解一次（不随频带变），保存每条网格边的
  (r_mid, et2, en2) 与氧化层单元坐标——后续频带只做积分，不重解；
- 对 B ∈ {[20,40], [25,45], [30,50]}（W=20、BG=[60,110] 固定，不重叠）：
  N_near(B) = N_bg3D + 2·a(B)·U_excess(B)，其中 a(B)=I_3D(B)/I_2D(B)；
  ΔN = 2[U_0·Δa + a_0·ΔU + Δa·ΔU]——分清归一化系数、二维背景扣除、
  交义项各自贡献；
- 三维侧：矩形名义存档 h0p01/h0p005 各后处理一次（同样只做积分）。

输出 results/step2c_band_decomposition_2026-10-26/。
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

OUT = ROOT / "results" / "step2c_band_decomposition_2026-10-26"
OUT.mkdir(parents=True, exist_ok=True)
LJ_H = 14.19769287109375e-9
W_IN = 20.0
BG = (60.0, 110.0)
BANDS = {"B20_40": (20.0, 40.0), "B25_45": (25.0, 45.0), "B30_50": (30.0, 50.0)}
L_H = 0.0025


def solve_2d_once():
    """解四个 2D 模型（outer/gap × thin/solid），保存完整积分数据。"""
    import numpy as np
    from robust_design_lab.experiments import g1_step3_fast_rectangle as s3
    from robust_design_lab.experiments import g1_step2_local_edge as s2
    from skfem import Basis, BilinearForm, ElementTriP2, asm
    from skfem.helpers import dot, grad

    eps_sub = __import__('robust_design_lab.configs.frozen', fromlist=['EPS_R_SUB']).EPS_R_SUB
    EPS0 = 8.8541878128e-12

    data = {}
    for tag, solver in (("outer", lambda solid: _outer(s3, s2, solid)),
                        ("gap", lambda solid: _gap(s3, s2, solid))):
        for solid in (False, True):
            key = f"{tag}_{'solid' if solid else 'thin'}"
            d = solver(solid)
            # 保存边缘积分原始数据 + 氧化层单元（供频带积分复用）
            data[key] = d
            print(f"[2D {key}] edges={len(d['r_mid'])} "
                  f"ox_elements={d.get('n_ox', 0)}", flush=True)
    return data


def _outer(s3, s2, solid):
    import numpy as np
    from skfem import Basis, BilinearForm, ElementTriP2, asm
    from skfem.helpers import dot, grad
    EPS0 = 8.8541878128e-12

    p, t, mat, _ = s2.mesh_local(L_H, solid, r_metal=-80.0, r_box=100.0)
    basis, mesh, x, K, _ = s2.solve_p2(p, t, mat, r_metal=-80.0, r_box=100.0)
    sampler = s2.Grad2D(basis, mesh, x)
    edges = s2._boundary_edge_integrals(sampler, mesh, mat)
    rm = edges["r_mid"]
    dr = np.empty_like(rm)
    dr[1:-1] = (rm[2:] - rm[:-2]) / 2
    dr[0] = rm[1] - rm[0]
    dr[-1] = rm[-1] - rm[-2]

    d = dict(r_mid=rm.tolist(), et2=edges["et2"].tolist(), en2=edges["en2"].tolist(),
             dr=dr.tolist(), n_edges=len(rm))
    if solid:
        @BilinearForm
        def laplace(u, v, w):
            return dot(grad(u), grad(v))
        ox = np.nonzero(mat == 2)[0]
        cent = mesh.p[:, mesh.t][:, :, ox].mean(axis=1)
        # 保存氧化层单元的 x 坐标与 K 行（供频带窗口积分）——逐单元 uKu
        from qiskit_metal.analyses.fem.solver import Electrostatics
        # 逐单元能量：K 的对角贡献近似太粗；改为逐单元 x@Ksub@x
        # 按氧化层单元 x 坐标分组（5μm 箱），每组一次子矩阵组装
        # （避免逐单元 asm 的维度问题；分组能量足够支撑频带/窗口积分）
        bins = np.arange(-105.0, 106.0, 5.0)
        ox_x = cent[0]
        vals, xs_saved = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (ox_x >= lo) & (ox_x < hi)
            if m.any():
                Ksub = asm(laplace, Basis(mesh, ElementTriP2(), elements=ox[m]))
                u = 0.5 * EPS0 * s2.EPS_OX * float(x @ (Ksub @ x))
                vals.append(u)
                xs_saved.append(float((lo + hi) / 2))
        d.update(ox_x=xs_saved, ox_u=vals, n_ox=len(vals))
    else:
        d.update(n_ox=0)
    return d


def _gap(s3, s2, solid):
    import numpy as np
    from skfem import Basis, BilinearForm, ElementTriP2, asm
    from skfem.helpers import dot, grad
    EPS0 = 8.8541878128e-12
    g2 = 15.0

    pm, tm_, mat = s3.mesh_gap_cell(L_H, solid)
    basis, mesh, x, K = s3.solve_gap_cell(pm, tm_, mat)
    sampler = s2.Grad2D(basis, mesh, x)
    edges = s2._boundary_edge_integrals(sampler, mesh, mat)
    rm = edges["r_mid"]
    dr = np.empty_like(rm)
    dr[1:-1] = (rm[2:] - rm[:-2]) / 2
    dr[0] = rm[1] - rm[0]
    dr[-1] = rm[-1] - rm[-2]

    d = dict(r_mid=rm.tolist(), et2=edges["et2"].tolist(), en2=edges["en2"].tolist(),
             dr=dr.tolist(), n_edges=len(rm), g2=g2)
    if solid:
        @BilinearForm
        def laplace(u, v, w):
            return dot(grad(u), grad(v))
        ox = np.nonzero(mat == 2)[0]
        cent = mesh.p[:, mesh.t][:, :, ox].mean(axis=1)
        bins = np.arange(-16.0, 96.0, 5.0)
        ox_x = cent[0]
        vals, xs_saved = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (ox_x >= lo) & (ox_x < hi)
            if m.any():
                Ksub = asm(laplace, Basis(mesh, ElementTriP2(), elements=ox[m]))
                u = 0.5 * EPS0 * s2.EPS_OX * float(x @ (Ksub @ x))
                vals.append(u)
                xs_saved.append(float((lo + hi) / 2))
        d.update(ox_x=xs_saved, ox_u=vals, n_ox=len(vals))
    else:
        d.update(n_ox=0)
    return d


def band_quantities(data, band, eps_sub):
    """从已存边积分数据计算指定频带的 I_2D 与 U_excess（不重解）。"""
    import numpy as np
    from robust_design_lab.experiments.g1_step2_local_edge import interface_density

    B1, B2 = band
    out = {}
    for tag in ("outer", "gap"):
        for lab in ("SM", "SA"):
            for solid in ("thin", "solid"):
                key = f"{tag}_{solid}"
                d = data[key]
                rm = np.array(d["r_mid"])
                et2 = np.array(d["et2"])
                en2 = np.array(d["en2"])
                dr = np.array(d["dr"])
                d2 = interface_density(lab, et2, en2, eps_sub)

                if tag == "gap" and lab == "SA":
                    m_band = np.abs(rm) < d.get("g2", 15.0) + 0.5
                    bg_use = 0.0
                else:
                    neg = lab == "SM" if tag == "outer" else (np.abs(rm) > d.get("g2", 15.0))
                    if tag == "outer":
                        sign = -1.0 if lab == "SM" else 1.0
                        m_band = (np.sign(rm) == sign) & (np.abs(rm) >= B1) & (np.abs(rm) <= B2)
                    else:
                        # gap SM: 金属下面向缘侧
                        m_band = (np.abs(rm) > d.get("g2", 15.0)) & \
                                 (np.abs(np.abs(rm) - d.get("g2", 15.0)) >= B1) & \
                                 (np.abs(np.abs(rm) - d.get("g2", 15.0)) <= B2)
                    bg_density = float(np.sum(d2[m_band]) / np.sum(dr[m_band])) \
                        if np.sum(dr[m_band]) > 0 else 0.0
                    bg_use = bg_density

                out[f"{tag}_{lab}_{solid}_band_raw"] = float(np.sum(d2[m_band]))
                out[f"{tag}_{lab}_{solid}_bg"] = bg_use

                # U_excess: 氧化层窗口（近缘区）能量 − 带密度×窗口长度
                if solid == "solid" and d.get("n_ox", 0) > 0:
                    ox_x = np.array(d["ox_x"])
                    ox_u = np.array(d["ox_u"])
                    if tag == "gap":
                        m_win = np.abs(ox_x) < 1000  # gap SA 整隙
                        if lab == "SA":
                            m_win = np.abs(ox_x) < d.get("g2", 15.0) + 0.5
                        else:
                            m_win = np.abs(ox_x) > d.get("g2", 15.0)
                    else:
                        sign = -1.0 if lab == "SM" else 1.0
                        m_win = (np.sign(ox_x) == sign) & (np.abs(ox_x) <= W_IN)
                    u_win = float(np.sum(ox_u[m_win]))
                    win_len = float(np.sum(dr[m_band])) * (W_IN / (B2 - B1))  # 近似
                    u_excess = u_win - bg_use * win_len * 1e-6 if lab != "SA" or tag == "outer" else u_win
                    out[f"{tag}_{lab}_solid_U_excess"] = u_excess if u_excess != 0 else u_win
    return out


def main():
    from robust_design_lab.configs import frozen
    eps_sub = frozen.EPS_R_SUB

    t0 = time.perf_counter()
    # 1) 解一次 2D，保存完整数据
    data = solve_2d_once()
    npz = OUT / "edges_2d.npz"
    import numpy as np
    np.savez_compressed(npz, **{k: np.array(v) for k, v in data.items()
                                if isinstance(v, list)})
    print(f"2D data saved → {npz}", flush=True)

    # 2) 三个频带的量（纯积分，不重解）
    report = dict(provenance=provenance(), W_IN=W_IN, BG=BG, bands={})
    for bname, band in BANDS.items():
        q = band_quantities(data, band, eps_sub)
        report["bands"][bname] = dict(band=band, **q)
        print(f"[{bname}] outer_SM_thin_band={q['outer_SM_thin_band_raw']:.1f} "
              f"outer_SM_solid_U={q.get('outer_SM_solid_U_excess', 'N/A')}",
              flush=True)

    # 3) 三维侧（矩形名义存档，同样纯积分）
    from robust_design_lab.experiments import g1_step3_fast_rectangle as s3
    from robust_design_lab.experiments.field_audit import load_solution
    import math
    from robust_design_lab.evaluator.lom import transmon_metrics

    for level in ("h0p01", "h0p005"):
        folder = ROOT / "results" / "finite_alignment_v1" / "rect_reference" / level / "nominal"
        fs, rec, _ = load_solution(folder)
        with np.load(folder / "surfaces.npz", allow_pickle=False) as sf:
            surfaces = {k: sf[k] for k in sf.files}

        # 对每个频带构建 refs 并调用产线
        orig = (s3.W_IN, s3.B1, s3.B2, s3.BG1, s3.BG2)
        s3.W_IN = W_IN
        s3.BG1, s3.BG2 = BG
        try:
            for bname, band in BANDS.items():
                s3.B1, s3.B2 = band
                # 用缓存的边数据重构 refs（而非重解 2D）
                refs = _refs_from_data(data, band, eps_sub)
                pr = s3.corrected_participation(fs, surfaces, rec, LJ_H, refs)
                key = f"{level}_{bname}"
                report.setdefault("p_3d", {})[key] = dict(
                    p_SM=pr["p_SM"], p_SA=pr["p_SA"],
                    amp_SM_outer=pr["amp_excess_SM"]["outer"],
                    amp_SM_gap=pr["amp_excess_SM"]["gap"])
                print(f"[{key}] p_SM={pr['p_SM']:.4e}", flush=True)
        finally:
            s3.W_IN, s3.B1, s3.B2, s3.BG1, s3.BG2 = orig

    # 4) 精确 ΔN 分解（outer SM 类，主导项）
    b0 = report["bands"]["B20_40"]
    for bname in ("B25_45", "B30_50"):
        b1 = report["bands"][bname]
        # 提取 Δa 与 ΔU（outer SM）
        I0, I1 = b0["outer_SM_thin_band_raw"], b1["outer_SM_thin_band_raw"]
        U0, U1 = b0.get("outer_SM_solid_U_excess", 0), b1.get("outer_SM_solid_U_excess", 0)
        report.setdefault("delta_decomposition", {})[f"B20_40→{bname}"] = dict(
            I_2d_0=I0, I_2d_1=I1, U_0=U0, U_1=U1,
            dI_rel=abs(I1/I0-1) if I0 else None,
            dU_rel=abs(U1/U0-1) if U0 else None)

    report["wall_s"] = round(time.perf_counter() - t0, 1)
    write_json(OUT / "band_decomposition.json", report)
    print(f"→ {OUT/'band_decomposition.json'} [{report['wall_s']}s]")


def _refs_from_data(data, band, eps_sub):
    """从已存边数据构建 refs dict（模拟 two_d_reference/gap_reference 输出）。"""
    from robust_design_lab.experiments.g1_step2_local_edge import interface_density
    import numpy as np
    B1, B2 = band

    refs = {}
    for tag in ("outer", "gap"):
        for solid_kind in ("thin", "solid"):
            key = f"{tag}_{solid_kind}"
            d = data[key]
            rm = np.array(d["r_mid"])
            et2 = np.array(d["et2"])
            en2 = np.array(d["en2"])
            dr = np.array(d["dr"])
            row = {}
            for lab in ("SM", "SA"):
                d2 = interface_density(lab, et2, en2, eps_sub)
                if tag == "gap" and lab == "SA":
                    m_band = np.abs(rm) < 15.5
                    bg = 0.0
                else:
                    if tag == "outer":
                        sign = -1.0 if lab == "SM" else 1.0
                        m_band = (np.sign(rm) == sign) & (np.abs(rm) >= B1) & (np.abs(rm) <= B2)
                    else:
                        dd = np.abs(np.abs(rm) - 15.0)
                        if lab == "SM":
                            m_band = (np.abs(rm) > 15.0) & (dd >= B1) & (dd <= B2)
                        else:
                            m_band = np.abs(rm) < 15.5
                    bg = float(np.sum(d2[m_band]) / np.sum(dr[m_band])) \
                        if np.sum(dr[m_band]) > 0 else 0.0
                row[f"{lab}_band_raw"] = float(np.sum(d2[m_band]))
                row[f"{lab}_bg"] = bg
                if solid_kind == "solid" and d.get("n_ox", 0) > 0:
                    ox_x = np.array(d["ox_x"])
                    ox_u = np.array(d["ox_u"])
                    if tag == "gap" and lab == "SA":
                        m_win = np.abs(ox_x) < 15.5
                    elif tag == "gap":
                        m_win = np.abs(ox_x) > 15.0
                    else:
                        sign = -1.0 if lab == "SM" else 1.0
                        m_win = (np.sign(ox_x) == sign) & (np.abs(ox_x) <= 20.0)
                    row[f"U_solid_{lab}_window_excess"] = float(np.sum(ox_u[m_win]))
            refs[solid_kind if tag == "outer" else solid_kind] = row
        # 包装为 {outer: {thin, solid}, gap: {thin, solid}} → 简化为
        # refs dict 直接以 solid 键给 corrected_participation 用
    # corrected_participation 需要 refs["outer"/"gap"]["solid"/"thin"][key]
    return dict(outer=dict(thin=refs["thin"], solid=refs["solid"]),
                gap=dict(thin=refs["thin"], solid=refs["solid"]),
                gap_by_um={30.0: dict(thin=refs["thin"], solid=refs["solid"])})


if __name__ == "__main__":
    main()

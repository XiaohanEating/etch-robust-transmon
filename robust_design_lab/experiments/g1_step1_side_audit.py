# -*- coding: utf-8 -*-
"""G1 第一步（方案 g1_diagnosis_next_steps_2026-10-23 第一步）的本地核对。

从既有存档（独立重网格 / 共用网格输运 / HFSS 三类）后处理：
1. SA 薄层积分的取场侧三口径（基片侧=现实现 / 空气侧 / Dn-Et 对称平均）；
2. SA 四边形点上两侧切向 E 与法向 D 的连续性残差分布（定位"残差~1"）；
3. MA 顶面 vs 侧面的贡献分解，SM 按距边缘距离分解（近缘/远区）；
4. 能量归一化等价性（快速层 2U·CΣ/Cq vs HFSS ∫E·D+CjV² 的模态电容互检）；
5. 三类结果的统一粗→细差异清单（带来源哈希）。

不启动新求解，不改评价器与协议。输出 results/g1_step1_side_2026-10-23/。
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.configs import frozen                      # noqa: E402
from robust_design_lab.evaluator.fields import ElectrostaticField  # noqa: E402
from robust_design_lab.evaluator.lom import transmon_metrics       # noqa: E402
from robust_design_lab.experiments.field_audit import load_solution, sha256  # noqa: E402

EPS0 = 8.8541878128e-12
OUT = ROOT / "results" / "g1_step1_side_2026-10-23"

ARCHIVES = {
    "independent_h0p005": {
        "nominal": ROOT / "results/finite_alignment_v1/rect_reference/h0p005/nominal",
        "etch_plus": ROOT / "results/finite_alignment_v1/rect_reference/h0p005/etch_plus",
        "etch_minus": ROOT / "results/finite_alignment_v1/rect_reference/h0p005/etch_minus",
    },
    "transport_h0p005": {
        "nominal": ROOT / "results/finite_alignment_v1/rect_reference/h0p005/nominal",
        "etch_plus": ROOT / "results/finite_alignment_transport_v1/etch_plus",
        "etch_minus": ROOT / "results/finite_alignment_transport_v1/etch_minus",
    },
}


def _interface_quad(fs, surfaces, label):
    """界面四边形点 + 两侧相邻单元（复用 skfem 三角二次规则，与正式实现一致）。"""
    from skfem.quadrature import get_quadrature_tri
    from qiskit_metal.analyses.fem.solver import _facet_index

    mesh = fs.est._mesh
    facets = _facet_index(mesh, surfaces[label])
    adjacent = mesh.f2t[:, facets]
    if np.any(adjacent < 0):
        raise ValueError(label + " must be shared by two dielectric regions")
    a, b = adjacent
    centres = mesh.p[2, mesh.t].mean(axis=0)
    above = centres[a] > centres[b]
    air_elements = np.where(above, a, b)
    sub_elements = np.where(above, b, a)
    verts = mesh.p[:, mesh.facets[:, facets]]
    area = np.linalg.norm(np.cross((verts[:, 1] - verts[:, 0]).T,
                                   (verts[:, 2] - verts[:, 0]).T), axis=1) / 2.0
    uv, weights = get_quadrature_tri(2)
    bary = np.vstack((1.0 - uv.sum(axis=0), uv))
    points = np.einsum("iv,nif->nvf", bary, verts).reshape(3, -1)
    wm2 = (weights[:, None] / weights.sum() * area[None, :]).ravel() * 1e-6
    return dict(points=points, wm2=wm2, air=np.tile(air_elements, len(weights)),
                sub=np.tile(sub_elements, len(weights)),
                face_area_m2=area * 1e-6, n_faces=len(facets),
                face_centroids_mm=verts.mean(axis=1))


def sa_side_variants(fs, surfaces, denominator):
    """SA 三口径 + 逐点连续性残差。返回 dict。"""
    from skfem.quadrature import get_quadrature_tri  # noqa: F401

    q = _interface_quad(fs, surfaces, "SA")
    field = ElectrostaticField(fs)
    e_air = field.electric_on_elements(q["points"], q["air"]) * 1e3      # V/m
    e_sub = field.electric_on_elements(q["points"], q["sub"]) * 1e3
    eps_air, eps_sub = 1.0, frozen.EPS_R_SUB
    eps_i = frozen.INTERFACE_EPS_R["SA"]
    ti = frozen.T_INTERFACE_NM["SA"] * 1e-9

    # 连续性残差（解析上 Et 连续、Dn 连续）
    et_air, et_sub = e_air[:2], e_sub[:2]
    dn_air, dn_sub = eps_air * e_air[2], eps_sub * e_sub[2]
    et_scale = np.maximum(np.linalg.norm(et_air, axis=0), np.linalg.norm(et_sub, axis=0))
    dn_scale = np.maximum(np.abs(dn_air), np.abs(dn_sub))
    rel_et = np.linalg.norm(et_air - et_sub, axis=0) / np.maximum(et_scale, 1e-30)
    rel_dn = np.abs(dn_air - dn_sub) / np.maximum(dn_scale, 1e-30)

    def dens(et2_side, ez2_side, eps_bulk_side):
        # 与正式实现同口径：εi·Et² + εbulk²/εi·Ez²（无量纲乘 E²；Dn 项 = ε0εbulkEz）
        return eps_i * et2_side + eps_bulk_side ** 2 / eps_i * ez2_side

    et2_sub = np.sum(e_sub[:2] ** 2, axis=0)
    et2_air = np.sum(e_air[:2] ** 2, axis=0)
    ez2_sub, ez2_air = e_sub[2] ** 2, e_air[2] ** 2
    d_sub = dens(et2_sub, ez2_sub, eps_sub)
    d_air = dens(et2_air, ez2_air, eps_air)
    p_sub = EPS0 * ti * float(np.dot(q["wm2"], d_sub)) / denominator
    p_air = EPS0 * ti * float(np.dot(q["wm2"], d_air)) / denominator
    p_sym = EPS0 * ti * float(np.dot(q["wm2"], 0.5 * (d_sub + d_air))) / denominator
    return dict(
        p_SA_substrate_side=p_sub, p_SA_air_side=p_air, p_SA_symmetric=p_sym,
        rel_et=dict(median=float(np.median(rel_et)), p95=float(np.percentile(rel_et, 95)),
                    max=float(rel_et.max())),
        rel_dn=dict(median=float(np.median(rel_dn)), p95=float(np.percentile(rel_dn, 95)),
                    max=float(rel_dn.max())),
        weighted_rel_et=float(np.dot(q["wm2"], rel_et) / q["wm2"].sum()),
        weighted_rel_dn=float(np.dot(q["wm2"], rel_dn) / q["wm2"].sum()),
    )


def sm_ma_decomposition(fs, surfaces, denominator, rec):
    """SM/MA 贡献的空间分解：MA 顶面 vs 侧面；SM/MA 近缘 vs 远区。"""
    from skfem.quadrature import get_quadrature_tri
    from qiskit_metal.analyses.fem.solver import _facet_index
    import shapely

    mesh = fs.est._mesh
    field = ElectrostaticField(fs)
    out = {}
    thickness = 200e-6  # mm，存档身份核验已确认
    for label in ("SM", "MA"):
        facets = _facet_index(mesh, surfaces[label])
        adjacent = mesh.f2t[:, facets]
        first = np.where(adjacent[0] >= 0, adjacent[0], adjacent[1])
        expected = 1 if label == "SM" else 0
        # 与正式实现相同的一侧选择：材料校验
        if np.any(fs.package_mesh.material[first] != expected):
            other = np.where(adjacent[0] >= 0, adjacent[1], adjacent[0])
            first = np.where(fs.package_mesh.material[first] == expected, first, other)
        verts = mesh.p[:, mesh.facets[:, facets]]
        area = np.linalg.norm(np.cross((verts[:, 1] - verts[:, 0]).T,
                                       (verts[:, 2] - verts[:, 0]).T), axis=1) / 2.0
        uv, weights = get_quadrature_tri(2)
        bary = np.vstack((1.0 - uv.sum(axis=0), uv))
        points = np.einsum("iv,nif->nvf", bary, verts).reshape(3, -1)
        wm2 = (weights[:, None] / weights.sum() * area[None, :]).ravel() * 1e-6
        E = field.electric_on_elements(points, np.tile(first, len(weights))) * 1e3
        eps_bulk = frozen.EPS_R_SUB if label == "SM" else 1.0
        eps_i = frozen.INTERFACE_EPS_R[label]
        ti = frozen.T_INTERFACE_NM[label] * 1e-9
        dens = eps_bulk ** 2 / eps_i * np.sum(E ** 2, axis=0)

        # 几何分类
        zq = points[2]
        is_top = zq > frozen.SUBSTRATE_MM + thickness * 0.9 if label == "MA" else None
        # 距金属边界距离（组件局部 μm）
        from shapely.geometry import MultiLineString
        boundary = MultiLineString([list(p.exterior.coords) for p in rec.polys])
        x_um = (points[0] - fs.origin_xy_mm[0]) * 1e3
        y_um = (points[1] - fs.origin_xy_mm[1]) * 1e3
        d_edge = shapely.distance(shapely.points(x_um, y_um), boundary)
        masks = {"all": np.ones(len(dens), bool)}
        if label == "MA":
            masks["top"] = is_top
            masks["side"] = ~is_top
        for cut, name in ((5.0, "near_edge_5um"), (20.0, "near_edge_20um")):
            masks[name] = d_edge <= cut
            masks[name.replace("near", "beyond")] = d_edge > cut
        parts = {}
        total = EPS0 * ti * float(np.dot(wm2, dens)) / denominator
        for name, m in masks.items():
            parts[name] = dict(
                fraction=(float(np.dot(wm2[m], dens[m])) / float(np.dot(wm2, dens)))
                if np.dot(wm2, dens) > 0 else 0.0,
                area_fraction=float(wm2[m].sum() / wm2.sum()),
            )
        out[label] = dict(total_participation=total, spatial_fractions=parts)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(scope="G1 step-1 local side/normalization audit; no new solves",
                  frozen_config=frozen.frozen_hash(), archives={}, sources={})

    for class_name, scenes in ARCHIVES.items():
        class_out = {}
        for scene, folder in scenes.items():
            fs, rec, _ = load_solution(folder)
            with np.load(folder / "surfaces.npz", allow_pickle=False) as saved:
                surfaces = {k: saved[k] for k in saved.files}
            field = ElectrostaticField(fs)
            # 分母与正式实现一致：2U·CΣ/Cq
            lj_H = json.loads((folder / "report.json").read_text(encoding="utf-8"))["lj_H"]
            tm = transmon_metrics(fs.C_q_fF * 1e-15, lj_H)
            denominator = 2 * field.energy * tm["C_sigma_fF"] / fs.C_q_fF
            entry = dict(
                geometry_hash=fs.geometry_hash,
                C_q_fF=fs.C_q_fF,
                energy_J=field.energy,
                denominator=denominator,
                sa=sa_side_variants(fs, surfaces, denominator),
                decomposition=sm_ma_decomposition(fs, surfaces, denominator, rec),
            )
            class_out[scene] = entry
            report["sources"][str(folder)] = {
                "solution_sha256": sha256(folder / "solution.npz"),
                "surfaces_sha256": sha256(folder / "surfaces.npz"),
                "report_sha256": sha256(folder / "report.json"),
            }
            print(f"[{class_name}/{scene}] p_SA: sub={entry['sa']['p_SA_substrate_side']:.4e} "
                  f"air={entry['sa']['p_SA_air_side']:.4e} sym={entry['sa']['p_SA_symmetric']:.4e} "
                  f"wRelEt={entry['sa']['weighted_rel_et']:.3f} wRelDn={entry['sa']['weighted_rel_dn']:.3f}",
                  flush=True)
        report["archives"][class_name] = class_out

    # HFSS 归一化互检：模态电容（来自已存 HFSS 报告）vs 快速层 C_q
    hfss_dirs = {
        "nominal_11": ROOT / "results/hfss_finite_film_v1/200nm_refine11/report.json",
        "nominal_13": ROOT / "results/hfss_finite_film_v1/200nm_refine13/report.json",
        "etch_plus_11": ROOT / "results/finite_alignment_v1/hfss/etch_plus_pass11/report.json",
        "etch_plus_13": ROOT / "results/finite_alignment_v1/hfss/etch_plus_pass13/report.json",
        "etch_minus_11": ROOT / "results/finite_alignment_v1/hfss/etch_minus_pass11/report.json",
        "etch_minus_13": ROOT / "results/finite_alignment_v1/hfss/etch_minus_pass13/report.json",
    }
    norm = {}
    for name, path in hfss_dirs.items():
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        rows = r.get("surface_integrals", [])
        per_face = [dict(label=x["label"], object=x["object"],
                         area_m2=x.get("area_m2"),
                         participation=x.get("participation"))
                    for x in rows]
        norm[name] = dict(
            modal_capacitance_fF=r.get("modal_capacitance_fF"),
            bulk_energy_integral=r.get("bulk_electric_energy_integral"),
            junction_integral=r.get("junction_electric_energy_integral"),
            participation=r.get("participation"),
            per_sheet_contributions=per_face,
            source_sha256=sha256(path),
        )
    report["hfss_normalization"] = norm

    out_path = OUT / "side_audit.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"step-1 side audit → {out_path}")


if __name__ == "__main__":
    main()

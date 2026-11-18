"""十个优质候选的 T1 估算与 metal 原生组件等价几何的对称对比。

- 参照：A 族 VLM 冠军的裸岛等价几何（560×178 矩形垫、gap 50——即
  TransmonPocket 调优终选的裸双岛形式），与新形状走完全相同的评价器。
- 评价器：corrected.evaluate_record（2D 快层 + 局部修正参与比），
  T1_us_nominal = 1/(ω·Σ p_i·tanδ_i)，tanδ 全接口 2e-3（保守）。
  已知边界：匹配带不确定度使 SM 绝对值 ±20%；本对比的意义在同一
  管线下的相对排序与量级。
- 漂浮岛结构（q_disk_arcs）评价器不支持（双导体假设），跳过并如实标注。
- Lj 钉死 12.31 nH 与 Q3D 口径一致（lj0_H）。
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance

OUT = ROOT / "results" / "quality_shapes_2026-10-31"
LJ0_H = 12.31e-9


def to_record(name, s, idx=0):
    from robust_design_lab.geometry.record import GeometryRecord
    # 1μm 轻量简化：降边界顶点密度，帮二维网格器处理超椭圆/交指类
    # 复杂边界（对百微米尺度几何无实质影响）
    pl = s["pad_left"].simplify(1.0)
    pr = s["pad_right"].simplify(1.0)
    return GeometryRecord(family="two_pad_free", category=name,
                          category_index=idx, seed=0,
                          pad_left=pl, pad_right=pr,
                          junction=s["junction"], note=s["label"])


def ref_a_champion():
    """A 族 VLM 冠军的裸岛等价几何：560×178 矩形垫、gap 50。"""
    from shapely.geometry import Polygon, LineString, box
    w, h, g = 560.0, 178.0, 50.0
    pl = box(-g / 2 - w, -h / 2, -g / 2, h / 2)
    pr = box(g / 2, -h / 2, g / 2 + w, h / 2)
    return dict(pad_left=Polygon(pl), pad_right=Polygon(pr),
                islands=[], junction=LineString([(-g / 2, 0), (g / 2, 0)]),
                label="A_champion_bare(560x178,g50)")


def main():
    from robust_design_lab.geometry.novel_structures import QUALITY_FAMILIES
    from robust_design_lab.evaluator.corrected import evaluate_record

    jobs = {name: g() for name, g in QUALITY_FAMILIES.items()
            if not g()["islands"]}
    jobs["A_champion_bare"] = ref_a_champion()

    results = []
    for name, s in jobs.items():
        rec = to_record(name, s)
        t0 = time.perf_counter()
        try:
            out = evaluate_record(rec, lj0_H=LJ0_H)
            row = dict(family=name, label=s["label"],
                       feasible=out.get("feasible"),
                       C_q_fF=out.get("C_q_fF"), f01_GHz=out.get("f01_GHz"),
                       alpha_MHz=out.get("alpha_MHz"),
                       p_SM=out.get("p_SM"), p_SA=out.get("p_SA"),
                       p_MA=out.get("p_MA"),
                       T1_us=out.get("T1_us_nominal"),
                       drift_geom=out.get("drift_geom"),
                       wall_s=round(time.perf_counter() - t0, 1))
        except Exception as e:
            row = dict(family=name, failed=f"{type(e).__name__}: {e}"[:200])
        # 打印在 try 外——绝不再让展示层异常吞掉已算好的结果
        def fmt(v, spec=".2e"):
            return format(v, spec) if isinstance(v, (int, float)) else str(v)
        if row.get("failed"):
            print(f"{name:18s} 异常: {row['failed'][:80]}", flush=True)
        elif row.get("T1_us") is not None:
            print(f"{name:18s} Cq={fmt(row['C_q_fF'], '.1f')} "
                  f"f01={fmt(row['f01_GHz'], '.2f')} "
                  f"T1={fmt(row['T1_us'], '.1f')}us "
                  f"p=({fmt(row['p_SM'])},{fmt(row['p_SA'])},"
                  f"{fmt(row['p_MA'])})", flush=True)
        else:
            print(f"{name:18s} 无名义结果 "
                  f"failures={str(out.get('failures'))[:120]}", flush=True)
        results.append(row)
        (OUT / "t1_check.json").write_text(
            json.dumps(dict(provenance=provenance(), LJ0_H=LJ0_H,
                            tan_delta=0.002, results=results),
                       indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
    print("saved:", OUT / "t1_check.json")


if __name__ == "__main__":
    main()

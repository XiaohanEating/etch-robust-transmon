# -*- coding: utf-8 -*-
"""VLM 形状探索：设计规则引导 + Q3D 评估反馈 + 结构化几何变异。

架构（理论实践结合）：
- 给 VLM 当前最优形状的渲染图 + Q3D 实测指标（C_Σ, f01, α, D）
- 给 VLM 文献设计规则（增 C_Σ 降漂移 / 圆弧化 / 多指冗余 / 宽臂）
- VLM 输出结构化参数调整建议（JSON），不是随意坐标
- Q3D 实际评估 VLM 的建议，真实 D 反馈给下一轮
- 基线：随机参数搜索（同预算）

本轮策略：
- 起始形状：A_dual_pad 当前最优 #2（D=0.0042）+ D_interdigit #3（D=0.0008）
- 3 轮迭代 × 3 个提案/轮 × 2 臂（VLM vs 随机）= 12 个 Q3D 场景（+ 基线 6 个）
- VLM 每轮看到：形状图 + 当前指标 + 上轮提案的实测结果
"""

import json
import math
import random
import sys
import time
import hashlib
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

OUT = ROOT / "results" / "vlm_shape_exploration_2026-10-26"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "qwen3-vl-plus"
ROUNDS = 3
PROPOSALS_PER_ROUND = 3

# ── 文献设计规则（写入 VLM 提示） ──────────────────────
DESIGN_RULES = """已知设计规则（来自文献和实测）：
1. 增大 C_Σ 可降低频率漂移灵敏度（δf/f ≈ δC_Σ/(2C_Σ)）。目标 C_Σ ≥ 70 fF。
2. 圆弧边界优于尖角（同心圆 D=0.003 vs 矩形 D=0.006）；均匀曲率分布场。
3. 交指电容（多指冗余）最鲁棒（D=0.001 vs 双岛 D=0.004）；手指侧面交叠。
4. 宽臂优于窄臂（臂宽变化占比小）。十字/星形臂应 ≥50μm。
5. 间隙/pad 尺寸比是关键参数：gap 越大 D 越小，但 C_Σ 也降。需平衡。
6. f01 = √(8·EJ·EC)/h − EC/h；目标 f01 ∈ [4.5, 5.5] GHz, α ∈ [−300,−150] MHz。
当前 Lj = 12.31 nH, Cj = 2 fF（固定）。"""


def render_shape(variant, family, filepath):
    """渲染形状为图片。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from robust_design_lab.geometry.four_shape_families import build_design_for

    design = build_design_for(variant, _comp_name(family))
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    tables = design.qgeometry.tables
    polys = tables.get("poly")
    for _, row in polys.iterrows():
        g, sub = row.geometry, row.get("subtract", False)
        if sub:
            if g.geom_type == "Polygon":
                ax.fill(*g.exterior.xy, color="#D5D5D5", zorder=1)
        else:
            ax.fill(*g.exterior.xy, color="#2196F3", alpha=0.85, zorder=3,
                    edgecolor="#0D47A1", linewidth=0.8)
    tbl = tables.get("junction")
    if tbl is not None:
        for _, row in tbl.iterrows():
            x, y = row.geometry.xy
            ax.plot(x, y, color="#F44336", linewidth=3, zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(filepath, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()


def _comp_name(family):
    return {
        "A_dual_pad": "TransmonPocket",
        "B_cross": "TransmonCross",
        "B_star": "TransmonStar",
        "C_concentric": "TransmonConcentric",
        "D_interdigit": "CorrectIDC",
    }[family]


def make_vlm_prompt(family, current_variant, metrics, history=None):
    """构造 VLM 提示（含设计规则 + 当前指标 + 历史反馈）。"""
    import numpy as np

    text = f"""你是超导量子比特电容几何设计专家。基于以下信息，提出 {PROPOSALS_PER_ROUND} 个参数调整方案。

{DESIGN_RULES}

当前形状族：{family}
当前参数：{json.dumps({k: v for k, v in current_variant.items() if isinstance(v, (str, int, float))}, ensure_ascii=False)}
当前实测指标：C_Σ = {metrics.get('C_sigma_fF', '?'):.1f} fF, f01 = {metrics.get('f01_GHz', '?'):.3f} GHz, α = {metrics.get('alpha_MHz', '?'):.0f} MHz, 漂移 D = {metrics.get('D', '?'):.4f}
"""
    if history:
        text += "\n上一轮提案及实测结果：\n"
        for h in history[-6:]:  # 最近 6 条
            text += f"  {h['proposal']} → C_Σ={h['C_sigma_fF']:.0f}fF f01={h['f01_GHz']:.2f}GHz D={h['D']:.4f}\n"

    text += f"""

请提出 {PROPOSALS_PER_ROUND} 个**具体的参数调整**方案。每个方案只修改 1-2 个参数，方向明确。
参数名和格式必须与当前参数完全一致（如 "pad_width": "350um"）。

只返回 JSON 数组，每个元素是一个参数修改字典：
[{{"param": "参数名", "value": "新值", "reason": "简短理由"}}]
不要返回完整参数集，只返回要修改的参数。"""

    return text


def make_random_proposal(current_variant, family, rng):
    """随机基线：随机调整 1-2 个参数。"""
    from robust_design_lab.geometry.four_shape_families import FAMILIES
    gen = FAMILIES[family][1]
    new_variant = gen(1, rng)[0]
    # 找出与当前不同的参数
    changes = {}
    for k, v in new_variant.items():
        if k in current_variant and current_variant[k] != v and isinstance(v, (str, int)):
            changes[k] = v
    if not changes:
        # 随机挑一个参数微调
        keys = [k for k in current_variant if isinstance(current_variant[k], str) and "um" in current_variant[k]]
        if keys:
            k = random.choice(keys)
            val = current_variant[k].replace("um", "")
            new_val = float(val) * rng.uniform(0.9, 1.1)
            changes[k] = f"{new_val:.0f}um"
    return changes


def eval_variant(family, variant_changes, base_variant):
    """用 Q3D 评估一个参数变体。"""
    from robust_design_lab.experiments.q3d_shape_sweep import (
        run_q3d, freq_from_csigma, perturb, ETCH_DELTA
    )
    from robust_design_lab.geometry.four_shape_families import build_design_for

    # 合并参数
    full = dict(base_variant)
    full.update(variant_changes)
    comp = _comp_name(family)

    try:
        nom = run_q3d(full, comp, family)
        f01, alpha = freq_from_csigma(nom["C_sigma_fF"])

        # 刻蚀
        ep = perturb(full, family, +ETCH_DELTA)
        em = perturb(full, family, -ETCH_DELTA)
        nom_ep = run_q3d(ep, comp, family)
        nom_em = run_q3d(em, comp, family)
        f01_ep, _ = freq_from_csigma(nom_ep["C_sigma_fF"])
        f01_em, _ = freq_from_csigma(nom_em["C_sigma_fF"])

        D = max(abs(math.log(f01_ep / f01)), abs(math.log(f01_em / f01)))
        return dict(
            C_sigma_fF=nom["C_sigma_fF"], f01_GHz=f01, alpha_MHz=alpha,
            D=D, f01_ep=f01_ep, f01_em=f01_em,
            changes=variant_changes,
            ok=True)
    except Exception as e:
        return dict(failed=f"{type(e).__name__}: {e}"[:200], changes=variant_changes,
                    ok=False)


def run_vlm_arm(family, base_variant, base_metrics):
    """VLM 引导的搜索循环。"""
    from robust_design_lab.vlm.aliyun import load_credentials, chat

    creds = load_credentials()
    current_variant = dict(base_variant)
    current_metrics = dict(base_metrics)
    history = []
    all_results = []

    for rnd in range(ROUNDS):
        # 渲染当前形状
        img_path = OUT / f"vlm_{family}_r{rnd}.png"
        render_shape(current_variant, family, img_path)

        # 构造提示
        prompt = make_vlm_prompt(family, current_variant, current_metrics, history)

        # 调用 VLM（带图）
        img_b64 = base64.b64encode(img_path.read_bytes()).decode()
        msg = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{img_b64}"}},
        ]}]
        try:
            r = chat(creds, MODEL, msg, max_tokens=600, timeout=90)
            content = r.get("content", "[]")
            # 解析提案
            import re
            content = re.sub(r"```json?\n?", "", content).replace("```", "").strip()
            proposals = json.loads(content)
            if isinstance(proposals, dict):
                proposals = [proposals]
        except Exception as e:
            print(f"  [VLM r{rnd}] 解析失败: {e}", flush=True)
            proposals = []

        # 评估每个提案
        print(f"  [VLM r{rnd}] {len(proposals)} 个提案", flush=True)
        best_this_round = None
        for pi, prop in enumerate(proposals):
            if not isinstance(prop, dict) or "param" not in prop or "value" not in prop:
                continue
            changes = {prop["param"]: prop["value"]}
            result = eval_variant(family, changes, current_variant)
            result["round"] = rnd
            result["proposal_idx"] = pi
            result["reason"] = prop.get("reason", "")
            all_results.append(result)

            if result.get("ok"):
                history.append(dict(
                    proposal=f"{prop['param']}={prop['value']}",
                    C_sigma_fF=result["C_sigma_fF"],
                    f01_GHz=result["f01_GHz"],
                    alpha_MHz=result["alpha_MHz"],
                    D=result["D"]))
                print(f"    提案{pi}: {prop['param']}={prop['value']} → "
                      f"CΣ={result['C_sigma_fF']:.0f} f01={result['f01_GHz']:.2f} "
                      f"D={result['D']:.4f}", flush=True)
                # 更新最优
                if best_this_round is None or result["D"] < best_this_round["D"]:
                    best_this_round = result
            else:
                print(f"    提案{pi}: 失败 {result.get('failed', '')[:60]}", flush=True)

        # 移动到本轮最优
        if best_this_round and best_this_round["D"] < current_metrics.get("D", 999):
            current_variant.update(best_this_round["changes"])
            current_metrics = {k: best_this_round[k]
                              for k in ("C_sigma_fF", "f01_GHz", "alpha_MHz", "D")}
            print(f"  → 移动到: D={current_metrics['D']:.4f}", flush=True)

    return dict(
        arm="vlm", family=family,
        final_variant=current_variant, final_metrics=current_metrics,
        all_results=all_results)


def run_random_arm(family, base_variant, base_metrics, seed=42):
    """随机搜索基线（同预算）。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    current_variant = dict(base_variant)
    current_metrics = dict(base_metrics)
    all_results = []

    for rnd in range(ROUNDS):
        for pi in range(PROPOSALS_PER_ROUND):
            changes = make_random_proposal(current_variant, family, rng)
            if not changes:
                continue
            result = eval_variant(family, changes, current_variant)
            result["round"] = rnd
            result["proposal_idx"] = pi
            all_results.append(result)
            if result.get("ok"):
                print(f"    [RAND r{rnd} p{pi}] CΣ={result['C_sigma_fF']:.0f} "
                      f"D={result['D']:.4f}", flush=True)
                if result["D"] < current_metrics.get("D", 999):
                    current_variant.update(changes)
                    current_metrics = {k: result[k]
                                       for k in ("C_sigma_fF", "f01_GHz", "alpha_MHz", "D")}

    return dict(
        arm="random", family=family,
        final_variant=current_variant, final_metrics=current_metrics,
        all_results=all_results)


def main():
    import numpy as np
    from robust_design_lab.experiments.q3d_shape_sweep import (
        run_q3d, freq_from_csigma
    )
    from robust_design_lab.geometry.four_shape_families import make_library

    t0 = time.perf_counter()
    lib = make_library(n_per_family=8, seed=99)

    # 起始形状：从已有扫描中选各族的当前最优
    # 简化：用 library 的第 0 个变体做起点
    starts = {
        "A_dual_pad": ("A_dual_pad", lib["A_dual_pad"]["variants"][2]),
        "D_interdigit": ("D_interdigit", lib["D_interdigit"]["variants"][3]),
    }

    results = []
    for family, (fam_key, variant) in starts.items():
        print(f"\n{'='*60}")
        print(f"族: {family}")
        print(f"{'='*60}")

        # 基线评估
        comp = _comp_name(family)
        print("基线评估...", flush=True)
        nom = run_q3d(variant, comp, family)
        f01, alpha = freq_from_csigma(nom["C_sigma_fF"])
        base_metrics = dict(C_sigma_fF=nom["C_sigma_fF"], f01_GHz=f01,
                            alpha_MHz=alpha, D=0.02)  # 初始 D 设高以允许改进
        print(f"  基线: CΣ={nom['C_sigma_fF']:.1f}fF f01={f01:.3f}GHz", flush=True)

        # VLM 臂
        print(f"\nVLM 引导搜索:", flush=True)
        vlm_result = run_vlm_arm(family, dict(variant), base_metrics)
        results.append(vlm_result)
        write_json(OUT / "vlm_exploration.json", dict(
            provenance=provenance(), model=MODEL, rounds=ROUNDS,
            design_rules=DESIGN_RULES, results=results))

        # 随机基线
        print(f"\n随机搜索基线:", flush=True)
        rand_result = run_random_arm(family, dict(variant), base_metrics)
        results.append(rand_result)
        write_json(OUT / "vlm_exploration.json", dict(
            provenance=provenance(), model=MODEL, rounds=ROUNDS,
            design_rules=DESIGN_RULES, results=results))

    # 汇总
    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    for r in results:
        m = r.get("final_metrics", {})
        print(f"{r['arm']:8s} {r['family']:16s} → D={m.get('D', '?'):.4f}  "
              f"CΣ={m.get('C_sigma_fF', '?'):.0f}fF  "
              f"f01={m.get('f01_GHz', '?'):.2f}GHz")
    wall = time.perf_counter() - t0
    print(f"\n总耗时: {wall:.0f}s → {OUT / 'vlm_exploration.json'}")


if __name__ == "__main__":
    main()

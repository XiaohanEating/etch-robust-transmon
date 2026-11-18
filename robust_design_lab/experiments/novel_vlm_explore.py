# -*- coding: utf-8 -*-
"""跨族新结构 VLM 探索：三个本质区别结构族上的设计规则引导搜索。

协议与 strong_baselines 完全对齐：每结构族 3 轮 × 3 提案 = 9 次评估，
每次评估 = 名义 + buffer±0.5μm 刻蚀三场景 Q3D；接受判据 = 实测 D 改进；
终选 = 已评估点最优。VLM 可见：设计规则 + 新结构渲染图 + 实测指标 +
历史反馈 + 原五族最优成绩（跨族参照）。

基线指标来自 novel_first_eval.json（首评实测，无占位值）。
"""

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance, write_json

OUT = ROOT / "results" / "novel_structures_2026-10-30"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "qwen3-vl-plus"
ROUNDS = 3
PROPOSALS = 3

# 各结构的可调参数集（提案将被吸附到此范围）
PARAM_SETS = {
    "hoya_perforated": dict(
        pad_w=(250, 450), pad_h=(200, 400), gap=(40, 100),
        n_holes_x=(2, 5), n_holes_y=(1, 3), hole_r=(15, 60)),
    "floating_bridge": dict(
        pad_w=(250, 450), pad_h=(200, 350), gap=(40, 100), n_isl=(2, 6),
        isl_w=(50, 120), isl_h=(40, 100), bridge_gap=(15, 40)),
    "koch": dict(order=(1, 3), r=(120, 300), gap=(40, 120)),
}

# 原五族最优（跨族参照，写进提示）
FAMILY_BEST = ("A双岛 0.00115 | B十字/星 未测 | C同心 ~0.003 | "
               "D交指 0.00039-0.00088 | 随机/强基线最差 0.0033+")


def current_params(name, struct_label):
    """从 label 反解当前参数（生成器写入 label 的即当前值）。"""
    return struct_label


def make_prompt(name, params, metrics, history):
    import numpy as np  # noqa: F401
    from robust_design_lab.experiments.novel_explore import FAMILY_NOTE
    rng_txt = json.dumps(PARAM_SETS[name])
    hist = ""
    if history:
        hist = "\n历史提案实测：\n" + "\n".join(
            f"  {h['proposal']} → D={h['D']:.5f}" for h in history[-6:])
    return f"""你是超导量子比特电容设计专家。当前是一个与常规结构有本质区别的新型电容结构。

结构：{name}。设计意图与文献规则：
{FAMILY_NOTE}

当前参数：{json.dumps(params)}
当前实测：C_Σ={metrics.get('C_sigma_fF', 0):.1f} fF, f01={metrics.get('f01_GHz', 0):.3f} GHz, 漂移 D={metrics.get('D', 9):.5f}
目标：C_Σ≥70 fF 且 f01∈[4.5,5.5] GHz 前提下最小化 D。
可调参数与范围：{rng_txt}
跨族参照（原五族最优）：{FAMILY_BEST}{hist}

提出 {PROPOSALS} 个参数调整（每个只改 1-2 个参数）。只返回 JSON 数组：
[{{"param": "参数名", "value": 数值, "reason": "简短理由"}}]"""


def apply_changes(name, base_kwargs, changes):
    """把提案合并进生成器参数；越界吸附、整数维取整。"""
    kw = dict(base_kwargs)
    lo_hi = PARAM_SETS[name]
    for k, v in changes.items():
        if k not in lo_hi:
            continue
        lo, hi = lo_hi[k]
        val = float(v)
        kw[k] = int(min(max(round(val), lo), hi)) if k.startswith(("n_", "order")) \
            else float(min(max(val, lo), hi))
    return kw


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return -1


def main():
    import base64
    from robust_design_lab.geometry import novel_structures as ns
    from robust_design_lab.experiments.novel_explore import eval_struct, render
    from robust_design_lab.vlm.aliyun import load_credentials, chat
    import re

    creds = load_credentials()
    first = json.loads((OUT / "novel_first_eval.json").read_text(encoding="utf-8"))
    base_metrics = {r["family"]: r for r in first["results"] if "D" in r}

    # 各结构默认参数（与生成器签名一致）
    import inspect
    defaults = {name: {k: v.default for k, v in
                       inspect.signature(g).parameters.items() if v.default is not inspect.Parameter.empty}
                for name, g in ns.NOVEL_FAMILIES.items()}

    results = []
    # 断点续跑：complete 的结构原样保留；未完成的从其 final_kw 继续
    prior = {}
    prior_path = OUT / "novel_vlm_exploration.json"
    if prior_path.exists():
        try:
            for e in json.loads(prior_path.read_text(encoding="utf-8"))["results"]:
                prior[e["family"]] = e
        except Exception:
            prior = {}

    for name, gen in ns.NOVEL_FAMILIES.items():
        if name not in base_metrics:
            print(f"跳过 {name}（首评失败）")
            continue
        if prior.get(name, {}).get("complete"):
            print(f"跳过 {name}（已完成: D={prior[name]['final_metrics'].get('D')})")
            results.append(prior[name])
            continue
        print(f"\n===== {name} =====", flush=True)
        if name in prior:  # 中断续跑：恢复参数/指标/历史
            kw = dict(prior[name]["final_kw"])
            cur_metrics = dict(prior[name]["final_metrics"])
            all_results = list(prior[name].get("all_results", []))
            history = [dict(proposal=json.dumps(r.get("changes", {}), ensure_ascii=False),
                            D=r["D"]) for r in all_results if "D" in r]
            print(f"  续跑自存档：D={cur_metrics.get('D')}, 已评估 {len(all_results)} 次",
                  flush=True)
        else:
            kw = dict(defaults[name])
            cur_metrics = dict(base_metrics[name])
            all_results = []
            history = []
        cur_struct = gen(**kw)
        start_round = 0
        if name in prior and all_results:
            start_round = max(r.get("round", 0) for r in all_results) + 1
        else:
            render(cur_struct, OUT / f"vlm_{name}_r0.png")
        entry = dict(family=name, final_kw=dict(kw),
                     final_metrics=dict(cur_metrics), all_results=all_results,
                     complete=False)
        results.append(entry)

        for rnd in range(start_round, ROUNDS):
            img = OUT / f"vlm_{name}_r{rnd}.png"
            if not img.exists():
                render(cur_struct, img)  # 上一轮未移动时无新图，现场渲染
            prompt = make_prompt(name, kw, cur_metrics, history)
            msg = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(img.read_bytes()).decode()}},
            ]}]
            print(f"  [rss] chat前 {rss_mb():.0f}MB", flush=True)
            try:
                r = chat(creds, MODEL, msg, max_tokens=600, timeout=90)
                content = re.sub(r"```json?\n?", "", r.get("content", "[]")).replace("```", "").strip()
                proposals = json.loads(content)
                if isinstance(proposals, dict):
                    proposals = [proposals]
            except Exception as e:
                print(f"  [r{rnd}] 解析失败: {e}", flush=True)
                proposals = []

            best = None
            for pi, prop in enumerate(proposals[:PROPOSALS]):
                if not isinstance(prop, dict) or "param" not in prop:
                    continue
                changes = {prop["param"]: prop["value"]}
                kw2 = apply_changes(name, kw, changes)
                if kw2 == kw:
                    continue
                struct = gen(**kw2)
                t0 = time.perf_counter()
                print(f"  [rss] r{rnd}p{pi} 评估前 {rss_mb():.0f}MB", flush=True)
                try:
                    rec = eval_struct(struct,
                                      ns.NOVEL_FAMILIES[name](**kw2)["label"])
                except Exception as e:  # 单次 COM/求解失败不终止整臂
                    rec = dict(label="failed", failed=f"{type(e).__name__}: {e}"[:200],
                               changes=changes)
                rec.update(round=rnd, proposal=pi, changes=changes,
                           reason=prop.get("reason", ""), kw=kw2,
                           eval_s=round(time.perf_counter() - t0, 1))
                all_results.append(rec)
                print(f"  [rss] r{rnd}p{pi} 评估后 {rss_mb():.0f}MB", flush=True)
                if "D" not in rec:
                    print(f"  r{rnd} p{pi} 失败: {rec.get('failed', '')[:60]}", flush=True)
                    continue
                history.append(dict(proposal=json.dumps(changes, ensure_ascii=False),
                                    D=rec["D"]))
                print(f"  r{rnd} p{pi} {changes} → D={rec['D']:.5f} "
                      f"CΣ={rec['C_sigma_fF']:.1f}", flush=True)
                if best is None or rec["D"] < best["D"]:
                    best = rec

            if best and best["D"] < cur_metrics.get("D", 9):
                kw, cur_struct = best["kw"], gen(**best["kw"])
                cur_metrics = dict(best)
                render(cur_struct, OUT / f"vlm_{name}_r{rnd+1}.png")
                print(f"  → 移动到 D={best['D']:.5f}", flush=True)

            # 逐轮落盘（崩溃可查、终态以 complete 标记）
            entry.update(final_kw=dict(kw), final_metrics=dict(cur_metrics),
                         complete=(rnd == ROUNDS - 1))
            write_json(OUT / "novel_vlm_exploration.json",
                       dict(provenance=provenance(), model=MODEL,
                            results=results))

    print("\n===== 汇总 =====")
    for r in results:
        m = r["final_metrics"]
        print(f"{r['family']:18s} D={m.get('D', float('nan')):.5f} "
              f"CΣ={m.get('C_sigma_fF', float('nan')):.1f} "
              f"f01={m.get('f01_GHz', float('nan')):.2f}")


if __name__ == "__main__":
    main()

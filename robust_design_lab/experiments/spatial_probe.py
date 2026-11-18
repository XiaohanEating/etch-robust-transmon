# -*- coding: utf-8 -*-
"""第二步诊断：空间信息配对实验（预选快照，减少新增求解）。

按 docs/plans/new_structures_next_steps_2026-11-04.md 第二步：
- 两个预选状态（环形起点、CMA 编辑冠军），同一文字（含逐段空间表
  spatial_table——所有方法共享同信息），有图臂附空间叠加图；
- 错图臂喂**同一合法拓扑但局部差异不同**的伪状态图（环形+内环 bulge
  段7），检验模型是否察觉图文冲突（跨拓扑图不再作主探针）；
- 简单物理规则基线（最窄间隙段 bulge）同格式记录——检验"简单规则
  是否已足够"的位置层证据（结果层比较留待受控编辑实验）；
- 只记录提案与理由，不做搜索、不产生结论性"视觉增益"声称。

物理层核查（场量可导出性）见 --check-fields：打开已存工程检查
可导出场量清单，不求解。
"""
import argparse
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance
from robust_design_lab.geometry import novel_structures as geo
from robust_design_lab.geometry import boundary_edit as be
from robust_design_lab.geometry import spatial_feedback as sf

OUT = ROOT / "results" / "spatial_probe_2026-11-04"
OUT.mkdir(parents=True, exist_ok=True)
CALLS = OUT / "calls"
CALLS.mkdir(exist_ok=True)
MODEL = "qwen3-vl-plus"

PROMPT = """设计规则：目标 C_Σ∈[70,105]fF 且 f01∈[4.5,5.5]GHz 下最小化刻蚀漂移 D（名义+刻蚀±0.5μm 三场景 max|log(f_δ/f_0)|）。出窗或无效动作罚 10.0。
当前为双岛电容结构（两电极+中间结线；结线上下 50μm 保护圆内不可编辑）。
下面是逐段空间信息表（局部间隙=该段到对置电极最近距离；到地边界=到地面口袋距离；曲率为边界离散曲率均值；对置≈间隙最近的对面段）：
{table}
空间提示：刻蚀使全部边界同步内缩 0.5μm——局部间隙小的对置段，其间隙相对变化大；边界密度高（曲率大/多转折）的区域刻蚀后面积相对损失大；到地边界近的段对地面刻蚀敏感。
可用编辑：op∈bulge(局部法向外凸/内凹)/notch(开圆槽,金属减)/tab(外伸舌,金属增)；island L/R；ring outer/inner(无内环无效)；region 0-7；mag 0.2-1.0；width 0.3-1.0。单次面积变化≤30%，编辑几何范围不得侵入结区保护圆。
提出 3 个编辑动作。只返回 JSON 数组：[{{"op":"bulge","island":"L","ring":"outer","region":3,"mag":0.6,"width":0.5,"reason":"引用空间量说明依据"}}]"""


def build_states():
    """预选状态：环形起点、CMA 编辑冠军（段3 内环 bulge）、伪状态
    （段7 内环 bulge，仅作错图源——同拓扑不同局部差异）。"""
    annulus = geo.q_annulus(r_out=253.9, r_in=121.5, gap=33.8)
    cma_champ = be.apply_edit(annulus, "bulge",
                              be.seg_encode("R", "inner", 3), 0.36, 0.30)
    decoy = be.apply_edit(annulus, "bulge",
                          be.seg_encode("R", "inner", 7), 0.36, 0.30)
    return {
        "annulus_start": annulus,
        "cma_champ": cma_champ,
        "decoy_seg7": decoy,
    }


def render_state(name, struct):
    p = OUT / f"spatial_{name}.png"
    sf.render_spatial(struct, p, title=name)
    return p


def one_call(creds, tag, prompt, img_path):
    from robust_design_lab.vlm.aliyun import chat
    content = [{"type": "text", "text": prompt}]
    img_sha = None
    if img_path is not None:
        img_sha = hashlib.sha256(img_path.read_bytes()).hexdigest()
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + b64}})
    msg = [{"role": "user", "content": content}]
    t0 = time.perf_counter()
    r = chat(creds, MODEL, msg, max_tokens=900, timeout=120)
    raw = r.get("content", "") if isinstance(r, dict) else str(r)
    (CALLS / f"{tag}.json").write_text(json.dumps(
        dict(tag=tag, model=MODEL, prompt=prompt,
             image=str(img_path) if img_path else None,
             sent_image_sha256=img_sha, request=msg, response_raw=r,
             wall_s=round(time.perf_counter() - t0, 1)),
        ensure_ascii=False, default=str), encoding="utf-8")
    return raw


def parse_actions(raw):
    """解析模型回复 → (动作列表, 丢弃清单)。

    严格编码（2026-11-07 复核 P1：旧版 region 越界被静默错译——
    L/inner/region15 经 seg_encode 变成 R/outer/region7；10 项被静默
    丢弃）。规则：三字段路径 region 必须在 [0,8)，越界即拒绝并记录
    原字段；seg 直通必须 [0,32)；op 必须合法。丢弃项含原因返回，
    调用方负责落盘——不静默、不猜测模型原意。"""
    import re
    from robust_design_lab.search import common_protocol as cp
    text = re.sub(r"```json?\n?", "", raw or "").replace("```", "").strip()
    arr = cp.json_loads_lenient(text)
    if isinstance(arr, dict):
        arr = [arr]
    out, dropped = [], []
    for p in arr if isinstance(arr, list) else []:
        if not isinstance(p, dict):
            dropped.append(dict(item=str(p)[:80], reason="非对象"))
            continue
        try:
            if "seg" in p and "island" not in p:
                seg = int(p["seg"])
                be.seg_decode(seg)
                op_i = be.EDIT_OPS.index(str(p["op"]))
            else:
                island = str(p["island"])
                ring = str(p["ring"])
                region = int(p["region"])
                if island not in ("L", "R"):
                    raise ValueError(f"island={island} 非法")
                if ring not in ("outer", "inner"):
                    raise ValueError(f"ring={ring} 非法")
                if not 0 <= region < 8:
                    raise ValueError(
                        f"region={region} 越界 [0,8)——模型可能把全局段号"
                        f"填入 region")
                op_i = be.EDIT_OPS.index(str(p["op"]))
                seg = be.seg_encode(island, ring, region)
            kw = dict(op=float(op_i), seg=float(seg),
                      mag=round(min(max(float(p["mag"]),
                                        be.MAG_RANGE[0]), be.MAG_RANGE[1]), 4),
                      width=round(min(max(float(p["width"]),
                                          be.WIDTH_RANGE[0]),
                                      be.WIDTH_RANGE[1]), 4))
            out.append((kw, str(p.get("reason", ""))))
        except Exception as e:
            dropped.append(dict(
                item=json.dumps({k: p.get(k) for k in
                                 ("op", "island", "ring", "region", "seg")},
                                ensure_ascii=False)[:100],
                reason=f"{type(e).__name__}: {e}"[:120]))
    return out, dropped


SPATIAL_WORDS = ("间隙", "曲率", "对置", "地边界", "gap", "边界密度",
                 "空间", "距离")


def run_probe():
    from robust_design_lab.vlm.aliyun import load_credentials
    creds = load_credentials()
    states = build_states()
    imgs = {n: render_state(n, s) for n, s in states.items()}
    tables = {n: sf.spatial_table(s) for n, s in states.items()}
    rules = {n: sf.simple_rule_action(s) for n, s in states.items()}

    results = []
    for state_name in ("annulus_start", "cma_champ"):
        struct = states[state_name]
        prompt = PROMPT.format(table="\n".join(
            "  " + r for r in tables[state_name]))
        arms = {}
        for arm, img in (("img", imgs[state_name]),
                         ("text", None),
                         ("mismatch", imgs["decoy_seg7"])):
            tag = f"{state_name}_{arm}"
            print(f"[调用] {tag}", flush=True)
            raw = one_call(creds, tag, prompt, img)
            acts, _dropped = parse_actions(raw)
            arms[arm] = dict(
                actions=[dict(kw=a[0], desc=be.action_describe(a[0]),
                              reason=a[1]) for a in acts],
                n_parsed=len(acts),
                reason_spatial=[any(w in r for w in SPATIAL_WORDS)
                                for _, r in acts])
        results.append(dict(
            state=state_name, prompt=prompt,
            table=tables[state_name],
            simple_rule=dict(kw=rules[state_name],
                             desc=be.action_describe(rules[state_name])),
            arms=arms,
            images={k: str(v) for k, v in imgs.items()}))
        (OUT / "spatial_probe.json").write_text(
            json.dumps(dict(provenance=provenance(), results=results),
                       ensure_ascii=False, default=str, indent=1),
            encoding="utf-8")

    # 汇总：三臂提案对比
    for r in results:
        print(f"\n== {r['state']} ==")
        print(f"  简单规则: {r['simple_rule']['desc']}")
        for arm in ("img", "text", "mismatch"):
            a = r["arms"][arm]
            acts = "; ".join(x["desc"] for x in a["actions"][:3])
            n_sp = sum(a["reason_spatial"])
            print(f"  {arm:9s} ({a['n_parsed']}提案, 空间引用{n_sp}): {acts}")


def check_fields():
    """物理层核查：打开最新已存工程查可导出场量（不求解）。

    工程由 run_q3d 统一保存在 q3d_shape_sweep_2026-10-26/（OUT），
    refine2 的 0.1% 工程为其中时间戳最新者。"""
    proj_dir = ROOT / "results" / "q3d_shape_sweep_2026-10-26"
    aedt = sorted(proj_dir.glob("*.aedt"), key=lambda p: p.stat().st_mtime)
    if not aedt:
        print("未找到工程文件")
        return
    from robust_design_lab.experiments.q3d_shape_sweep import get_desktop
    desktop = get_desktop()
    desk = desktop._desktop
    p = aedt[0]
    print(f"打开 {p.name} ...")
    opened = desk.OpenProject(str(p))
    proj_obj = desk.GetActiveProject()
    designs = list(proj_obj.GetTopDesignList())
    out = dict(project=str(p), designs=designs, field_quantities=None)
    try:
        import win32com.client as w32c
        odesign = proj_obj.GetActiveDesign() if designs else None
        odesign = odesign or proj_obj.SetActiveDesign(designs[0])
        mod = odesign.GetModule("FieldsReporter")
        qty = list(mod.GetValidQuantities())
        out["field_quantities"] = qty
        print("FieldsReporter 可用量:", qty[:20])
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:200]
        print("场量读取失败:", out["error"])
    try:
        desk.CloseProject(p.stem)
    except Exception:
        pass
    (OUT / "field_export_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--check-fields", action="store_true")
    args = ap.parse_args()
    if args.check_fields:
        check_fields()
        return
    if args.probe:
        run_probe()
        return
    print("用法: --probe（VLM 空间配对诊断）或 --check-fields（场量核查）")


if __name__ == "__main__":
    main()

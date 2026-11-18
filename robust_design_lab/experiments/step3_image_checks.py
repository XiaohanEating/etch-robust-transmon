# -*- coding: utf-8 -*-
"""第三步：图像定位正控制 + 同状态同文字配对调用（2026-10-25 审阅 §4/§5）。

A. 正控制（排除接口问题）：发送已知答案的图像定位任务——
   结间隙（图像中央）是场最集中区域；另发一次无图版本，验证模型无法凭文字
   猜出位置（负控制）。
B. 同状态配对：同一几何、同一数值摘要、同一提示文字，分别配
   真图 / 错配图（另一几何的三联图）/ 无图；记录实发图像哈希与原始回复。
   仅作敏感性证据（提案是否随图变化），不作视觉有效性结论。
C. 完整提示原文落盘（此前只存键名）。

输出 results/step3_image_checks_2026-10-25/。
"""

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance, write_json  # noqa: E402

OUT = ROOT / "results" / "step3_image_checks_2026-10-25"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "qwen3-vl-plus"


def sha256f(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def call(messages):
    from robust_design_lab.vlm.aliyun import load_credentials, chat
    creds = load_credentials()
    t0 = time.perf_counter()
    r = chat(creds, MODEL, messages, max_tokens=600, timeout=90)
    r["wall_s"] = round(time.perf_counter() - t0, 1)
    return r


def b64(p):
    import base64
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def main():
    from robust_design_lab.experiments.rendering import render_triptych
    from robust_design_lab.geometry import library
    from robust_design_lab.configs import frozen

    W2 = (frozen.FOOTPRINT_UM[0] - frozen.JUNCTION_GAP_UM) / 2
    rec_a = library.rect_record(W2, frozen.FOOTPRINT_UM[1], 0.0)     # 真图几何
    rec_b = library.make_record("ellipse_arc", 0)                     # 错配图几何
    img_a = OUT / "triptych_rect_r0.png"
    img_b = OUT / "triptych_ellipse.png"
    render_triptych(rec_a, img_a)
    render_triptych(rec_b, img_b)
    summary = dict(category=rec_a.category, geometry_hash=rec_a.geometry_hash,
                   C_q_fF=58.18, f01_GHz=5.10, alpha_MHz=-384.8,
                   drift_geom=0.0119)

    report = dict(provenance=provenance(), model=MODEL, calls={})

    # ── A. 图像定位正/负控制 ──────────────────────────────
    q_loc = ("请只回答一个字母：第二幅图（电场图）中最亮的强场集中区域"
             "位于图像的什么位置？A 左岛上方边缘；B 两个岛之间的中央间隙；"
             "C 右岛上方边缘；D 图像四角。只输出字母。")
    m_pos = [{"role": "user", "content": [
        {"type": "text", "text": q_loc},
        {"type": "image_url", "image_url": {"url": b64(img_a)}}]}]
    r_pos = call(m_pos)
    m_neg = [{"role": "user", "content": q_loc}]          # 无图负控制
    r_neg = call(m_neg)
    report["calls"]["localization_with_image"] = dict(
        prompt=q_loc, image_sha256=sha256f(img_a), content=r_pos.get("content"),
        usage=r_pos.get("usage"), wall_s=r_pos["wall_s"])
    report["calls"]["localization_no_image"] = dict(
        prompt=q_loc, content=r_neg.get("content"),
        usage=r_neg.get("usage"), wall_s=r_neg["wall_s"])
    print("[A] 有图:", str(r_pos.get("content"))[:60], flush=True)
    print("[A] 无图:", str(r_neg.get("content"))[:60], flush=True)

    # ── B. 同状态配对：真图 / 错配图 / 无图 ────────────────
    from robust_design_lab.vlm.experiment import propose_message
    base_msg = propose_message(summary)                    # 无图版本（含完整提示）
    full_text = base_msg[0]["content"]

    pairs = {}
    for name, img in (("real", img_a), ("mismatch", img_b), ("none", None)):
        if img is None:
            msg = [{"role": "user", "content": full_text}]
        else:
            msg = [{"role": "user", "content": [
                {"type": "text", "text": full_text},
                {"type": "image_url", "image_url": {"url": b64(img)}}]}]
        r = call(msg)
        pairs[name] = dict(image=str(img) if img else None,
                           image_sha256=sha256f(img) if img else None,
                           content=r.get("content"), request_id=r.get("request_id"),
                           usage=r.get("usage"), wall_s=r["wall_s"])
        print(f"[B/{name}] {str(r.get('content'))[:80]}", flush=True)
        write_json(OUT / "image_checks.json", dict(
            provenance=provenance(), summary=summary,
            full_prompt_text=full_text,
            localization=report["calls"], pairs=pairs))

    # 敏感性汇总（仅证据，非结论）
    contents = {k: v["content"] for k, v in pairs.items()}
    report["pairs"] = pairs
    report["full_prompt_text"] = full_text
    report["sensitivity"] = dict(
        real_vs_mismatch_identical=(contents["real"] == contents["mismatch"]),
        real_vs_none_identical=(contents["real"] == contents["none"]),
        note=("同状态同文字配对；差异=图像输入影响提案的敏感性证据。"
              "不构成视觉有效/无效结论；定位正控制先排除接口问题。"))
    write_json(OUT / "image_checks.json", report)
    print("sensitivity:", report["sensitivity"]["real_vs_mismatch_identical"],
          report["sensitivity"]["real_vs_none_identical"], "→",
          OUT / "image_checks.json")


if __name__ == "__main__":
    main()

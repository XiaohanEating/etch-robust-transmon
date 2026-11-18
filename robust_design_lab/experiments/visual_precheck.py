"""视觉前置检查（用户执行顺序第 5 步后半的前提）。

三条通过判据（对应旧轮"错配图响应同哈希=图像通道未被使用"的教训）：
  P1 真图作答与该结构的几何真值一致；
  P2 错配图作答跟随图片内容（等于图片结构的真值，而非文字主题的）；
  P3 无图作答不给出几何细节（无法判断/泛化描述）。

问题设计为"由图可答、由文字不可猜"：
  Q1 金属轮廓类型（圆环/带手指/变宽度光滑双岛/实心圆盘）
  Q2 是否存在从主体伸出的手指状结构（有/无）
  Q3 金属中是否存在内孔（环形镂空）（有/无）
三个结构的真值组合互不相同，构成可判别编码。
"""
import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from robust_design_lab.runtime.records import provenance

OUT = ROOT / "results" / "visual_precheck_2026-11-02"
OUT.mkdir(parents=True, exist_ok=True)

# 结构真值编码（由生成器定义直接给出，不由模型裁定）
TRUTH = {
    "m_variable_gap": dict(Q1="C", Q2="无", Q3="无"),   # 变宽度光滑双岛/无指/无孔
    "q_annulus": dict(Q1="A", Q2="无", Q3="有"),        # 圆环/无指/有孔
    "q_ellipse_idc": dict(Q1="B", Q2="有", Q3="无"),    # 带手指/有指/无孔
}

QUESTIONS = """只依据图片中的几何（不要依赖上面文字），回答三问，只输出一行
"Q1=X;Q2=有/无;Q3=有/无"：
Q1 两个金属导体的轮廓最接近哪种？A.圆环形 B.椭圆垫且带伸出的手指 C.光滑但宽度渐变的双岛 D.实心圆盘
Q2 是否存在从主体伸出的细长手指状结构？有/无
Q3 金属中是否存在环形的内孔（镂空）？有/无"""


def render_all():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from robust_design_lab.geometry.novel_structures import (
        MAINLINE_FAMILIES, q_ellipse_idc)
    imgs = {}
    figs = dict(m_variable_gap=MAINLINE_FAMILIES["m_variable_gap"],
                q_annulus=q_ellipse_idc, q_ellipse_idc=q_ellipse_idc)
    figs["q_annulus"] = MAINLINE_FAMILIES  # placeholder 修下行
    from robust_design_lab.geometry.novel_structures import q_annulus
    figs = dict(m_variable_gap=MAINLINE_FAMILIES["m_variable_gap"],
                q_annulus=q_annulus, q_ellipse_idc=q_ellipse_idc)
    for name, g in figs.items():
        s = g()
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        for pd_ in (s["pad_left"], s["pad_right"]):
            ax.fill(*pd_.exterior.xy, color="#2196F3", alpha=0.85,
                    edgecolor="#0D47A1", linewidth=0.8, zorder=3)
            for hole in pd_.interiors:
                ax.fill(*hole.xy, color="white", zorder=4)
        x, y = s["junction"].xy
        ax.plot(x, y, color="#F44336", linewidth=2.5, zorder=5)
        ax.set_aspect("equal"); ax.axis("off")
        p = OUT / f"{name}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        imgs[name] = p
    return imgs


def main():
    import re
    from robust_design_lab.vlm.aliyun import load_credentials, chat
    creds = load_credentials()
    imgs = render_all()
    names = list(TRUTH)
    rows = []
    for topic in names:
        for cond in ("true", "mismatch", "noimg"):
            img_struct = topic if cond == "true" else (
                names[(names.index(topic) + 1) % 3] if cond == "mismatch" else None)
            # 主题名匿名化：名字本身不得携带形状线索（首测泄漏教训）
            text = (f"这是一个超导量子比特电容结构（编号 "
                    f"{hashlib.md5(topic.encode()).hexdigest()[:6]}，"
                    f"几何参数保密）。\n{QUESTIONS}")
            content = [{"type": "text", "text": text}]
            if img_struct:
                b64 = base64.b64encode(imgs[img_struct].read_bytes()).decode()
                content.append({"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + b64}})
            try:
                r = chat(creds, "qwen3-vl-plus",
                         [{"role": "user", "content": content}],
                         max_tokens=120, timeout=60)
                ans = r.get("content", "").strip()
            except Exception as e:
                ans = f"ERROR {type(e).__name__}"
            rows.append(dict(topic=topic, cond=cond, image=img_struct,
                             answer=ans))
            print(f"{topic:15s} {cond:9s} ←{str(img_struct):15s} {ans[:60]}",
                  flush=True)
    # 判定
    def parse(a):
        m1 = re.search(r"Q1\s*=\s*([ABCD])", a)
        m2 = re.search(r"Q2\s*=\s*(有|无)", a)
        m3 = re.search(r"Q3\s*=\s*(有|无)", a)
        return (m1.group(1) if m1 else None,
                m2.group(1) if m2 else None,
                m3.group(1) if m3 else None)

    verdict = dict(P1=True, P2=True, P3=True)
    for row in rows:
        q1, q2, q3 = parse(row["answer"])
        ans3 = dict(Q1=q1, Q2=q2, Q3=q3)
        if row["cond"] == "true":
            if ans3 != TRUTH[row["topic"]]:
                verdict["P1"] = False
        elif row["cond"] == "mismatch":
            # 应跟随图片（image 字段的结构），而不是文字主题
            if ans3 != TRUTH[row["image"]]:
                verdict["P2"] = False
        else:
            # P3 本意：无图答案须与主题无互信息（全同固定猜测或拒答，命中≤偶然）
            if q1 is not None and q1 == TRUTH[row["topic"]]["Q1"]:
                verdict["P3"] = False
    print("\n判据:", verdict,
          "→ 全部通过" if all(verdict.values()) else "→ 未通过，视觉通道存疑")
    (OUT / "precheck.json").write_text(
        json.dumps(dict(provenance=provenance(), truth=TRUTH, rows=rows,
                        verdict=verdict), indent=2, ensure_ascii=False),
        encoding="utf-8")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""冻结的实验参数（方案 v2.0 §16"执行前需要冻结的记录"）。

单一事实来源：所有阶段脚本从这里取参数，不得在脚本内散落硬编码。
修改任何数值 = 实验协议变更，必须先落档 docs/decisions.md 并生成新版本结果目录。
材料参数当前来源为文献典型值（标注待核），G1 校准后如调整需回写此文件并记录。
"""

import hashlib
import json

# ── 器件几何契约 ─────────────────────────────────────────────
FOOTPRINT_UM = (430.0, 300.0)   # 双岛外接框 (x, y)，对齐 QIP 2026 文献
JUNCTION_GAP_UM = 30.0          # 结区长（两岛对面边距离）
JUNCTION_WIDTH_UM = 20.0        # 结线绘制宽度
ISLAND_MARGIN_UM = 2.0          # 岛不得超出外接框的余量

# 制造约束（硬）
MIN_LINEWIDTH_UM = 5.0
MIN_GAP_UM = 5.0
MIN_VERTEX_SPACING_UM = 0.5        # 硬下限（防零长度曲线/退化）；2μm 为生成器软目标
JUNCTION_PROTECT_HALF_UM = 20.0   # 结线两侧 |y|<该值的对面边必须保持直边
MIN_AREA_FRACTION = 0.15          # 岛面积下限（占满框岛面积的比例）

# ── 电路与材料 ───────────────────────────────────────────────
LJ0_NH = 12.0                   # 标称结电感
CJ_FF = 2.0                     # 结电容
PHI0 = 2.067833848e-15          # 磁通量子 (Wb)

EPS_R_SUB = 11.45               # 高阻 Si（待核：qiskit-metal 默认值）
SUBSTRATE_MM = 0.5              # 衬底厚度（与 tutorial 4.4x 验证配置一致）

# 界面损耗（待核：文献典型值；G1 敏感性检查后冻结）
TAN_DELTA = {"SM": 2e-3, "MA": 2e-3, "SA": 2e-3}
T_INTERFACE_NM = {"SM": 3.0, "MA": 3.0, "SA": 3.0}

# Explicit oxide assumption, not a measured process parameter.
# Reference: Wenner 2011 supplement. Legacy thickness/tan(delta) stay assumptions.
INTERFACE_EPS_R = {"SM": 10.0, "MA": 10.0, "SA": 10.0}

# 有限厚度修正口径（corrected-finite-v1）的 α 窗：2026-10-24 重冻结
# （旧窗基于零厚度电容量级；有限模型满幅 C_q≈58 fF、α≈−385，无候选可达旧窗）
WINDOW_ALPHA_MHZ_FINITE = (-420.0, -300.0)

# 电路窗口（G0/G1 标定显示可达才启用，否则 Phase A 重冻结，见方案 §6.2）
WINDOW_F01_GHZ = (4.8, 5.4)
WINDOW_ALPHA_MHZ = (-330.0, -150.0)

# ── 快速层 FEM 配置 ──────────────────────────────────────────
# 搜索档（G2 实测成本后可整体调档，但不许逐候选变化）
SEARCH_BOX_MM = (6.0, 6.0, 3.0)     # 封装 (Lx, Ly, Lz)，与冒烟一致
SEARCH_MESH = {"h_metal": 0.08, "h_max": 0.6, "grow": 3.0}
SEARCH_ELEMENT_ORDER = 2            # P2（绝对电容；P1 受 sheet-edge 奇异影响偏高）

# 参与比场采样
PART_SAMPLE = {"dz_um": 2.0, "step_um": 4.0, "margin_um": 60.0}

# ── 扰动协议（方案 §5，用户裁决最小集）───────────────────────
ETCH_UM = 0.25           # 全局刻蚀偏移 ±
LJ_REL = 0.05            # 结电感 ±5%
TI_REL = 0.10            # 界面层厚度 ±10%


def frozen_hash() -> str:
    """冻结参数的内容哈希，写入每个结果目录以绑定协议版本。"""
    payload = {
        k: v for k, v in globals().items()
        if k.isupper() and not k.startswith("_")
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

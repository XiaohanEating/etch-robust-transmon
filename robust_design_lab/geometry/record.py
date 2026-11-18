# -*- coding: utf-8 -*-
"""GeometryRecord：候选几何的不可变快照（方案 v2.0 §3.2）。

同时驱动 Metal 绘图、快速 FEM、扰动变换和复验；记录组件家族、拓扑、
多边形（μm, shapely）、结线与操作历史。任何派生（如刻蚀偏移）生成新记录，
不修改原记录。
"""

from dataclasses import dataclass, field
import hashlib

from shapely.geometry import LineString, Polygon


@dataclass(frozen=True)
class GeometryRecord:
    family: str                     # 组件家族（两岛单结 = "two_pad_free"）
    category: str                   # 形状库类别名
    category_index: int             # 类内序号
    seed: int                       # 生成种子
    pad_left: Polygon               # μm，组件局部坐标，结沿 x 轴
    pad_right: Polygon
    junction: LineString            # 两端落在两岛对面边上
    operation_history: tuple = ()   # 产生该几何的操作链（VLM/数值操作留痕）
    note: str = ""

    # 派生量（frozen dataclass 用 property，不占构造参数）
    @property
    def polys(self) -> tuple:
        return (self.pad_left, self.pad_right)

    @property
    def bounds(self) -> tuple:
        """两岛合并外接框 (minx, miny, maxx, maxy)，μm。"""
        return tuple(self.pad_left.union(self.pad_right).bounds)

    @property
    def geometry_hash(self) -> str:
        blob = "|".join((
            self.family,
            self.pad_left.wkt,
            self.pad_right.wkt,
            self.junction.wkt,
        ))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def with_operation(self, op: str, new_left: Polygon, new_right: Polygon,
                       new_junction: LineString = None) -> "GeometryRecord":
        """应用一次几何操作，返回带历史的新记录（不可变派生）。"""
        return GeometryRecord(
            family=self.family,
            category=self.category,
            category_index=self.category_index,
            seed=self.seed,
            pad_left=new_left,
            pad_right=new_right,
            junction=new_junction if new_junction is not None else self.junction,
            operation_history=self.operation_history + (op,),
            note=self.note,
        )

"""Native TransmonPocket reference with an explicitly fully removed ground.

The pocket spans the whole chip. This is the supported no-ground contract,
not the usual finite-pocket grounded TransmonPocket device.
"""
import json
import time
import numpy as np
from shapely.geometry import box
from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
from qiskit_metal.analyses.fem.solver import Electrostatics,mesh_package,package_from_design
from ..configs import frozen
from ..geometry import library,two_pad
from .fast import FieldSolution
from .participation import surface_participation
from ..runtime.records import ROOT,provenance,write_json


def main():
    rec=library.rect_record(200,300,0,note='native component reference')
    design,_=two_pad.build_design(rec)
    design.delete_component('Q1')
    TransmonPocket(design,'Native',options=dict(pos_x='3mm',pos_y='3mm',orientation='90',
        pad_width='300um',pad_height='200um',pad_gap='30um',inductor_width='20um',
        pocket_width='6mm',pocket_height='6mm'))
    design.rebuild()
    table=design.qgeometry.tables['poly']
    cuts=table[table['subtract'].astype(bool)]
    assert len(cuts)==1 and cuts.iloc[0]['geometry'].covers(box(0,0,6,6)), 'Native ground not fully removed'
    metals=table[~table['subtract'].astype(bool)]
    from shapely.affinity import scale,translate
    expected=[translate(scale(p,xfact=.001,yfact=.001,origin=(0,0)),xoff=3,yoff=3) for p in rec.polys]
    for p in expected:
        assert min(p.symmetric_difference(g).area for g in metals.geometry)<1e-12
    pkg=package_from_design(design,eps_r=frozen.EPS_R_SUB)
    rows=[]
    for h in (.03,.02):
        t=time.perf_counter();mesh=mesh_package(pkg,'full',h_metal=h,h_max=.6,grow=3)
        est=Electrostatics(mesh,order=2)
        fs=FieldSolution(rec.geometry_hash,est.C*1e15,float(est.port_capacitance(*est.names)*1e15),
            (3,3),est,est.names[0],est.names[1],substrate_mm=.5,package_mesh=mesh)
        m=surface_participation(fs,rec,lj_H=12e-9)
        rows.append(dict(h_metal_mm=h,metrics=m,elapsed_s=time.perf_counter()-t))
    report=dict(native_component='TransmonPocket',ground_condition='Pocket covers the entire chip',
        polygon_equal=True,geometry_hash=rec.geometry_hash,rows=rows,provenance=provenance(),
        full_native_component_coverage=False)
    write_json(ROOT/'results'/'calibration_v2'/'native_metal_reference.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('provenance','rows')},ensure_ascii=True))


if __name__=='__main__':main()

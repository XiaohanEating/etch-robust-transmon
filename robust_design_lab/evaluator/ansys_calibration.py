"""Same polygon/box Q3D and HFSS calibration through the installed COM API.

Uses pyEPR's public geometry, setup and matrix/field wrappers. Projects are
created under this experiment's result folder; existing projects are untouched.
"""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
from ..configs import frozen
from ..evaluator.lom import match_lj, transmon_metrics
from ..geometry import library
from ..perturbation.channels import apply_etch
from ..runtime.records import ROOT, provenance, write_json


def connect():
    from pyEPR.ansys import HfssApp
    return HfssApp('Ansoft.ElectronicsDesktop.2022.1').get_app_desktop()


def eigenmode_frequency_hz(frequencies_ghz):
    """Convert pyEPR's eigenmodes() frequency output (GHz) to Hz."""
    values = np.asarray(frequencies_ghz, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values[0]) or values[0] <= 0:
        raise ValueError('HFSS eigenmode frequency must be a finite positive GHz value')
    return float(values[0] * 1e9)


def plasma_frequency_hz(lom_values):
    """Linear LC plasma frequency for comparison with an HFSS eigenmode.

    HFSS returns the harmonic mode, while transmon_metrics()['f01_GHz'] is the
    anharmonic E1-E0 transition.  Comparing those two quantities directly
    would mix different observables.
    """
    ej = float(lom_values['EJ_GHz'])
    ec = float(lom_values['EC_GHz'])
    if not np.isfinite(ej) or not np.isfinite(ec) or ej <= 0 or ec <= 0:
        raise ValueError('EJ and EC must be finite positive GHz values')
    return float(np.sqrt(8.0 * ej * ec) * 1e9)


def geometry(design, rec, hfss, metal_thickness_nm=200.0):
    m = design.modeler
    m.set_units('mm')
    # The frozen config stores dimensions as strings for the Metal renderer;
    # COM-side coordinate transforms require numeric values.
    lx, ly, lz = map(float, frozen.SEARCH_BOX_MM)
    z = float(frozen.SUBSTRATE_MM)
    if metal_thickness_nm <= 0:
        raise ValueError('Metal thickness must be positive')
    def dim(v): return str(float(v))+'mm'
    def box(name, pos, size, material):
        return design._modeler.CreateBox(['NAME:BoxParameters',
            'XPosition:=',dim(pos[0]),'YPosition:=',dim(pos[1]),'ZPosition:=',dim(pos[2]),
            'XSize:=',dim(size[0]),'YSize:=',dim(size[1]),'ZSize:=',dim(size[2])],
            m._attributes_array(name=name,material=material,solve_inside=True))
    # Explicit material; do not use a vendor silicon default with unknown eps.
    materials = design.parent._project.GetDefinitionManager()
    material = 'robust_si_11p45'
    try:
        materials.AddMaterial(['NAME:'+material, 'CoordinateSystemType:=','Cartesian',
            'BulkOrSurfaceType:=',1,['NAME:PhysicsTypes','set:=',['Electromagnetic']],
            'permittivity:=',str(frozen.EPS_R_SUB),'conductivity:=','0',
            'dielectric_loss_tangent:=','0'])
    except Exception:
        # Material may already exist in this experiment project.
        if not materials.DoesMaterialExist(material): raise
    box('Substrate',(0,0,0),(lx,ly,z),material)
    box('Air',(0,0,z),(lx,ly,lz-z),'vacuum')
    for name, poly in zip(('PadLeft','PadRight'), rec.polys):
        points=[[dim(x/1000+lx/2),dim(y/1000+ly/2),dim(z)]
                for x,y in list(poly.exterior.coords)[:-1]]
        m.draw_polyline(points,name=name,material='pec',solve_inside=False)
    # Six physical PEC sheets reproduce the FEM grounded package boundary.
    wall_defs=[((0,0,0),(lx,ly,0)),((0,0,lz),(lx,ly,0)),
               ((0,0,0),(lx,0,lz)),((0,ly,0),(lx,0,lz)),
               ((0,0,0),(0,ly,lz)),((lx,0,0),(0,ly,lz))]
    walls=[]
    for i,(pos,size) in enumerate(wall_defs):
        name='Wall'+str(i);walls.append(name)
        indices=[j for j,v in enumerate(size) if v]
        a,b=indices
        vertices=[list(pos) for _ in range(4)]
        vertices[1][a]+=size[a];vertices[2][a]+=size[a]
        vertices[2][b]+=size[b];vertices[3][b]+=size[b]
        m.draw_polyline([[dim(v) for v in p] for p in vertices],
                        name=name,material='pec',solve_inside=False)
    if hfss:
        m.assign_perfect_E(['PadLeft','PadRight']+walls,name='PEC')
    else:
        # Q3D thin-sheet electrostatics, same geometry as the sheet FEM.
        design._boundaries.AssignThinConductor(['NAME:MetalSheets','Objects:=',
            ['PadLeft','PadRight']+walls,'Material:=','pec',
            'Thickness:=',f'{metal_thickness_nm:g}nm'])
        for name in ('PadLeft','PadRight'):
            design._boundaries.AssignSignalNet(['NAME:'+name,'Objects:=',[name]])
        design._boundaries.AssignGroundNet(['NAME:PackageGround','Objects:=',walls])
    m.mesh_length('PadMesh',['PadLeft','PadRight'],MaxLength='0.02mm',
        RefineInside=False,Enabled=True,RestrictElem=False,RestrictLength=True)
    # AEDT's COM primitives interpret bare coordinates in metres even when the
    # editor displays mm. Inspect the exported vertices before any solve.
    for obj,expected in (('Substrate',(lx,ly,z)),('Air',(lx,ly,lz-z))):
        ids=design._modeler.GetVertexIDsFromObject(obj)
        vertices=np.array([[float(v) for v in design._modeler.GetVertexPosition(int(i))] for i in ids])
        if not np.allclose(np.ptp(vertices,axis=0),expected,atol=1e-7,rtol=1e-7):
            raise ValueError('Ansys geometry units disagree for '+obj)
    return m


def q3d(project, rec, name, output, metal_thickness_nm=200.0):
    design=project.new_q3d_design(name)
    geometry(design, rec, False, metal_thickness_nm)
    setup=design.create_q3d_setup(freq_ghz=5.1,save_fields=True,
         max_passes=15,min_passes=3,min_converged_passes=2,percent_error=0.5)
    checkpoint(project, output)
    setup.analyze()
    c, units, _, variation=setup.get_matrix()
    write_json(output/(name+'_matrix.json'),dict(names=list(c.columns),
               rows=list(c.index),values=c.values.tolist(),units=units,variation=variation))
    names=list(c.columns)
    indices=[next(i for i,n in enumerate(names) if target.lower() in n.lower())
             for target in ('PadLeft','PadRight')]
    matrix=np.asarray(c.values,dtype=float)[np.ix_(indices,indices)]
    # Export units are explicitly inspected; no undocumented multiplier.
    scale={'fF':1.0,'ff':1.0,'farad':1e15,'F':1e15,'pF':1000.0}.get(units.strip())
    if scale is None: raise ValueError('Unrecognized capacitance unit: '+units)
    e=np.array([1.0,-1.0]); cq=scale/(e@np.linalg.solve(matrix,e))
    return dict(C_q_fF=float(cq),units=units,matrix=matrix.tolist(),
                design=name,geometry_hash=rec.geometry_hash,
                metal_thickness_nm=float(metal_thickness_nm))


def hfss(project, rec, name, output, lj, q3d_c_q_f, metal_thickness_nm=200.0):
    from pyEPR.ansys import CalcObject
    design=project.new_em_design(name)
    m=geometry(design,rec,True,metal_thickness_nm)
    lx,ly,_=map(float,frozen.SEARCH_BOX_MM); z=float(frozen.SUBSTRATE_MM)
    coords=list(rec.junction.coords)
    x0,x1=coords[0][0]/1000+lx/2,coords[-1][0]/1000+lx/2
    y=ly/2; width=frozen.JUNCTION_WIDTH_UM/1000
    junction=m.draw_polyline([[str(v)+'mm' for v in p] for p in
         ((x0,y-width/2,z),(x1,y-width/2,z),(x1,y+width/2,z),(x0,y+width/2,z))],
        name='Junction',material='vacuum',solve_inside=True)
    m._make_lumped_rlc(0,str(lj)+'H',str(frozen.CJ_FF)+'fF',
        [str(x0)+'mm',str(y)+'mm',str(z)+'mm'],
        [str(x1)+'mm',str(y)+'mm',str(z)+'mm'],['Objects:=',[str(junction)]])
    m.mesh_length('JunctionMesh',['Junction'],MaxLength='0.005mm',RefineInside=False,
                  Enabled=True,RestrictElem=False,RestrictLength=True)
    setup=design.create_em_setup(min_freq_ghz=3,n_modes=1,max_delta_f=0.1,
              max_passes=15,min_passes=3,min_converged=2)
    checkpoint(project, output)
    setup.analyze()
    solutions=setup.get_solutions()
    freqs,_=solutions.eigenmodes();solutions.set_mode(1)
    f_hfss=eigenmode_frequency_hz(freqs)
    lom_values=transmon_metrics(float(q3d_c_q_f),lj)
    f_lom=lom_values['f01_GHz']*1e9
    f_plasma=plasma_frequency_hz(lom_values)
    def qty(): return CalcObject([],setup).getQty('E')
    # pyEPR's field calculator in AEDT 2022 rejects the composed expression
    # after an eigenmode solve. Keep the solved mode and use its exported
    # eigenfrequency as the cross-solver observable; interface loss remains a
    # separate calibration task.
    return dict(frequencies_GHz=np.asarray(freqs,dtype=float).reshape(-1).tolist(),
                f_hfss_eigen_Hz=f_hfss,f01_lom_Hz=f_lom,
                f_plasma_lom_Hz=f_plasma,
                f_hfss_vs_plasma_relative_error=abs(f_hfss-f_plasma)/f_plasma,
                geometry_hash=rec.geometry_hash,design=name,lj0_H=lj,
                metal_thickness_nm=float(metal_thickness_nm),
                metal_model='zero-thickness PEC sheet',
                participation_validated=False,
                note='HFSS eigenmode solved; pyEPR field calculator expression rejected by AEDT 2022.1, so interface participation is not claimed')


def checkpoint(project, output):
    if getattr(project, '_robust_saved', False):
        project.save()
    else:
        project.save(str((output/('qmetal_'+output.parent.name+'_'+output.name+'.aedt')).resolve()))
        project._robust_saved=True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results'/'calibration_v2'/'ansys')
    parser.add_argument('--hfss',action='store_true')
    parser.add_argument('--etched',action='store_true')
    parser.add_argument('--round',type=float,default=0)
    parser.add_argument('--metal-thickness-nm',type=float,default=200.0,
        help='Q3D thin-conductor thickness assumption; default is 200 nm')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    tag=('round'+str(args.round)).replace('.','p')
    report=dict(provenance=provenance(),passed=False,rows=[],
                metal_thickness_nm=args.metal_thickness_nm,
                thickness_scope='Q3D thin conductor; HFSS uses PEC sheet')
    start=time.perf_counter();project=None
    try:
        desktop=connect();report['ansys_version']=desktop.get_version()
        project=desktop.new_project()
        # Unique project name avoids overwriting any previously opened project.
        folder=args.output/tag
        folder.mkdir(parents=True,exist_ok=True)
        base=library.rect_record(200,300,args.round,note='same-device')
        for i,delta in enumerate((0.0,0.25,-0.25) if args.etched else (0.0,)):
            rec,error=apply_etch(base,delta)
            if error: raise ValueError(error)
            row=dict(etch_um=delta,geometry_hash=rec.geometry_hash,
                     metal_thickness_nm=args.metal_thickness_nm)
            report['rows'].append(row)
            print('Q3D',tag,delta,flush=True)
            try: row['q3d']=q3d(project,rec,'Q3D_'+str(i),folder,args.metal_thickness_nm)
            except Exception as error: row['q3d']=dict(failed=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
            write_json(folder/'report.json',report)
            if delta==0.0 and row.get('q3d',{}).get('C_q_fF') is not None:
                row['lj0_H'],row['lj_match']=match_lj(row['q3d']['C_q_fF']*1e-15,5.10)
                if row['lj0_H'] is not None:
                    # GeometryRecord is an immutable snapshot.  Keep the
                    # Q3D-derived calibration in the report instead of
                    # mutating the record before the HFSS stage.
                    row['q3d_C_q_F']=row['q3d']['C_q_fF']*1e-15
                    report['nominal_q3d_C_q_F']=row['q3d_C_q_F']
                    report['nominal_lj0_H']=row['lj0_H']
            row['lj0_H']=report.get('nominal_lj0_H')
            if row['lj0_H'] is None:
                row['circuit_match_failed']=True
            if args.hfss:
                print('HFSS',tag,delta,flush=True)
                try:
                    c_q_f = row.get('q3d',{}).get('C_q_fF')
                    if c_q_f is None or row['lj0_H'] is None:
                        raise ValueError('Q3D capacitance or nominal junction match is unavailable')
                    row['hfss']=hfss(project,rec,'HFSS_'+str(i),folder,row['lj0_H'],
                                      c_q_f*1e-15,args.metal_thickness_nm)
                except Exception as error: row['hfss']=dict(failed=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
            write_json(folder/'report.json',report)
        report['messages']=list(desktop.get_messages('','',2))[-20:]
        checkpoint(project,folder)
        row_checks=[r for r in report['rows'] if 'hfss' in r and r.get('q3d',{}).get('C_q_fF') is not None]
        report['electrostatic_frequency_cross_solver_passed']=bool(args.hfss and len(row_checks)==3 and all(
            r.get('hfss',{}).get('f_hfss_vs_plasma_relative_error',float('inf'))<=0.02
            for r in row_checks))
        report['participation_validated']=False
        report['passed']=False
        report['note']='C and eigenfrequency only; surface participation and 200nm 3D thickness sensitivity remain unvalidated'
    except Exception as error:
        report['failed']=type(error).__name__+': '+str(error)
    report['elapsed_s']=time.perf_counter()-start
    write_json(args.output/(tag+'_execution.json'),report)
    print(json.dumps(report,ensure_ascii=True),flush=True)


if __name__=='__main__': main()

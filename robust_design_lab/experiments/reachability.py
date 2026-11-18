"""All 1024 shapes, bounded size checks, and explicit nominal feasibility.

This experiment measures C/f/alpha only. It does not rank uncalibrated T1.
Each process solves a distinct job; only the parent commits atomic records.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from ..configs import frozen
from ..geometry import checks,library,operations
from ..evaluator import fast,lom
from ..perturbation.channels import apply_etch
from ..runtime.records import ROOT,provenance,write_json


def job_record(category,index,variant):
    record=library.make_record(category,index)
    return operations.fill_extent(record) if variant=='extent' else record


def evaluate(job):
    category,index,variant=job
    start=time.perf_counter()
    row=dict(category=category,index=index,variant=variant,window_ok=False)
    try:
        rec=job_record(*job)
        row.update(hash=rec.geometry_hash,polygons_wkt=[p.wkt for p in rec.polys],
                   junction_wkt=rec.junction.wkt,operations=rec.operation_history)
        valid,reasons=checks.check_record(rec)
        row.update(checks_ok=valid,geometry_failures=reasons)
        if not valid:return row
        fs=fast.solve_fields(rec)
        lj,info=lom.match_lj(fs.C_q_fF*1e-15,5.10)
        row.update(C_q_fF=fs.C_q_fF,lj0_H=lj,match_status=info,
                   mesh_time_s=fs.mesh_time_s,solve_time_s=fs.solve_time_s)
        if lj is not None:
            metrics=lom.transmon_metrics(fs.C_q_fF*1e-15,lj)
            row['nominal']=metrics
            row['window_ok']=bool(frozen.WINDOW_F01_GHZ[0]<=metrics['f01_GHz']<=frozen.WINDOW_F01_GHZ[1]
                and frozen.WINDOW_ALPHA_MHZ[0]<=metrics['alpha_MHz']<=frozen.WINDOW_ALPHA_MHZ[1])
        row['etch_geometry_failures']={str(d):apply_etch(rec,d)[1] for d in (0.25,-0.25)}
    except Exception as error:
        row['failed']=type(error).__name__+': '+str(error)
    finally:
        row['elapsed_s']=time.perf_counter()-start
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--per-class',type=int,default=128)
    parser.add_argument('--extent-per-class',type=int,default=8)
    parser.add_argument('--output',type=Path,default=ROOT/'results'/'reachability_v2')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    lock=args.output/'run.lock'
    try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Another reachability run owns this output directory')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    start=time.perf_counter()
    try:
        manifest=provenance()
        relevant={k:v for k,v in manifest['source_sha256'].items()
                  if k.startswith(('geometry/','configs/','evaluator/fast','evaluator/lom','experiments/reachability'))}
        signature=hashlib.sha256(json.dumps(relevant,sort_keys=True).encode()).hexdigest()
        manifest.update(protocol_signature=signature,workers=args.workers,
                        nominal_only=True,shape_count=8*args.per_class,
                        extent_count=8*args.extent_per_class)
        old=args.output/'provenance.json'
        if old.exists() and json.loads(old.read_text())['protocol_signature']!=signature:
            raise RuntimeError('Evaluation source changed; use a new output directory')
        write_json(old,manifest)
        jobs=[(c,i,'original') for c in library.CATEGORY_NAMES for i in range(args.per_class)]
        jobs += [(c,i,'extent') for c in library.CATEGORY_NAMES for i in range(args.extent_per_class)]
        def path(job):return args.output/'raw'/('_'.join(map(str,job))+'.json')
        pending=[j for j in jobs if not path(j).exists()]
        print('JOBS',len(jobs),'pending',len(pending),'workers',args.workers,flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures={executor.submit(evaluate,j):j for j in pending}
            for i,future in enumerate(as_completed(futures),1):
                j=futures[future]
                try:row=future.result()
                except Exception as error:row=dict(category=j[0],index=j[1],variant=j[2],failed=str(error))
                row['protocol_signature']=signature;write_json(path(j),row)
                if i%64==0 or i==len(pending):print('DONE',i,'/',len(pending),'elapsed',round(time.perf_counter()-start,1),flush=True)
        rows=[json.loads(path(j).read_text(encoding='utf-8')) for j in jobs]
        categories={}
        for c in library.CATEGORY_NAMES:
            values=[r for r in rows if r['category']==c]
            categories[c]={}
            for v in ('original','extent'):
                part=[r for r in values if r['variant']==v]
                caps=[r['C_q_fF'] for r in part if 'C_q_fF' in r]
                categories[c][v]=dict(count=len(part),geometry_ok=sum(r.get('checks_ok',False) for r in part),
                    solved=len(caps),window_ok=sum(r.get('window_ok',False) for r in part),
                    capacitance_range_fF=[min(caps),max(caps)] if caps else None,
                    feasible_hashes=[r['hash'] for r in part if r.get('window_ok')])
        g0=[r for r in rows if r['variant']=='original' and r['index']<3]
        g0pass=sum('C_q_fF' in r for r in g0)
        write_json(args.output/'g0_report.json',dict(gate='G0-geometry',n_total=len(g0),n_passed=g0pass,
            passed=len(g0)==24 and g0pass>=20 and all(any(r['category']==c and 'C_q_fF' in r for r in g0) for c in library.CATEGORY_NAMES),
            note='Custom no-ground two-island geometry contract only',rows=g0))
        elapsed=time.perf_counter()-start
        write_json(args.output/'report.json',dict(n_jobs=len(rows),complete=True,per_category=categories,
            elapsed_s=elapsed,throughput_candidates_s=len(pending)/max(elapsed,1e-9),workers=args.workers,
            nominal_only=True,T1_ranked=False,protocol_signature=signature,
            note='Measured reachability, not proof of impossibility for unsampled shapes; coarse C needs final mesh recheck'))
        print(json.dumps(categories),flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()

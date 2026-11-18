"""Versioned G1 diagnostics; missing cross-solver evidence always fails G1."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..configs import frozen
from ..evaluator import fast, lom, participation
from ..geometry import library
from ..objectives import drift
from ..perturbation import channels
from ..runtime.records import ROOT, provenance, write_json


def representatives():
    out = [library.make_record(cat, 0) for cat in library.CATEGORY_NAMES]
    out += [library.rect_record(200.0, 300.0, r, note='calibration') for r in (0.0, 30.0)]
    return out


def evaluate_record(rec, h, diagnostics=False):
    cache = fast.FieldCache()
    mesh = dict(h_metal=h, h_max=0.6, grow=3.0)
    rows = []
    lj = None
    for delta in (0.0, frozen.ETCH_UM, -frozen.ETCH_UM):
        altered, error = channels.apply_etch(rec, delta)
        if error:
            rows.append(dict(etch_um=delta, failed=error))
            continue
        fs = cache.get_or_solve(altered, mesh)
        if delta == 0.0:
            lj, info = lom.match_lj(fs.C_q_fF * 1e-15, 5.10)
        chosen_lj = frozen.LJ0_NH * 1e-9 if lj is None else lj
        metrics = participation.surface_participation(fs, altered, lj_H=chosen_lj)
        row = dict(etch_um=delta, hash=altered.geometry_hash, metrics=metrics,
                   mesh_time_s=fs.mesh_time_s, solve_time_s=fs.solve_time_s,
                   tetrahedra=int(fs.est._t.shape[1]))
        if diagnostics and delta == 0.0:
            row['diagnostics'] = {}
            for tag, sample in (('quadrature4', dict(quadrature_order=4)),
                                ('SA_air', dict(sa_side='air')),
                                ('offset_0.5um', dict(offset_um=0.5)),
                                ('offset_1um', dict(offset_um=1.0)),
                                ('edge_excluded_0.5um', dict(edge_cutoff_um=0.5)),
                                ('edge_excluded_1um', dict(edge_cutoff_um=1.0))):
                row['diagnostics'][tag] = participation.surface_participation(fs, altered,
                                                            sample=sample, lj_H=chosen_lj)
        rows.append(row)
    nominal = rows[0].get('metrics', {})
    results = {str(row['etch_um']): row.get('metrics', dict(failed=row.get('failed', 'No metrics')))
               for row in rows}
    return dict(category=rec.category, index=rec.category_index, hash=rec.geometry_hash,
                h_metal_mm=h, lj0_H=lj, rows=rows, drift=drift.drift_metrics(nominal, results))


def summarize(rows, mesh_sizes, ansys=None):
    by_hash = {}
    for row in rows:
        by_hash.setdefault(row['hash'], {})[row['h_metal_mm']] = row
    errors = []
    fine, previous = sorted(mesh_sizes)[:2] if len(mesh_sizes) >= 2 else (0, 0)
    for gh, levels in by_hash.items():
        if fine not in levels or previous not in levels:
            errors.append(dict(hash=gh, failed='Missing consecutive fine meshes'))
            continue
        a, b = levels[previous], levels[fine]
        if not a['drift']['valid'] or not b['drift']['valid']:
            errors.append(dict(hash=gh, failed='A required etched scene failed'))
            continue
        ma, mb = a['rows'][0]['metrics'], b['rows'][0]['metrics']
        rel = {k: abs(mb[k] / ma[k] - 1) for k in ('C_q_fF', 'p_SM', 'p_MA', 'p_SA')}
        delta = abs(b['drift']['D_T1']['max'] - a['drift']['D_T1']['max'])
        errors.append(dict(hash=gh, relative_change=rel, drift_absolute_change=delta,
                           passed=rel['C_q_fF'] <= 0.02 and max(rel[k] for k in ('p_SM','p_MA','p_SA')) <= 0.05
                                  and delta <= 0.005))
    numerical = bool(errors and all(e.get('passed', False) for e in errors))
    cross = bool(ansys and ansys.get('passed', False))
    return dict(gate='G1-physics', evaluator_version=participation.VERSION,
                numerical_passed=numerical, per_geometry=errors, cross_solver_passed=cross,
                passed=numerical and cross,
                missing=[] if cross else ['Q3D/HFSS/pyEPR same-device comparison'],
                note='Small mesh errors alone do not establish physical edge-model convergence.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', nargs='+', type=float, default=[0.05, 0.03, 0.02])
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--rectangles-only', action='store_true')
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'calibration_v2')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest=provenance()
    relevant={k:v for k,v in manifest['source_sha256'].items()
              if k.startswith(('configs/','geometry/','evaluator/fields','evaluator/participation',
                               'evaluator/fast','evaluator/lom','experiments/calibration'))}
    signature=hashlib.sha256(json.dumps(relevant,sort_keys=True).encode()).hexdigest()
    manifest.update(evaluation_signature=signature,diagnostics=args.diagnostics,meshes=args.mesh)
    previous=args.output/'provenance.json'
    if previous.exists():
        old=json.loads(previous.read_text(encoding='utf-8'))
        if old.get('evaluation_signature')!=signature:
            raise RuntimeError('Source version changed or unversioned legacy cache; use a new output directory')
    write_json(previous,manifest)
    records = representatives()
    if args.rectangles_only:
        records=records[-2:]
    if args.limit:
        records = records[:args.limit]
    rows = []
    start = time.perf_counter()
    for rec in records:
        for h in args.mesh:
            path = args.output / 'raw' / (rec.geometry_hash + '_h' + str(h) + '.json')
            if path.exists():
                row = json.loads(path.read_text(encoding='utf-8'))
            else:
                print('SOLVE', rec.category, rec.category_index, 'h=', h, flush=True)
                try:
                    row = evaluate_record(rec, h, diagnostics=args.diagnostics)
                except Exception as error:
                    row = dict(category=rec.category, hash=rec.geometry_hash, h_metal_mm=h,
                               failed=type(error).__name__ + ': ' + str(error))
                write_json(path, row)
            rows.append(row)
            print('DONE', rec.category, h, 'elapsed=', round(time.perf_counter()-start,2), flush=True)
    complete = [row for row in rows if 'drift' in row]
    report = summarize(complete, args.mesh)
    report.update(n_cases=len(rows), n_failed=sum('failed' in r for r in rows),
                  representative_count=len(records), elapsed_s=time.perf_counter()-start,
                  criteria=dict(capacitance_relative=0.02, participation_relative=0.05, drift_absolute=0.005),
                  raw_files='raw/*.json')
    if len(records) != 10 or len(complete) != len(rows):
        report['passed'] = False
        report['missing'].append('Complete representative matrix')
    write_json(args.output / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

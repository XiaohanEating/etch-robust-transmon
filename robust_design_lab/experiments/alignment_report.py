"""Read saved finite-film diagnostics, retaining missing cases and directions."""

import argparse
import json
import math
from pathlib import Path

from ..configs import frozen
from ..evaluator.lom import transmon_metrics
from ..perturbation.channels import apply_etch
from ..runtime.records import provenance, write_json
from .calibration import representatives
from .field_audit import sha256

SCENES = {'nominal': 0.0, 'etch_plus': frozen.ETCH_UM, 'etch_minus': -frozen.ETCH_UM}
CRITERIA = dict(capacitance_relative=0.02, participation_relative=0.05, drift_absolute=0.005)
PARTS = ('p_SM', 'p_MA', 'p_SA')


def source_issues(report, geometry_hash, scene, kind, lj_H):
    """Reject comparisons between different physical problems or protocols."""
    issues = []
    if kind == 'fast':
        identity = report.get('identity', {})
        thickness = identity.get('settings', {}).get('thickness_nm')
        inductance = report.get('lj_H')
        config = identity.get('frozen_config')
    elif kind == 'hfss':
        identity = report.get('model_geometry', {})
        thickness = identity.get('thickness_nm')
        inductance = identity.get('lj0_H')
        config = identity.get('provenance', {}).get('frozen_config')
        if identity.get('etch_um', 0.0) != SCENES[scene]:
            issues.append('etched scene differs from requested direction')
        if not report.get('diagnostic_completed'):
            issues.append('HFSS diagnostic incomplete: '+str(report.get('failed', 'unknown')))
        if not report.get('frequency_converged'):
            issues.append('HFSS frequency convergence not recorded')
    else:
        raise ValueError('Unknown source kind: '+kind)
    if identity.get('geometry_hash') != geometry_hash:
        issues.append('geometry hash differs')
    if thickness != 200.0:
        issues.append('metal thickness differs from the 200 nm comparison')
    if (not isinstance(inductance, (int, float)) or not math.isfinite(inductance)
            or not math.isclose(inductance, lj_H, rel_tol=1e-10, abs_tol=0)):
        issues.append('nominal junction inductance differs')
    if config != frozen.frozen_hash():
        issues.append('frozen physical configuration differs')
    if report.get('provenance', {}).get('frozen_config') != frozen.frozen_hash():
        issues.append('report provenance configuration differs')
    return issues


def compare_triplets(coarse, fine):
    """Check each scene and signed etch difference, never only the maximum."""
    missing = [f'{level}/{scene}' for level, rows in (('coarse', coarse), ('fine', fine))
               for scene in SCENES if scene not in rows]
    if missing:
        return dict(passed=False, missing=missing)
    relative = {}
    for scene in SCENES:
        relative[scene] = {}
        for key in ('C_q_fF', *PARTS, 'T1_us'):
            a, b = coarse[scene][key], fine[scene][key]
            if not all(math.isfinite(x) and x > 0 for x in (a, b)):
                raise ValueError('Nonpositive/nonfinite physical value: '+scene+'/'+key)
            relative[scene][key] = abs(b/a-1)
    drifts = {}
    for scene in ('etch_plus', 'etch_minus'):
        a = math.log(coarse[scene]['T1_us']/coarse['nominal']['T1_us'])
        b = math.log(fine[scene]['T1_us']/fine['nominal']['T1_us'])
        drifts[scene] = dict(coarse_log_ratio=a, fine_log_ratio=b,
                             signed_difference_change=abs(b-a),
                             absolute_drift_change=abs(abs(b)-abs(a)),
                             direction_same=a*b > 0 or a == b == 0)
    passes = all(row['C_q_fF'] <= CRITERIA['capacitance_relative'] and
                 max(row[k] for k in PARTS) <= CRITERIA['participation_relative']
                 for row in relative.values())
    passes &= all(row['signed_difference_change'] <= CRITERIA['drift_absolute']
                  for row in drifts.values())
    return dict(passed=bool(passes), relative_change_by_scene=relative,
                etch_log_drift_by_direction=drifts, criteria=CRITERIA)


def hfss_metrics(report):
    """Derive circuit metrics from saved mode integrals, retaining raw fields."""
    fields = report['fields']
    capacitance = fields['bulk_electric_energy_integral']/fields['junction_voltage_squared']
    tm = transmon_metrics(capacitance, report['model_geometry']['lj0_H'])
    p = fields['participation']
    rate = 2*math.pi*tm['f01_GHz']*1e9*sum(p['p_'+k]*frozen.TAN_DELTA[k] for k in frozen.TAN_DELTA)
    return dict(tm, **p, C_q_fF=capacitance*1e15, T1_us=1e6/rate,
                capacitance_definition='HFSS bulk electric-energy integral / junction voltage squared',
                f_eigen_Hz=fields['f_eigen_Hz'])


def build_report(folder):
    folder = Path(folder)
    records = representatives()
    rec = records[-2]
    expected = {scene: apply_etch(rec, delta)[0].geometry_hash for scene, delta in SCENES.items()}
    reference = Path('robust_design_lab/results/hfss_finite_film_v1')
    reference_model = reference/'200nm_refine13'/'model.json'
    lj_H = json.loads(reference_model.read_text(encoding='utf-8'))['lj0_H']
    validation = {}
    paths, saved = {}, {}
    for level in ('h0p01', 'h0p005'):
        saved[level] = {}
        for scene in SCENES:
            path = folder/'rect_reference'/level/scene/'report.json'
            if path.exists():
                row = json.loads(path.read_text(encoding='utf-8'))
                paths[str(path)] = sha256(path)
                issues = source_issues(row, expected[scene], scene, 'fast', lj_H)
                try:
                    metrics = row['metrics']
                    compare_triplets({s: metrics for s in SCENES}, {s: metrics for s in SCENES})
                except (KeyError, TypeError, ValueError) as error:
                    issues.append('invalid physical metrics: '+str(error))
                validation[str(path)] = dict(passed=not issues, issues=issues)
                if not issues:
                    saved[level][scene] = metrics
    numerical = compare_triplets(saved['h0p01'], saved['h0p005'])
    hfss = {}
    for level in (11, 13):
        hfss[level] = {}
        for scene in SCENES:
            path = (reference/f'200nm_refine{level}'/'report.json' if scene == 'nominal'
                    else folder/'hfss'/f'{scene}_pass{level}'/'report.json')
            if path.exists():
                row = json.loads(path.read_text(encoding='utf-8'))
                paths[str(path)] = sha256(path)
                issues = source_issues(row, expected[scene], scene, 'hfss', lj_H)
                try:
                    metrics = hfss_metrics(row)
                    compare_triplets({s: metrics for s in SCENES}, {s: metrics for s in SCENES})
                except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                    issues.append('invalid physical metrics: '+str(error))
                validation[str(path)] = dict(passed=not issues, issues=issues)
                if not issues:
                    hfss[level][scene] = metrics
    hfss_checks = compare_triplets(hfss[11], hfss[13])
    cross = {}
    for scene in SCENES:
        if scene in saved['h0p005'] and scene in hfss[13]:
            a, b = saved['h0p005'][scene], hfss[13][scene]
            cross[scene] = {k: dict(fast=a[k], hfss=b[k], fast_over_hfss_minus_one=a[k]/b[k]-1)
                             for k in ('C_q_fF', *PARTS, 'T1_us')}
    cross_drift = {}
    for scene in ('etch_plus', 'etch_minus'):
        if all(k in saved['h0p005'] and k in hfss[13] for k in ('nominal', scene)):
            a = math.log(saved['h0p005'][scene]['T1_us']/saved['h0p005']['nominal']['T1_us'])
            b = math.log(hfss[13][scene]['T1_us']/hfss[13]['nominal']['T1_us'])
            cross_drift[scene] = dict(fast_log_ratio=a, hfss_log_ratio=b,
                                     direction_same=a*b > 0 or a == b == 0,
                                     difference=abs(a-b),
                                     both_exceed_drift_tolerance=min(abs(a),abs(b)) > CRITERIA['drift_absolute'])
    matrix = []
    reference_hash = records[-2].geometry_hash
    for index, rec in enumerate(records):
        for scene, delta in SCENES.items():
            altered, error = apply_etch(rec, delta)
            is_reference = rec.geometry_hash == reference_hash
            matrix.append(dict(representative=index, category=rec.category, nominal_hash=rec.geometry_hash,
                               scene=scene, etch_um=delta,
                               geometry_hash=None if error else altered.geometry_hash, geometry_error=error,
                               fast={level: 'completed' if is_reference and scene in saved[level] else 'not_run'
                                     for level in saved},
                               hfss={str(level): 'completed' if is_reference and scene in hfss[level] else 'not_run'
                                     for level in hfss}))
    complete = all(row['geometry_error'] is None and
                   all(v == 'completed' for v in (*row['fast'].values(), *row['hfss'].values()))
                   for row in matrix)
    return dict(gate='G1-physics', G1_passed=False, representative_count=len(records),
                required_physical_scenes=len(matrix), full_matrix_complete=complete,
                rectangle_fast_convergence=numerical, rectangle_hfss_convergence=hfss_checks,
                fast_metrics=saved, hfss_metrics=hfss, cross_solver_relative=cross,
                cross_solver_etch_drift=cross_drift, matrix=matrix, source_report_sha256=paths,
                source_validation=validation,
                source_validation_passed=bool(validation) and all(r['passed'] for r in validation.values()),
                completed_rectangle_scenes={k: len(v) for k, v in saved.items()},
                completed_hfss_rectangle_scenes={str(k): len(v) for k, v in hfss.items()},
                downstream=dict(cost_50_candidates='not_run_G1_not_passed',
                                matched_budget_visual_experiments='not_run_G1_not_passed'),
                note='A rectangle subcheck cannot establish G1; all missing cases remain listed.',
                provenance=provenance())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.input)
    write_json(args.input/'alignment_report.json', report)
    print(json.dumps({k: report[k] for k in ('G1_passed', 'full_matrix_complete',
                     'rectangle_fast_convergence', 'rectangle_hfss_convergence',
                     'cross_solver_etch_drift')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

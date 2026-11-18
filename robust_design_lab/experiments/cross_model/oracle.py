"""Q3D evaluation and checked reuse of previously audited starting geometries."""
import json
import math
from pathlib import Path
import re
import time

from robust_design_lab.runtime.records import write_json
from robust_design_lab.geometry import novel_structures as geo
from robust_design_lab.experiments.q3d_shape_sweep import compute_c_sigma, freq_from_csigma, run_q3d
from .config import SETTINGS, TASKS, identity, structure


def metrics_from_scenarios(scenarios):
    if set(scenarios) != {'nom', 'ep', 'em'}:
        raise ValueError('Exactly three scenarios required')
    cs, frequencies = {}, {}
    for tag, row in scenarios.items():
        if row.get('converged') is not True or row.get('percent_error') != SETTINGS['percent_error']:
            raise ValueError('Unconverged or mismatched precision: ' + tag)
        cs[tag] = compute_c_sigma(row['matrix_fF'], row['nets'], 'A_dual_pad')
        if abs(cs[tag] - row['C_sigma_fF']) > 1e-8:
            raise ValueError('Capacitance matrix and summary differ')
        frequencies[tag] = freq_from_csigma(cs[tag])[0]
        if not math.isfinite(frequencies[tag]) or frequencies[tag] <= 0:
            raise ValueError('Invalid frequency')
    f0 = frequencies['nom']
    return dict(C_sigma_fF=cs['nom'], f01_GHz=f0,
                alpha_MHz=freq_from_csigma(cs['nom'])[1],
                D=max(abs(math.log(frequencies[k] / f0)) for k in ('ep', 'em')),
                signed_shift_MHz={k: 1000 * (frequencies[k] - f0) for k in ('ep', 'em')},
                scenarios=scenarios)


def compare_saved_geometry(project, expected):
    """These two task families have solid islands; compare their saved polygons."""
    from shapely.geometry import Polygon, box
    text = Path(project).read_bytes().decode('utf-8', errors='replace')
    parts = {}
    for part in re.findall(r"\$begin 'GeometryPart'(.*?)\$end 'GeometryPart'", text, re.S):
        name = re.search(r"Name='([^']+)'", part)[1]
        if name not in ('pad_left_Q1', 'pad_right_Q1', 'rect_pk_Q1'):
            continue
        if 'ToolOperations(' in part:
            raise ValueError('Hole/subtract history requires a full geometry audit')
        polyline = re.search(r"\$begin 'PolylineParameters'(.*?)\$end 'PolylineParameters'", part, re.S)
        rectangle = re.search(r"\$begin 'RectangleParameters'(.*?)\$end 'RectangleParameters'", part, re.S)
        if polyline:
            points = re.findall(r"\$begin 'PLPoint'(.*?)\$end 'PLPoint'", polyline[1], re.S)
            xy = [(float(re.search(r"X='([^']+)'", p)[1]) * 1e6,
                   float(re.search(r"Y='([^']+)'", p)[1]) * 1e6) for p in points]
            parts[name] = Polygon(xy)
        elif rectangle:
            values = {key: float(re.search(key + r"='([^']+)'", rectangle[1])[1]) * 1e6
                      for key in ('XStart', 'YStart', 'Width', 'Height')}
            x, y = values['XStart'], values['YStart']
            parts[name] = box(x, y, x + values['Width'], y + values['Height'])
    mapping = {'pad_left_Q1': expected['pad_left'], 'pad_right_Q1': expected['pad_right'],
               'rect_pk_Q1': expected.get('pocket') if expected.get('pocket') is not None
               else geo.nominal_pocket(expected)}
    differences = {key: parts[key].symmetric_difference(poly).area for key, poly in mapping.items()}
    if max(differences.values()) >= 0.01:
        raise ValueError('Saved project geometry differs: ' + str(differences))
    if float(re.search(r'\bPerError=([0-9.]+)', text)[1]) != SETTINGS['percent_error']:
        raise ValueError('Saved precision differs')
    return differences


class Oracle:
    def __init__(self, root, source_identity):
        self.root = Path(root)
        self.source_identity = source_identity
        self.root.mkdir(parents=True, exist_ok=True)

    def description(self, task, parameters):
        base = structure(task, parameters)
        scenes = {'nom': base, 'ep': geo.etch_variant(base, SETTINGS['etch_um']),
                  'em': geo.etch_variant(base, -SETTINGS['etch_um'])}
        for tag, st in scenes.items():
            errors = geo.check_structure(st)
            if errors:
                raise ValueError(tag + ': ' + '; '.join(errors))
        options = {tag: geo.struct_to_options(st) for tag, st in scenes.items()}
        description = dict(task=task, parameters=parameters, options=options,
                           solver={k: SETTINGS[k] for k in ('percent_error', 'max_passes')},
                           source_identity=self.source_identity)
        return identity(description), description, scenes

    def import_start(self, task, reference_dir):
        parameters = TASKS[task]['start']
        key, description, structs = self.description(task, parameters)
        target = self.root / key
        source = Path(reference_dir) / TASKS[task]['reference']
        original = json.loads(source.read_text(encoding='utf-8'))['row']
        metrics = metrics_from_scenarios(original['scenarios'])
        comparisons = {tag: compare_saved_geometry(row['project_path'], structs[tag])
                       for tag, row in original['scenarios'].items()}
        import hashlib
        origin = dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      geometry_comparisons_um2=comparisons,
                      historical_D=original['D'],
                      note='reuse converged matrices; frequency recomputed with exact constants')
        result = dict(identity=key, description=description, metrics=metrics, origin=origin)
        write_json(target / 'result.json', result)
        return dict(metrics, oracle_identity=key, oracle_result=str(target / 'result.json'))

    def import_measurements(self, previous_root):
        """Reuse field results after checking saved geometry, settings and matrices.

        Only complete converged oracle results qualify. This does not copy search
        outcomes or silently reinterpret a previous model run under a new protocol.
        """
        import hashlib
        imported = []
        for source in sorted(Path(previous_root).glob('*/result.json')):
            previous = json.loads(source.read_text(encoding='utf-8'))
            old = previous['description']
            key, description, structs = self.description(old['task'], old['parameters'])
            if old['options'] != description['options'] or old['solver'] != description['solver']:
                raise ValueError('Previous measurement geometry/solver differs')
            metrics = metrics_from_scenarios(previous['metrics']['scenarios'])
            checks = {tag: compare_saved_geometry(row['project_path'], structs[tag])
                      for tag, row in metrics['scenarios'].items()}
            target = self.root / key / 'result.json'
            if target.exists():
                continue
            origin = dict(path=str(source.resolve()), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          previous_oracle_identity=previous['identity'], geometry_comparisons_um2=checks,
                          note='field-result reuse only; model search is independently repeated')
            write_json(target, dict(identity=key, description=description, metrics=metrics, origin=origin))
            imported.append(dict(source=str(source), target=str(target), D=metrics['D']))
        write_json(self.root.parent / 'imported_measurements.json', imported)
        return imported

    def __call__(self, task, parameters):
        key, description, structs = self.description(task, parameters)
        target = self.root / key
        result_path = target / 'result.json'
        if result_path.exists():
            saved = json.loads(result_path.read_text(encoding='utf-8'))
            if saved['identity'] != key or identity(saved['description']) != key:
                raise ValueError('Oracle cache identity differs')
            return dict(saved['metrics'], oracle_identity=key, oracle_result=str(result_path))
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / 'geometry.json', dict(identity=key, **description))
        scenarios = {}
        for tag in ('nom', 'ep', 'em'):
            path = target / (tag + '.json')
            if path.exists():
                saved = json.loads(path.read_text(encoding='utf-8'))
                if saved['identity'] != key:
                    raise ValueError('Scenario identity differs')
                if saved.get('error'):
                    raise RuntimeError(saved['error'])
                row = saved['result']
            else:
                t0 = time.perf_counter()
                try:
                    row = run_q3d(description['options'][tag], 'NovelTwoPadTransmon', 'A_dual_pad',
                                  max_passes=SETTINGS['max_passes'],
                                  percent_error=SETTINGS['percent_error'], output_dir=target / tag)
                    write_json(path, dict(identity=key, result=row, wall_s=time.perf_counter() - t0))
                except Exception as exc:
                    write_json(path, dict(identity=key, error='%s: %s' % (type(exc).__name__, str(exc)),
                                         wall_s=time.perf_counter() - t0))
                    raise
            if row.get('converged') is not True or row.get('project_saved') is not True:
                raise ValueError('Rejected scene %s: convergence/save status' % tag)
            scenarios[tag] = row
            print(json.dumps(dict(scene=tag, task=task, parameters=parameters,
                                  passes=len(row['capacitance_all_passes_fF'] or {}),
                                  C_sigma_fF=row['C_sigma_fF'], solve_s=row['solve_s'])), flush=True)
        metrics = metrics_from_scenarios(scenarios)
        write_json(result_path, dict(identity=key, description=description, metrics=metrics))
        return dict(metrics, oracle_identity=key, oracle_result=str(result_path))

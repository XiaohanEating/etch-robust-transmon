"""Finite PEC film and separately named SM/MA/SA HFSS integration surfaces.

This is a same-rectangle diagnostic, not a G1 pass or a search evaluator.
Metal rises into air; SM is the bottom face, MA the top and side faces.
The assumed loss layers are evaluated from boundary fields, not meshed oxides.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np

from ..configs import frozen
from ..evaluator.ansys_calibration import (
    connect, eigenmode_frequency_hz, geometry, plasma_frequency_hz,
)
from ..evaluator.lom import transmon_metrics
from ..geometry import library
from ..perturbation.channels import apply_etch
from ..runtime.records import provenance, write_json

EPS0 = 8.8541878128e-12


def partition_faces(face_vertices, substrate_mm, thickness_nm):
    """Classify a finite metal's bottom and remaining faces by geometry."""
    bottom, other = [], []
    top = substrate_mm + thickness_nm * 1e-6
    for face, vertices in face_vertices.items():
        z = np.asarray(vertices, dtype=float)[:, 2]
        if z.min() < substrate_mm - 1e-8 or z.max() > top + 1e-8:
            raise ValueError('Face extends beyond the requested finite metal')
        (bottom if np.allclose(z, substrate_mm, atol=1e-8, rtol=0) else other).append(int(face))
    if not bottom or not other:
        raise ValueError('Finite metal must have distinct bottom and air-side faces')
    return bottom, other


def detach_surface(design, obj, face_ids, name):
    """Copy specified faces as named, nonmodel integration surfaces."""
    editor = design._modeler
    before = set(editor.GetObjectsInGroup('Sheets'))
    editor.CreateObjectFromFaces(
        ['NAME:Selections', 'Selections:=', obj, 'NewPartsModelFlag:=', 'NonModel'],
        ['NAME:Parameters', ['NAME:BodyFromFaceToParameters',
                             'FacesToDetach:=', face_ids, 'DetachFaces:=', False]],
        ['CreateGroupsForNewObjects:=', False])
    created = set(editor.GetObjectsInGroup('Sheets')) - before
    if not created:
        raise ValueError('No integration sheet was copied')
    names = []
    for k, old in enumerate(sorted(created)):
        target = name + '_' + str(k)
        editor.ChangeProperty(['NAME:AllTabs', ['NAME:Geometry3DAttributeTab',
                               ['NAME:PropServers', old],
                               ['NAME:ChangedProps', ['NAME:Name', 'Value:=', target]]]])
        names.append(target)
    return names


def build(project, rec, thickness_nm, lj_H, minimum_passes=3):
    design = project.new_em_design('FiniteFilm')
    m = geometry(design, rec, True, thickness_nm)
    dim = lambda v: str(float(v)) + 'mm'
    thickness_mm = thickness_nm * 1e-6
    pads = ['PadLeft', 'PadRight']
    m.sweep_along_vector(pads, ['0mm', '0mm', dim(thickness_mm)])
    m.subtract('Air', pads, keep_originals=True)
    surface_names = {'SM': [], 'MA': [], 'SA': ['SA']}
    face_data = {}
    for name in pads:
        ids = design._modeler.GetFaceIDs(name)
        vertices = {int(face): [[float(v) for v in design._modeler.GetVertexPosition(int(i))]
                               for i in design._modeler.GetVertexIDsFromFace(int(face))]
                    for face in ids}
        bottom, other = partition_faces(vertices, frozen.SUBSTRATE_MM, thickness_nm)
        all_vertices = np.concatenate(list(vertices.values()))
        if not np.isclose(np.ptp(all_vertices[:, 2]), thickness_mm, rtol=1e-5, atol=1e-10):
            raise ValueError('COM solid thickness differs from requested film thickness')
        face_data[name] = dict(vertices_mm=vertices, SM_face_ids=bottom, MA_face_ids=other,
                               measured_thickness_nm=float(np.ptp(all_vertices[:, 2]) * 1e6))
        for label, faces in (('SM', bottom), ('MA', other)):
            surface_names[label].extend(detach_surface(design, name, faces, label + '_' + name))
    lx, ly, _ = map(float, frozen.SEARCH_BOX_MM)
    z0 = float(frozen.SUBSTRATE_MM)
    m.draw_polyline([[dim(x), dim(y), dim(z0)] for x, y in
                     ((0, 0), (lx, 0), (lx, ly), (0, ly))],
                    name='SA', material='vacuum', nonmodel=True, solve_inside=True)
    m.subtract('SA', surface_names['SM'], keep_originals=True)
    # Existing sheet-PEC assignments survive extrusion by object identity.
    m.mesh_length('FinitePadMesh', pads, MaxLength='0.01mm', RefineInside=False,
                  Enabled=True, RestrictElem=False, RestrictLength=True)
    coords = np.asarray(rec.junction.coords) / 1000 + np.array([lx / 2, ly / 2])
    x0, x1 = coords[0, 0], coords[-1, 0]
    y = coords[0, 1]
    width = frozen.JUNCTION_WIDTH_UM / 1000
    zj = z0 + thickness_mm / 2
    junction = m.draw_polyline([[dim(v) for v in p] for p in (
        (x0, y - width / 2, zj), (x1, y - width / 2, zj),
        (x1, y + width / 2, zj), (x0, y + width / 2, zj))],
        name='Junction', material='vacuum', solve_inside=True)
    m._make_lumped_rlc(0, str(lj_H) + 'H', str(frozen.CJ_FF) + 'fF',
                       [dim(x0), dim(y), dim(zj)], [dim(x1), dim(y), dim(zj)],
                       ['Objects:=', [str(junction)]])
    m.draw_polyline([[dim(x0), dim(y), dim(zj)], [dim(x1), dim(y), dim(zj)]],
                    closed=False, name='JunctionVoltage', nonmodel=True)
    m.mesh_length('JunctionMesh', ['Junction'], MaxLength='0.005mm',
                  RefineInside=False, Enabled=True, RestrictElem=False, RestrictLength=True)
    setup = design.create_em_setup(min_freq_ghz=3, n_modes=1, max_delta_f=0.1,
                                   max_passes=15, min_passes=minimum_passes, min_converged=2)
    return design, setup, surface_names, face_data


def postprocess(setup, surface_names):
    from pyEPR.ansys import CalcObject

    solutions = setup.get_solutions()
    frequencies, _ = solutions.eigenmodes()
    solutions.set_mode(1)

    def qty(name):
        return CalcObject([], setup).getQty(name)

    def area(name):
        return CalcObject([('EnterScalar', 1)], setup).integrate_surf(name).evaluate()

    def scalar_abs2(expression):
        # AEDT's Mag operation applies to vectors; use the scalar product.
        return (expression.conj() * expression).real()

    # AEDT 2022 exposes D as the vector E*eps, rather than EnterQty('D').
    bulk_integral = qty('E').conj().dot(qty('E').times_eps()).real().integrate_vol().evaluate()
    # This model's straight junction line is parallel to x. Integrate each
    # phasor part before squaring; integral(|E|^2) is a different quantity.
    voltage_real = qty('E').scalar_x().real().integrate_line('JunctionVoltage').evaluate()
    voltage_imag = qty('E').scalar_x().imag().integrate_line('JunctionVoltage').evaluate()
    voltage2 = voltage_real ** 2 + voltage_imag ** 2
    if not np.isfinite(voltage2) or voltage2 <= 0:
        raise ValueError('Nonpositive junction voltage squared')
    junction_integral = frozen.CJ_FF * 1e-15 * voltage2
    denominator = bulk_integral + junction_integral
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('Nonpositive total electric-energy integral')
    rows = []
    p = {}
    for label in ('SM', 'MA', 'SA'):
        numerator = 0.0
        ti = frozen.T_INTERFACE_NM[label] * 1e-9
        eps_i = frozen.INTERFACE_EPS_R[label]
        for name in surface_names[label]:
            row = dict(label=label, object=name, area_m2=area(name))
            if label == 'SA':
                et = scalar_abs2(qty('E').scalar_x()) + scalar_abs2(qty('E').scalar_y())
                et2 = et.integrate_surf(name).evaluate()
                dn2 = scalar_abs2(qty('E').times_eps().scalar_z()).integrate_surf(name).evaluate()
                part = ti * (EPS0 * eps_i * et2 + dn2 / (EPS0 * eps_i))
                row.update(tangential_E2_integral=et2, normal_D2_integral=dn2)
            else:
                e2 = qty('E').norm_2().integrate_surf(name).evaluate()
                eps_bulk = frozen.EPS_R_SUB if label == 'SM' else 1.0
                part = ti * EPS0 * eps_bulk ** 2 / eps_i * e2
                row['E2_integral'] = e2
            row['participation'] = float(part / denominator)
            numerator += part
            rows.append(row)
        p['p_' + label] = float(numerator / denominator)
    return dict(frequencies_GHz=np.asarray(frequencies).reshape(-1).tolist(),
                f_eigen_Hz=eigenmode_frequency_hz(frequencies),
                bulk_electric_energy_integral=bulk_integral,
                junction_electric_energy_integral=junction_integral,
                junction_voltage_squared=voltage2, denominator=denominator,
                modal_capacitance_fF=bulk_integral / voltage2 * 1e15,
                surface_integrals=rows, participation=p)


def find_project(desktop, path):
    path = Path(path).resolve()
    for project in desktop.get_projects():
        candidate = Path(project.get_path()) / (project._project.GetName() + '.aedt')
        if candidate.resolve() == path:
            return project
    from pyEPR.ansys import HfssProject
    return HfssProject(desktop, desktop._desktop.OpenProject(str(path)))


def save_diagnostic_report(path, report, desktop, project):
    """Commit scientific results even if AEDT's optional message API fails."""
    write_json(path, report)
    try:
        report['messages'] = (list(desktop.get_messages(
            project._project.GetName(), 'FiniteFilm', 2))[-20:] if project is not None else [])
    except Exception as error:
        report['messages'] = []
        report['message_collection_error'] = type(error).__name__+': '+str(error)
    write_json(path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--thickness-nm', type=float, default=200.0)
    parser.add_argument('--postprocess-only', action='store_true')
    parser.add_argument('--refine-from', type=Path,
                        help='Copy a saved model for additional adaptive passes')
    parser.add_argument('--minimum-passes', type=int, default=9,
                        help='Minimum adaptive passes when refining a saved model')
    parser.add_argument('--initial-minimum-passes', type=int, default=3,
                        help='Minimum adaptive passes for a newly created model')
    parser.add_argument('--etch-um', type=float, default=0.0)
    args = parser.parse_args()
    if not np.isfinite(args.thickness_nm) or args.thickness_nm <= 0:
        parser.error('Film thickness must be finite and positive')
    if not 3 <= args.minimum_passes <= 15:
        parser.error('Minimum adaptive passes must be in [3, 15]')
    if not 3 <= args.initial_minimum_passes <= 15:
        parser.error('Initial minimum adaptive passes must be in [3, 15]')
    if args.etch_um not in (0.0, frozen.ETCH_UM, -frozen.ETCH_UM):
        parser.error('Use only the frozen nominal and two etch scenarios')
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    suffix = '_refine' + str(args.minimum_passes) if args.refine_from else ''
    project_path = folder / ('finite_film_' + str(args.thickness_nm).replace('.', 'p') + 'nm' + suffix + '.aedt')
    model_path, report_path = folder / 'model.json', folder / 'report.json'
    if not args.postprocess_only and (model_path.exists() or project_path.exists()):
        parser.error('Use --postprocess-only or a new result folder; saved results are preserved')
    desktop = connect()
    start = time.perf_counter()
    report = dict(provenance=provenance(), ansys_version=desktop.get_version(),
                  G1_passed=False, thickness_nm=args.thickness_nm, etch_um=args.etch_um,
                  participation_mesh_converged=False,
                  model='Finite PEC body; postprocessed assumed thin dielectric interfaces',
                  oxide_layer_thickness_nm=frozen.T_INTERFACE_NM,
                  metal_corner_radius_nm=0.0,
                  note='One rectangle only; participation convergence and full G1 matrix remain required.')
    project = None
    try:
        if args.postprocess_only:
            meta = json.loads(model_path.read_text(encoding='utf-8'))
            if meta['thickness_nm'] != args.thickness_nm:
                raise ValueError('Thickness differs from the saved model')
            if meta.get('etch_um', 0.0) != args.etch_um:
                raise ValueError('Etch direction differs from the saved model')
            if meta['provenance']['frozen_config'] != frozen.frozen_hash():
                raise ValueError('Frozen parameters differ from the saved model')
            project_path = Path(meta['project_path'])
            project = find_project(desktop, project_path)
            design = project.get_design('FiniteFilm')
            setup = design.get_setup()
        else:
            if args.refine_from:
                source = args.refine_from.resolve()
                source_meta = source / 'model.json'
                meta = json.loads(source_meta.read_text(encoding='utf-8'))
                if (meta['thickness_nm'] != args.thickness_nm or
                        meta.get('etch_um', 0.0) != args.etch_um or
                        meta['provenance']['frozen_config'] != frozen.frozen_hash()):
                    raise ValueError('Refinement must retain thickness and frozen parameters')
                original = Path(meta['project_path'])
                original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
                project = find_project(desktop, original)
                project.save(str(project_path))
                if hashlib.sha256(original.read_bytes()).hexdigest() != original_hash:
                    raise ValueError('Original project changed during SaveAs')
                design = project.get_design('FiniteFilm')
                setup = design.get_setup()
                setup._setup_module.EditSetup(setup.name, [
                    'NAME:' + setup.name, 'MinimumFrequency:=', '3GHz',
                    'NumModes:=', 1, 'MaxDeltaFreq:=', 0.1,
                    'ConvergeOnRealFreq:=', True, 'MaximumPasses:=', 15,
                    'MinimumPasses:=', args.minimum_passes, 'MinimumConvergedPasses:=', 2,
                    'PercentRefinement:=', 30, 'IsEnabled:=', True, 'BasisOrder:=', -1])
                meta.update(parent_model_directory=str(source),
                            parent_model_sha256=hashlib.sha256(source_meta.read_bytes()).hexdigest(),
                            parent_project_sha256=original_hash, minimum_passes=args.minimum_passes,
                            project_path=str(project_path), provenance=provenance())
            else:
                nominal_rec = library.rect_record(200, 300, 0, note='finite-thickness-calibration')
                rec, etch_error = apply_etch(nominal_rec, args.etch_um)
                if etch_error:
                    raise ValueError(etch_error)
                # Retain the existing same-device Q3D junction match for diagnosis.
                reference = json.loads(Path('robust_design_lab/results/calibration_v3_ansys_plasma_compare/'
                                          '200nm/round0p0/report.json').read_text(encoding='utf-8'))
                nominal = reference['rows'][0]
                lj = nominal['lj0_H']
                project = desktop.new_project()
                design, setup, surfaces, face_data = build(
                    project, rec, args.thickness_nm, lj, args.initial_minimum_passes)
                meta = dict(geometry_hash=rec.geometry_hash, thickness_nm=args.thickness_nm,
                            nominal_geometry_hash=nominal_rec.geometry_hash, etch_um=args.etch_um,
                            initial_minimum_passes=args.initial_minimum_passes,
                            projected_area_um2=sum(p.area for p in rec.polys),
                            surfaces=surfaces, metal_faces=face_data, lj0_H=lj,
                            reference_q3d_C_q_fF=nominal['q3d']['C_q_fF'],
                            project_path=str(project_path), provenance=provenance())
            write_json(model_path, meta)
            project.save(str(project_path))
            print('SOLVE finite PEC film', args.thickness_nm, 'nm', flush=True)
            solve_start = time.perf_counter()
            setup.analyze()
            project.save()
            write_json(folder / 'solve.json', dict(
                analysis_returned=True, elapsed_s=time.perf_counter() - solve_start,
                project_path=str(project_path), provenance=provenance()))
        report['model_geometry'] = meta
        report['fields'] = postprocess(setup, meta['surfaces'])
        metric = transmon_metrics(meta['reference_q3d_C_q_fF'] * 1e-15, meta['lj0_H'])
        report['reference_sheet_plasma_Hz'] = plasma_frequency_hz(metric)
        report['diagnostic_completed'] = True
        convergence = setup.get_convergence()[1]
        (folder / 'convergence.txt').write_text(convergence, encoding='utf-8')
        report['frequency_converged'] = bool(re.search(r'Converged\s*:\s*Yes', convergence))
        if meta.get('parent_model_directory'):
            parent_path = Path(meta['parent_model_directory']) / 'report.json'
            previous = json.loads(parent_path.read_text(encoding='utf-8'))
            changes = {k: abs(v / previous['fields']['participation'][k] - 1)
                       for k, v in report['fields']['participation'].items()}
            report['refinement_comparison'] = dict(
                parent_report_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                participation_relative_change=changes,
                single_geometry_participation_check_passed=max(changes.values()) <= 0.05,
                note='One nominal geometry only; full G1 matrix and model alignment still required.')
    except Exception as error:
        report.update(diagnostic_completed=False, failed=type(error).__name__ + ': ' + str(error),
                      traceback=traceback.format_exc())
    report['elapsed_s'] = time.perf_counter() - start
    save_diagnostic_report(report_path, report, desktop, project)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('provenance', 'model_geometry')}, ensure_ascii=True), flush=True)
    if not report.get('diagnostic_completed', False):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

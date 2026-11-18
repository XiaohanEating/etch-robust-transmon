"""Archived finite-PEC diagnostics; no search and no automatic G1 pass."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..configs import frozen
from ..evaluator import finite_film
from ..geometry import library
from ..perturbation.channels import apply_etch
from ..runtime.records import provenance, write_json
from .field_audit import load_solution, save_solution, sha256


def mesh_checks(fs, surfaces, rec, settings):
    """Check full physical coverage, finite thickness, and dielectric volumes."""
    p, t = fs.est._p, fs.est._t
    vertices = p[:, t].transpose(2, 1, 0)
    volumes = abs(np.linalg.det(vertices[:, 1:] - vertices[:, :1])) / 6
    lx, ly, lz = frozen.SEARCH_BOX_MM
    area = sum(poly.area for poly in rec.polys) * 1e-6
    perimeter = sum(poly.length for poly in rec.polys) * 1e-3
    thickness = settings.thickness_nm * 1e-6
    expected_areas = dict(SM=area, MA=area + perimeter * thickness, SA=lx*ly-area)
    areas = {}
    for name, tris in surfaces.items():
        v = p[:, tris].transpose(1, 2, 0)
        areas[name] = float(np.linalg.norm(np.cross(v[:, 1]-v[:, 0],
                                                   v[:, 2]-v[:, 0]), axis=1).sum()/2)
    area_errors = {k: abs(areas[k]/expected_areas[k]-1) for k in areas}
    measured = {str(mat): float(volumes[fs.package_mesh.material == mat].sum())
                for mat in (0, 1)}
    expected_volumes = {'0': lx*ly*(lz-frozen.SUBSTRATE_MM)-area*thickness,
                        '1': lx*ly*frozen.SUBSTRATE_MM}
    volume_errors = {k: abs(measured[k]/expected_volumes[k]-1) for k in measured}
    metal_z = p[2, np.unique(np.concatenate(list(fs.package_mesh.metal_tris.values())))]
    height = float(np.ptp(metal_z))
    if (max(area_errors.values()) > 1e-7 or max(volume_errors.values()) > 1e-7
            or not np.isclose(height, thickness, rtol=1e-7, atol=1e-12)):
        raise ValueError('Finite PEC physical coverage/volume/thickness check failed')
    return dict(areas_mm2=areas, expected_areas_mm2=expected_areas,
                area_relative_errors=area_errors, dielectric_volumes_mm3=measured,
                expected_volumes_mm3=expected_volumes, volume_relative_errors=volume_errors,
                measured_thickness_nm=height*1e6, tetrahedra=t.shape[1],
                degrees_of_freedom=fs.est._basis.N)


def evaluate(folder, rec, settings, lj_H, transported_from=None):
    """Solve once, verify archive identity, then replay interface integration."""
    folder = Path(folder)
    meta_path = folder / 'finite_model.json'
    identity = dict(geometry_hash=rec.geometry_hash, settings=asdict(settings),
                    evaluator_version=finite_film.VERSION, frozen_config=frozen.frozen_hash(),
                    evaluator_sha256=sha256(Path(finite_film.__file__)))
    if transported_from is not None:
        from ..evaluator import mesh_transport
        parent = Path(transported_from).resolve()
        identity['mesh_transport'] = dict(version=mesh_transport.VERSION, parent=str(parent),
            source_sha256=sha256(Path(mesh_transport.__file__)),
            parent_sha256={name: sha256(parent/name) for name in
                           ('finite_model.json', 'solution.json', 'solution.npz',
                            'stiffness.npz', 'surfaces.npz')})
    start = time.perf_counter()
    if (folder / 'solution.json').exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['identity'] != identity:
            raise ValueError('Existing finite archive uses different geometry/configuration/source')
        if sha256(folder / 'surfaces.npz') != meta['surfaces_sha256']:
            raise ValueError('Interface archive checksum differs')
        fs, saved_rec, _ = load_solution(folder)
        if saved_rec.geometry_hash != rec.geometry_hash:
            raise ValueError('Requested geometry differs from archived geometry')
        with np.load(folder / 'surfaces.npz', allow_pickle=False) as saved:
            surfaces = {k: saved[k] for k in saved.files}
    else:
        if meta_path.exists():
            raise FileExistsError('Incomplete archive exists; preserve it and use a new folder')
        if transported_from is None:
            fs, surfaces = finite_film.solve_fields(rec, settings)
        else:
            fs, surfaces = mesh_transport.solve_from_archive(transported_from, rec, settings)
        checks = mesh_checks(fs, surfaces, rec, settings)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'surfaces.npz', **surfaces)
        write_json(meta_path, dict(identity=identity, mesh_checks=checks,
                                  surfaces_sha256=sha256(folder / 'surfaces.npz')))
        save_solution(folder, fs, rec, asdict(settings))
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    metrics = finite_film.surface_participation(fs, surfaces, lj_H)
    report = dict(identity=identity, lj_H=lj_H, metrics=metrics, mesh_checks=meta['mesh_checks'],
                  mesh_time_s=fs.mesh_time_s, solve_time_s=fs.solve_time_s,
                  elapsed_this_run_s=time.perf_counter()-start,
                  phi_sha256=hashlib.sha256(__import__(
                      'robust_design_lab.evaluator.fields', fromlist=['ElectrostaticField']
                      ).ElectrostaticField(fs).phi.tobytes()).hexdigest(),
                  archive_sha256={k: sha256(folder/k) for k in
                                  ('solution.npz', 'stiffness.npz', 'surfaces.npz')},
                  G1_passed=False, provenance=provenance())
    write_json(folder / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--h-edge', type=float, default=0.005)
    parser.add_argument('--etch-um', type=float, default=0.0)
    parser.add_argument('--thickness-nm', type=float, default=200.0)
    parser.add_argument('--transport-from', type=Path,
                        help='Paired rectangle diagnostic using a saved nominal mesh')
    args = parser.parse_args()
    if args.etch_um not in (0, -frozen.ETCH_UM, frozen.ETCH_UM):
        parser.error('Use only the frozen nominal and two etch scenarios')
    nominal = library.rect_record(200, 300, 0, note='finite-alignment-reference')
    rec, error = apply_etch(nominal, args.etch_um)
    if error:
        parser.error(error)
    reference = Path('robust_design_lab/results/hfss_finite_film_v1/200nm_refine13/model.json')
    lj_H = json.loads(reference.read_text(encoding='utf-8'))['lj0_H']
    settings = finite_film.MeshSettings(thickness_nm=args.thickness_nm, h_edge_mm=args.h_edge)
    row = evaluate(args.output, rec, settings, lj_H, transported_from=args.transport_from)
    print(json.dumps({k: row[k] for k in ('metrics', 'mesh_checks', 'mesh_time_s',
                                         'solve_time_s', 'phi_sha256')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

"""Per-scenario persistence, immutable batch identity and official LOM summaries."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

from robust_design_lab.geometry import novel_structures as geo
from robust_design_lab.geometry.readout import ReadoutLayout, attach_readout, etch_loaded
from robust_design_lab.evaluator.purcell import PurcellCircuit, evaluate_matrix, lifetime_margins
from robust_design_lab.runtime.records import write_json

ROOT = Path(__file__).resolve().parents[2]
OBJECTS = ("pad", "annulus_edited", "idc")
SCENARIOS = {"nom": 0.0, "ep": 0.5, "em": -0.5}
NODE_ORDER = ["ground_main_plane", "pad_left_Q1", "pad_right_Q1", "readout_Q1"]


def candidates():
    from robust_design_lab.experiments.verify_candidates import simple_dual_pad, load_champion_action
    from robust_design_lab.geometry import boundary_edit as be
    action, historical_d = load_champion_action()
    source = ROOT / "results/region_edit_pilot_2026-11-03/runs.jsonl"
    ring = geo.q_annulus(r_out=253.9, r_in=121.5, gap=33.8)
    return {
        "pad": (simple_dual_pad(w=734, h=218), {"w": 734, "h": 218}),
        "annulus_edited": (be.apply_edit(ring, *be.action_from_kw(action)),
                           {"r_out": 253.9, "r_in": 121.5, "gap": 33.8,
                            "action": action, "historical_isolated_D": historical_d,
                            "action_source": str(source),
                            "action_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}),
        "idc": (geo.q_ellipse_idc(a=177, b=140, D=140, n_f=4,
                                  f_w=16, f_gap=32, f_len=88, jg=40),
                {"a": 177, "b": 140, "D": 140, "n_f": 4,
                 "f_w": 16, "f_gap": 32, "f_len": 88, "jg": 40})}


def source_hashes():
    paths = [p for p in ROOT.rglob("*.py") if not
             {"results", "tests", "__pycache__"}.intersection(p.relative_to(ROOT).parts)]
    paths += list((ROOT.parent / "src/qiskit_metal").rglob("*.py"))
    return {str(p.relative_to(ROOT.parent)).replace("\\", "/"):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def identity(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def build_manifest():
    objects = {}
    for name, (base, origin) in candidates().items():
        nominal = attach_readout(base)
        scenes = {}
        for scene, delta in SCENARIOS.items():
            struct = nominal if delta == 0 else etch_loaded(nominal, delta)
            options = geo.struct_to_options(struct)
            scenes[scene] = dict(delta_um=delta, options=options,
                                 geometry_sha256=identity(options),
                                 wkt={key: struct[key].wkt for key in
                                      ("pad_left", "pad_right", "junction", "pocket",
                                       "readout", "readout_cutout")})
        objects[name] = dict(origin=origin, scenarios=scenes)
    payload = dict(schema=1, objects=objects, circuit=asdict(PurcellCircuit()),
                   layout=asdict(ReadoutLayout()), solver=dict(percent_error=0.1, max_passes=30),
                   source_sha256=source_hashes(), python=sys.executable,
                   python_version=sys.version, node_order=NODE_ORDER,
                   targets_other_loss_us=[100, 300, 1000],
                   lifetime_qualification="raw_values_no_user_threshold",
                   conditions="one_readout_no_bus_no_Purcell_filter; uniform_metal_and_ground_etch")
    return dict(batch_sha256=identity(payload), **payload)


def ensure_manifest(run_dir, manifest):
    path = Path(run_dir) / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        # JSON turns tuple fields into lists; compare the serialized identity.
        if identity(old) != identity(manifest):
            raise ValueError("Batch configuration/source/geometry changed; use a new run directory")
    else:
        write_json(path, manifest)


def summarize(run_dir, manifest):
    summary = dict(batch_sha256=manifest["batch_sha256"], objects={})
    for name in OBJECTS:
        rows = {}
        for scene in SCENARIOS:
            path = Path(run_dir) / name / f"{scene}.json"
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
                if row["batch_sha256"] != manifest["batch_sha256"]:
                    raise ValueError("Scenario belongs to a different batch")
                rows[scene] = row
        accepted = len(rows) == 3 and all(r["status"] == "accepted" for r in rows.values())
        item = dict(complete=accepted, scenario_status={k: r["status"] for k, r in rows.items()},
                    D_f=None, nominal=rows.get("nom", {}).get("lom"), qualified=None)
        if accepted:
            f0 = rows["nom"]["lom"]["f01_GHz"]
            item["D_f"] = max(abs(math.log(r["lom"]["f01_GHz"] / f0)) for r in rows.values())
            item["signed_shift_MHz"] = {s: 1000 * (r["lom"]["f01_GHz"] - f0)
                                        for s, r in rows.items()}
            item["T1_Purcell_us"] = {s: r["lom"]["T1_Purcell_us"] for s, r in rows.items()}
            item["nominal_lifetime_margins"] = lifetime_margins(rows["nom"]["lom"]["T1_Purcell_us"])
        summary["objects"][name] = item
    write_json(Path(run_dir) / "summary.json", summary)
    return summary


def run_batch(run_dir, objects=OBJECTS, scenarios=tuple(SCENARIOS), *, solver=None):
    if not set(objects) <= set(OBJECTS) or not set(scenarios) <= set(SCENARIOS):
        raise ValueError("Unknown object or scenario")
    if solver is None:
        from robust_design_lab.experiments.q3d_shape_sweep import run_q3d
        solver = run_q3d
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents simultaneous consumers of a partially saved run.
    lock = run_dir / "run.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"{datetime.now(timezone.utc).isoformat()}\n{sys.executable}\n")
    try:
        manifest = build_manifest()
        ensure_manifest(run_dir, manifest)
        for name in objects:
            for scene in scenarios:
                path = run_dir / name / f"{scene}.json"
                if path.exists():
                    old = json.loads(path.read_text(encoding="utf-8"))
                    if old["batch_sha256"] != manifest["batch_sha256"]:
                        raise ValueError("Cached scenario identity mismatch")
                    if old["status"] == "running":
                        raise ValueError(f"Interrupted scenario needs review before retry: {path}")
                    continue
                spec = manifest["objects"][name]["scenarios"][scene]
                row = dict(object=name, scenario=scene, batch_sha256=manifest["batch_sha256"],
                           geometry_sha256=spec["geometry_sha256"], status="running",
                           started_utc=datetime.now(timezone.utc).isoformat())
                write_json(path, row)
                print(f"Starting {name}/{scene}", flush=True)
                start = time.perf_counter()
                try:
                    raw = solver(spec["options"], "NovelTwoPadTransmon", f"purcell_{name}_{scene}",
                                 max_passes=30, percent_error=0.1,
                                 output_dir=run_dir / name / scene, legacy_c_sigma=False)
                    row["raw"] = raw
                    write_json(path, row)
                    if raw.get("project_saved") is not True:
                        raise ValueError("Solved project was not saved")
                    row["lom"] = evaluate_matrix(raw["matrix_fF"], raw["nets"], NODE_ORDER,
                                                  converged=raw.get("converged"))
                    row["status"] = "accepted"
                except Exception as exc:
                    row.update(status="rejected", error=repr(exc), traceback=traceback.format_exc())
                row["elapsed_s"] = time.perf_counter() - start
                write_json(path, row)
                summarize(run_dir, manifest)
                print(f"Saved {name}/{scene}: {row['status']} ({row['elapsed_s']:.1f}s)", flush=True)
        return summarize(run_dir, manifest)
    finally:
        lock.unlink()

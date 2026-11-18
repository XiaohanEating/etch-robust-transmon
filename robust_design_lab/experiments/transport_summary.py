"""Summarize paired-mesh transport diagnostics without changing the G1 gate."""

import argparse
import json
import math
from pathlib import Path

from ..configs import frozen
from ..runtime.records import provenance, write_json
from .field_audit import sha256


SCENES = ("nominal", "etch_plus", "etch_minus")
PARTS = ("p_SM", "p_MA", "p_SA")
CRITERIA = dict(capacitance_relative=0.02, participation_relative=0.05,
                drift_absolute=0.005)


def _load(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")), sha256(path)


def _paths(root, base, level):
    """Return the paired nominal/transport reports for one mesh level."""
    root = Path(root)
    base = Path(base)
    if level == "h0p005":
        return {
            "nominal": base / "h0p005" / "nominal" / "report.json",
            "etch_plus": root / "etch_plus" / "report.json",
            "etch_minus": root / "etch_minus" / "report.json",
        }
    if level == "h0p01":
        return {
            "nominal": base / "h0p01" / "nominal" / "report.json",
            "etch_plus": root / "h0p01_etch_plus" / "report.json",
            "etch_minus": root / "h0p01_etch_minus" / "report.json",
        }
    raise ValueError(f"unknown mesh level: {level}")


def _metrics(paths):
    rows, hashes = {}, {}
    missing = []
    for scene, path in paths.items():
        if not path.exists():
            missing.append(f"{scene}: {path}")
            continue
        report, digest = _load(path)
        rows[scene] = report["metrics"]
        hashes[str(path)] = digest
    return rows, hashes, missing


def _level_comparison(coarse, fine):
    relative = {}
    for scene in SCENES:
        a, b = coarse[scene], fine[scene]
        relative[scene] = {
            key: abs(b[key] / a[key] - 1.0)
            for key in ("C_q_fF", *PARTS, "T1_us")
        }

    drifts = {}
    for scene in ("etch_plus", "etch_minus"):
        a = math.log(coarse[scene]["T1_us"] / coarse["nominal"]["T1_us"])
        b = math.log(fine[scene]["T1_us"] / fine["nominal"]["T1_us"])
        drifts[scene] = {
            "h0p005_log_ratio": a,
            "h0p01_log_ratio": b,
            "signed_difference": abs(b - a),
            "absolute_drift_difference": abs(abs(b) - abs(a)),
            "direction_same": bool(a * b > 0 or a == b == 0),
        }

    max_capacitance = max(row["C_q_fF"] for row in relative.values())
    max_participation = max(row[key] for row in relative.values() for key in PARTS)
    max_t1 = max(row["T1_us"] for row in relative.values())
    max_drift = max(row["signed_difference"] for row in drifts.values())
    return {
        "relative_change_between_levels": relative,
        "etch_log_drift_difference": drifts,
        "maxima": {
            "capacitance_relative": max_capacitance,
            "participation_relative": max_participation,
            "T1_relative": max_t1,
            "drift_difference": max_drift,
        },
        "checks": {
            "capacitance_level_alignment_passed":
                max_capacitance <= CRITERIA["capacitance_relative"],
            "participation_level_alignment_passed":
                max_participation <= CRITERIA["participation_relative"],
            "drift_difference_passed":
                max_drift <= CRITERIA["drift_absolute"],
        },
        "diagnostic_passed": bool(
            max_capacitance <= CRITERIA["capacitance_relative"]
            and max_participation <= CRITERIA["participation_relative"]
            and max_drift <= CRITERIA["drift_absolute"]
        ),
    }


def build_summary(root, base):
    root, base = Path(root), Path(base)
    levels, source_hashes, missing = {}, {}, []
    for level in ("h0p005", "h0p01"):
        rows, hashes, absent = _metrics(_paths(root, base, level))
        levels[level] = rows
        source_hashes.update(hashes)
        missing.extend(f"{level}/{item}" for item in absent)

    comparison = None
    if not missing:
        comparison = _level_comparison(levels["h0p005"], levels["h0p01"])

    return {
        "protocol": "finite_alignment_transport_v2",
        "description": "Same nominal finite-PEC mesh transported to +/- etch rectangles at two frozen edge resolutions.",
        "G1_passed": False,
        "settings": {
            "thickness_nm": 200.0,
            "h_edge_mm": [0.005, 0.01],
            "etch_um": [0.0, frozen.ETCH_UM, -frozen.ETCH_UM],
            "mapping": "rectangle-piecewise-affine-v1",
        },
        "criteria": CRITERIA,
        "levels": levels,
        "level_comparison": comparison,
        "missing": missing,
        "source_sha256": source_hashes,
        "interpretation": {
            "drift_stability": (
                "The two transported mesh levels give same-sign etch T1 drift and "
                "their signed drift differences are below 0.005."
                if comparison and comparison["checks"]["drift_difference_passed"]
                else "Drift stability could not be established."
            ),
            "absolute_participation": (
                "Absolute participation remains mesh-level dependent; this diagnostic "
                "cannot pass the participation convergence gate."
                if not comparison or not comparison["checks"]["participation_level_alignment_passed"]
                else "Absolute participation meets the paired-level diagnostic tolerance."
            ),
            "scope": "Axis-aligned rectangle only; 200 nm remains a provisional process-thickness assumption.",
        },
        "provenance": provenance(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="finite_alignment_transport_v1 result directory")
    parser.add_argument("--base", type=Path,
                        default=Path("robust_design_lab/results/finite_alignment_v1/rect_reference"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_summary(args.root, args.base)
    write_json(args.output, report)
    print(json.dumps({
        "missing": report["missing"],
        "level_comparison": report["level_comparison"],
        "G1_passed": report["G1_passed"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

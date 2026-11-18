"""Physical regression checks for the common-readout evaluation path."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from robust_design_lab.evaluator.purcell import (
    PurcellCircuit, evaluate_matrix, lifetime_from_modes)
from robust_design_lab.experiments.purcell_qualification import batch
from robust_design_lab.geometry import novel_structures as geo
from robust_design_lab.geometry.readout import attach_readout, etch_loaded

CASES = json.loads((batch.ROOT / "results/native_q3d_lom_review_2026-10-23/replay.json")
                   .read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES)
def test_official_tutorial_and_net_permutation(case):
    settings = case["settings"]
    config = PurcellCircuit(bus_GHz=tuple(settings["bus_GHz"]))
    names = case["matrix_nodes"]
    mat = np.array(case["matrix_values"])
    result = evaluate_matrix(mat, names, names, converged=True, circuit=config)
    expected = case["current_native_LOManalysis"]
    assert result["f01_GHz"] == pytest.approx(expected["comparison"]["fQ"]["current"], abs=1e-9)
    assert result["T1_Purcell_us"] == pytest.approx(expected["printed_Purcell_T1_us"], abs=5.1e-7)
    ix = list(reversed(range(len(names))))
    permuted = evaluate_matrix(mat[np.ix_(ix, ix)], [names[i] for i in ix], names,
                               converged=True, circuit=config)
    assert result == permuted


@pytest.mark.parametrize("converged", [False, None])
def test_nonconvergence_rejected(converged):
    c = CASES[1]
    with pytest.raises(ValueError, match="converged"):
        evaluate_matrix(c["matrix_values"], c["matrix_nodes"], c["matrix_nodes"], converged=converged)


def test_network_and_loss_contract():
    c = CASES[1]
    n = c["matrix_nodes"]
    for names in (n[:-1], [n[0], n[1], n[1], n[3]]):
        with pytest.raises(ValueError, match="network"):
            evaluate_matrix(c["matrix_values"], names, n, converged=True)
    with pytest.raises(ValueError, match="official"):
        evaluate_matrix(c["matrix_values"], n, n, converged=True,
                        circuit=replace(PurcellCircuit(), readout_Q=20000))
    with pytest.raises(ValueError, match="zero coupling"):
        lifetime_from_modes(4.5, [0], [7], [10000])


def test_lifetimes_add_as_rates():
    total, modes = lifetime_from_modes(4.5, [50, 30], [7, 6], [10000, 100000])
    assert total == pytest.approx(1 / sum(1 / x for x in modes))
    assert total < min(modes)


@pytest.mark.parametrize("name", batch.OBJECTS)
def test_loaded_geometry_export_and_etch(name):
    from robust_design_lab.geometry.four_shape_families import build_design_for
    base, _ = batch.candidates()[name]
    nominal = attach_readout(base)
    for delta in (0.0, 0.5, -0.5):
        s = nominal if delta == 0 else etch_loaded(nominal, delta)
        assert s["readout"].equals(nominal["readout"].buffer(-delta, quad_segs=8))
        assert s["pocket"].equals(nominal["pocket"].buffer(delta, join_style=2))
        assert s["readout_cutout"].equals(nominal["readout_cutout"].buffer(delta, join_style=2))
        for key in ("pad_left", "pad_right"):
            assert len(s[key].interiors) == len(base[key].interiors)
        design = build_design_for(geo.struct_to_options(s), "NovelTwoPadTransmon")
        table = design.qgeometry.tables["poly"]
        for key in ("pad_left", "pad_right", "readout", "readout_cutout"):
            actual = table[table["name"] == key].iloc[0]["geometry"]
            from shapely.affinity import scale
            expected = scale(s[key], 0.001, 0.001, origin=(0, 0))
            # Metal rounds coordinates to its declared grid on table insertion.
            grid_mm = 10.0 ** -design.template_options.PRECISION
            assert actual.hausdorff_distance(expected) <= grid_mm + 1e-14
            assert actual.symmetric_difference(expected).area <= expected.length * grid_mm + 1e-14


def test_manifest_rejects_changed_source(tmp_path):
    batch.ensure_manifest(tmp_path, {"batch_sha256": "old"})
    with pytest.raises(ValueError, match="changed"):
        batch.ensure_manifest(tmp_path, {"batch_sha256": "new"})


def test_manifest_resume_after_json_roundtrip(tmp_path):
    manifest = {"batch_sha256": "same", "circuit": {"bus_GHz": ()}}
    batch.ensure_manifest(tmp_path, manifest)
    batch.ensure_manifest(tmp_path, manifest)


def test_failed_solve_keeps_record_and_no_finite_D(tmp_path, monkeypatch):
    manifest = {"batch_sha256": "test", "objects": {
        "pad": {"scenarios": {"nom": {"options": {}, "geometry_sha256": "nom"}}}}}
    monkeypatch.setattr(batch, "build_manifest", lambda: manifest)
    def failing_solver(*args, **kwargs):
        raise RuntimeError("controlled solver failure")
    out = batch.run_batch(tmp_path, ("pad",), ("nom",), solver=failing_solver)
    row = json.loads((tmp_path / "pad/nom.json").read_text(encoding="utf-8"))
    assert row["status"] == "rejected" and "controlled solver failure" in row["error"]
    assert out["objects"]["pad"]["D_f"] is None
    assert out["objects"]["pad"]["complete"] is False
    assert not (tmp_path / "run.lock").exists()


def test_readout_linewidth_and_field_energy_units():
    from robust_design_lab.analysis.purcell_readout_requirements import resonator_times
    times = resonator_times(7, 120000)
    assert times["linewidth_kHz"] == pytest.approx(58.3333333333)
    # Energy decays as exp(-kappa*t); the field amplitude as exp(-kappa*t/2).
    assert times["energy_decay_us"] == pytest.approx(120000 / (2 * np.pi * 7000))
    assert times["field_amplitude_decay_us"] == pytest.approx(2 * times["energy_decay_us"])
    with pytest.raises(ValueError):
        resonator_times(7, 0)


@pytest.mark.parametrize("name", batch.OBJECTS)
def test_readout_requirements_recover_target_in_both_routes(name):
    from robust_design_lab.analysis.purcell_readout_requirements import design_requirements
    run = "run01" if name == "pad" else "run02"
    path = batch.ROOT / "results/purcell_qualification_2026-11-08" / run / name / "nom.json"
    lom = json.loads(path.read_text(encoding="utf-8"))["lom"]
    req = design_requirements(lom)
    f, g, fr = lom["f01_GHz"], abs(lom["g_MHz"][0]), lom["circuit"]["readout_GHz"]
    q_req = req["narrow_linewidth"]["required_loaded_Q"]
    g_req = req["weak_coupling"]["max_abs_g_over_2pi_MHz"]
    # Independently express the official formula entirely in cyclic MHz.
    def lifetime_us(q, coupling_MHz):
        fq_MHz, fr_MHz = f * 1000, fr * 1000
        return (fr_MHz**2 - fq_MHz**2)**2 / (
            8 * np.pi * (fr_MHz / q) * coupling_MHz**2 * fq_MHz**2)
    assert lifetime_us(q_req, g) == pytest.approx(3000, rel=1e-12)
    assert lifetime_us(10000, g_req) == pytest.approx(3000, rel=1e-12)
    assert req["narrow_linewidth"]["conditional_Purcell_us"] == pytest.approx(
        lifetime_us(120000, g), rel=1e-12)
    assert lom["circuit"]["readout_Q"] == 10000

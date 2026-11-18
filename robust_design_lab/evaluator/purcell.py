"""Official floating-transmon LOM with explicit net ordering and Purcell units."""
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
import io
import math
import re

import numpy as np


@dataclass(frozen=True)
class PurcellCircuit:
    Lj_nH: float = 12.31
    Cj_fF: float = 2.0
    readout_GHz: float = 7.0
    bus_GHz: tuple = ()
    # These values are fixed in the official extraction implementation.
    readout_Q: float = 10000.0
    bus_Q: float = 100000.0
    resonator_impedance_ohm: float = 50.0


def lifetime_from_modes(f_GHz, g_MHz, resonator_GHz, quality_factors):
    """Return lifetimes in us using the exact formula in lumped_capacitive.py."""
    g = np.asarray(g_MHz, dtype=float) * (2 * np.pi * 1e6)
    wr = np.asarray(resonator_GHz, dtype=float) * (2 * np.pi * 1e9)
    quality = np.asarray(quality_factors, dtype=float)
    if g.ndim != 1 or not len(g) or g.shape != wr.shape or g.shape != quality.shape:
        raise ValueError("At least one explicit, matched resonator mode is required")
    if not (np.isfinite(g).all() and np.isfinite(wr).all() and
            np.isfinite(quality).all() and math.isfinite(f_GHz)):
        raise ValueError("Non-finite circuit value")
    if f_GHz <= 0 or np.any(wr <= 0) or np.any(quality <= 0) or np.any(g == 0):
        raise ValueError("Nonpositive circuit value or zero coupling")
    wq = 2 * np.pi * f_GHz * 1e9
    modes_us = (wr**2 - wq**2)**2 / (4 * (wr / quality) * g**2 * wq**2) * 1e6
    if np.any(modes_us <= 0) or not np.isfinite(modes_us).all():
        raise ValueError("Invalid Purcell lifetime")
    return float(1 / np.sum(1 / modes_us)), modes_us.tolist()


def evaluate_matrix(matrix_fF, nets, node_order, *, converged, circuit=PurcellCircuit()):
    """node_order is buses, ground, left pad, right pad, readout (official order)."""
    if converged is not True:
        raise ValueError("Every accepted scenario requires converged=True")
    n_modes = 1 + len(circuit.bus_GHz)
    if (len(nets) != n_modes + 3 or len(set(nets)) != len(nets) or
            len(node_order) != len(nets) or len(set(node_order)) != len(nets) or
            set(node_order) != set(nets)):
        raise ValueError("Missing, duplicated or unexpected conductor network")
    if (circuit.readout_Q != 1e4 or circuit.bus_Q != 1e5 or
            circuit.resonator_impedance_ohm != 50):
        raise ValueError("Circuit loss/impedance must match the official implementation")
    if min(circuit.Lj_nH, circuit.Cj_fF, circuit.readout_GHz,
           *circuit.bus_GHz) <= 0:
        raise ValueError("Circuit values must be positive")
    mat = np.asarray(matrix_fF, dtype=float)
    if mat.shape != (len(nets), len(nets)) or not np.isfinite(mat).all():
        raise ValueError("Invalid capacitance matrix")
    if not np.allclose(mat, mat.T, rtol=5e-3, atol=1e-6):
        raise ValueError("Capacitance matrix is not reciprocal within solver tolerance")
    if np.any(np.diag(mat) <= 0) or np.any((mat - np.diag(np.diag(mat))) > 1e-6):
        raise ValueError("Expected a Maxwell matrix in fF")
    ix = [nets.index(name) for name in node_order]
    cm = mat[np.ix_(ix, ix)] * 1e-15
    ground = n_modes - 1
    p1, p2 = ground + 1, ground + 2
    external = [ground, *range(ground), len(nets) - 1]
    c1, c2 = -cm[p1, external].sum(), -cm[p2, external].sum()
    if c1 <= 0 or c2 <= 0:
        raise ValueError("Missing shunt coupling")
    c_sigma = (-cm[p1, p2] + c1 * c2 / (c1 + c2)) * 1e15 + circuit.Cj_fF

    from qiskit_metal.analyses.quantization.constants import Ic_from_Lj
    from qiskit_metal.analyses.quantization.lumped_capacitive import (
        extract_transmon_coupled_Noscillator)
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        lom = extract_transmon_coupled_Noscillator(
            cm, Ic_from_Lj(circuit.Lj_nH, "nH", "A"), circuit.Cj_fF * 1e-15,
            n_modes, list(circuit.bus_GHz), circuit.readout_GHz, print_info=True)
    lifetime, modes = lifetime_from_modes(
        lom["fQ"], lom["gbus"], [circuit.readout_GHz, *circuit.bus_GHz],
        [circuit.readout_Q, *([circuit.bus_Q] * len(circuit.bus_GHz))])
    match = re.search(r"^T1 ([\d.eE+-]+) \[us\]", stdout.getvalue(), re.M)
    if not match or not math.isclose(lifetime, float(match[1]), abs_tol=5.1e-7, rel_tol=1e-12):
        raise ValueError("Full precision Purcell calculation disagrees with official printout")
    return dict(f01_GHz=float(lom["fQ"]), C_sigma_fF=float(c_sigma),
                EC_MHz=float(lom["EC"]), EJ_GHz=float(lom["EJ"]),
                EJ_over_EC=float(lom["EJ"] * 1000 / lom["EC"]),
                alpha_MHz=float(lom["alpha"]), g_MHz=np.asarray(lom["gbus"]).tolist(),
                chi_MHz=np.asarray(lom["chi_in_MHz"]).tolist(),
                T1_Purcell_us=lifetime, T1_modes_us=modes,
                official_stdout=stdout.getvalue(), circuit=asdict(circuit),
                node_order=list(node_order), matrix_units="fF")


def lifetime_margins(purcell_us, targets_us=(100.0, 300.0, 1000.0)):
    """Targets represent other-loss lifetimes, not a prediction of measured T1."""
    return [dict(target_other_loss_us=target, purcell_to_target=purcell_us / target,
                 combined_us=1 / (1 / target + 1 / purcell_us),
                 relative_reduction=target / (target + purcell_us))
            for target in targets_us]

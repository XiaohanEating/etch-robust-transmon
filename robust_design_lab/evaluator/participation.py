# -*- coding: utf-8 -*-
"""Dielectric interface energy with normal-D and tangential-E conditions.

Reference: Wenner et al. 2011 supplementary equations (1)-(5). The complete
chip interface is integrated; no assertion of edge convergence is implicit.
"""

import numpy as np

from ..configs import frozen
from .fields import interface_quadrature

EPS0 = 8.8541878128e-12
VERSION = 'interfaces-v2-vector-facet'


def layer_energy(E, weights_m2, mask, thickness_m, eps_interface, eps_bulk, metal):
    """Layer density: eps_i*Et^2 + eps_bulk^2/eps_i*En^2. Fields V/m."""
    E = np.asarray(E, dtype=float)
    et2 = np.sum(E[:2] ** 2, axis=0) if not metal else np.zeros(E.shape[1])
    density = eps_interface * et2 + eps_bulk ** 2 / eps_interface * E[2] ** 2
    return float(0.5 * EPS0 * thickness_m * np.sum(weights_m2[mask] * density[mask]))


def surface_participation(fs, rec, sample=None, lj_H=None, ti_scale=1.0):
    """Return surface-loss-limited T1, including junction capacitance energy.

    Diagnostics: quadrature_order, offset_um, edge_cutoff_um, sa_side. Legacy
    dz_um selects an explicit offset; step/margin no longer truncate SA area.
    """
    if not np.isfinite(ti_scale) or ti_scale <= 0:
        raise ValueError('Interface thickness multiplier must be positive')
    s = {} if sample is None else dict(sample)
    q = interface_quadrature(fs, rec, order=int(s.get('quadrature_order', 2)),
            offset_um=float(s.get('offset_um', s.get('dz_um', 0.0))),
            edge_cutoff_um=float(s.get('edge_cutoff_um', 0.0)))
    metal, sa = q['metal'] & q['keep'], ~q['metal'] & q['keep']
    source = s.get('sa_side', 'substrate')
    if source not in ('substrate', 'air'):
        raise ValueError('sa_side must be substrate or air')
    energies = {}
    for name, E, eps_bulk, mask, is_metal in (
        ('MA', q['E_air'], 1.0, metal, True),
        ('SM', q['E_sub'], frozen.EPS_R_SUB, metal, True),
        ('SA', q['E_sub'] if source == 'substrate' else q['E_air'],
         frozen.EPS_R_SUB if source == 'substrate' else 1.0, sa, False),
    ):
        energies[name] = layer_energy(E, q['weights_m2'], mask,
                frozen.T_INTERFACE_NM[name] * 1e-9 * ti_scale,
                frozen.INTERFACE_EPS_R[name], eps_bulk, is_metal)
    from .lom import transmon_metrics
    tm = transmon_metrics(fs.C_q_fF * 1e-15,
                          frozen.LJ0_NH * 1e-9 if lj_H is None else lj_H)
    dilution = fs.C_q_fF / tm['C_sigma_fF']
    p = {'p_' + name: value / q['energy_J'] * dilution for name, value in energies.items()}
    omega = 2.0 * np.pi * tm['f01_GHz'] * 1e9
    rate = omega * sum(p['p_' + name] * frozen.TAN_DELTA[name] for name in energies)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError('Nonpositive or nonfinite interface loss rate')
    et_error = np.linalg.norm((q['E_air'][:2] - q['E_sub'][:2])[:, sa])
    et_scale = np.linalg.norm(q['E_sub'][:2, sa])
    d_error = np.linalg.norm((q['E_air'][2] - frozen.EPS_R_SUB * q['E_sub'][2])[sa])
    d_scale = np.linalg.norm((frozen.EPS_R_SUB * q['E_sub'][2])[sa])
    p.update(tm)
    p.update(T1_us=1e6 / rate, C_q_fF=fs.C_q_fF, evaluator_version=VERSION,
             energy_J=q['energy_J'], junction_electric_fraction=dilution,
             integration_area_m2=q['total_area_m2'], interface_faces=q['n_faces'],
             interface_parameters=dict(eps_r=frozen.INTERFACE_EPS_R,
                thickness_nm={k: v * ti_scale for k, v in frozen.T_INTERFACE_NM.items()},
                tan_delta=frozen.TAN_DELTA),
             boundary_residual=dict(SA_tangential=float(et_error / max(et_scale, 1e-30)),
                                    SA_displacement=float(d_error / max(d_scale, 1e-30))))
    return p

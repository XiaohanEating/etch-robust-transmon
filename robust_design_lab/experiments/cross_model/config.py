"""The registered task definitions; every method receives the same constraints."""
import hashlib
import json

from robust_design_lab.geometry import novel_structures as geo
from robust_design_lab.experiments.verify_candidates import simple_dual_pad

TASKS = {
    'pad': dict(start=dict(w=734.0, h=218.0, gap=30.0),
                box=dict(w=(650., 820.), h=(180., 260.), gap=(20., 45.)),
                reference='refs_b_pad_ref.json'),
    'idc': dict(start=dict(a=177., b=140., D=140., f_w=16., f_gap=32., f_len=88.),
                box=dict(a=(150., 210.), b=(125., 155.), D=(120., 160.),
                         f_w=(12., 20.), f_gap=(28., 36.), f_len=(75., 100.)),
                reference='refs_b_idc_ref_30p.json'),
}
SETTINGS = dict(rounds=3, per_round=3, repeats=5, temperature=0.6, max_tokens=1500,
                timeout_s=120, percent_error=0.1, max_passes=30,
                etch_um=0.5, Lj_nH=12.31, Cj_fF=2.0,
                C_sigma_fF=(70., 105.), f01_GHz=(4.5, 5.5),
                objective='min D among feasible candidates including shared start',
                budget='9 attempted slots; duplicates and failures consume slots',
                frequency='q3d_shape_sweep.freq_from_csigma: exact physical constants',
                lifetime='not evaluated during search; terminal Purcell validation separate')


def identity(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def structure(task, parameters):
    if task == 'pad':
        return simple_dual_pad(**parameters)
    if task == 'idc':
        return geo.q_ellipse_idc(**parameters, n_f=4, jg=40.)
    raise KeyError(task)


def feasible(metrics):
    return metrics is not None and all(
        low <= metrics[key] <= high for key, (low, high) in
        ((key, SETTINGS[key]) for key in ('C_sigma_fF', 'f01_GHz')))

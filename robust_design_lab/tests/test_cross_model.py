"""Counterexamples for response failure, fixed slots, recovery and matrix gates."""
import json

import pytest

from robust_design_lab.experiments.cross_model.config import TASKS
from robust_design_lab.experiments.cross_model.protocol import decode_slots, run_search
from robust_design_lab.experiments.cross_model.oracle import metrics_from_scenarios


def metric(d=.004):
    return dict(D=d, C_sigma_fF=90., f01_GHz=4.6, alpha_MHz=-215.)


@pytest.mark.parametrize('raw', ['not json', '{}', '[]', '[{"changes":{"bogus":5}}]',
                                 '[{"changes":{"w":NaN}}]', '[{"changes":{"w":Infinity}}]'])
def test_all_rejected_does_not_become_success(tmp_path, raw):
    def solve(*args):
        raise AssertionError('Rejected actions must not reach solver')
    result = run_search('pad', metric(), lambda *a: {'content': raw}, solve, tmp_path, 'run')
    assert result['attempts'] == result['n_invalid'] == 9
    assert result['valid_new'] == 0 and result['complete'] is False


def test_excess_and_invalid_actions_do_not_shift_round_budgets(tmp_path):
    calls = []
    def propose(rnd, *args):
        return {'content': json.dumps([
            {'changes': {'bogus': 3}},
            {'changes': {'w': 700 + rnd}},
            {'changes': {'w': 710 + rnd}},
            {'changes': {'w': 720 + rnd}}])}
    def solve(task, kw):
        calls.append(kw)
        return metric()
    result = run_search('pad', metric(), propose, solve, tmp_path, 'run')
    assert result['attempts'] == 9 and len(calls) == 6
    assert [r['attempts'] for r in result['rounds']] == [3, 3, 3]
    assert not any(kw['w'] in (720, 721, 722) for kw in calls)
    assert result['complete'] and result['n_invalid'] == 3


def test_repeated_start_consumes_attempts_but_is_not_new_evidence(tmp_path):
    raw = json.dumps([{'changes': {'w': 734}}] * 3)
    result = run_search('pad', metric(), lambda *a: {'content': raw},
                        lambda *a: metric(), tmp_path, 'run')
    assert result['attempts'] == result['n_duplicates'] == 9
    assert result['complete'] is False


def test_restore_does_not_call_model_or_solver_again(tmp_path):
    raw = json.dumps([{'changes': {'w': 700 + i}} for i in range(3)])
    result = run_search('pad', metric(), lambda *a: {'content': raw},
                        lambda *a: metric(.003), tmp_path, 'run')
    def forbidden(*args):
        raise AssertionError('Unexpected repeated side effect')
    restored = run_search('pad', metric(), forbidden, forbidden, tmp_path, 'run')
    assert result == restored
    with pytest.raises(ValueError, match='identity changed'):
        run_search('pad', metric(), forbidden, forbidden, tmp_path, 'different')


def test_invalid_low_d_cannot_replace_feasible_start(tmp_path):
    raw = json.dumps([{'changes': {'w': 700 + i}} for i in range(3)])
    result = run_search('pad', metric(), lambda *a: {'content': raw},
                        lambda *a: dict(metric(1e-8), C_sigma_fF=40.), tmp_path, 'run')
    assert result['final']['source'] == 'shared_start'
    assert result['final']['metrics']['D'] == .004


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), 900, '700um'])
def test_invalid_numeric_values_are_not_silently_clipped(value):
    raw = json.dumps([{'changes': {'w': value}}])
    slots = decode_slots(raw, TASKS['pad']['start'], TASKS['pad']['box'])
    assert slots[0]['error'] in ('invalid_proposal', 'response_parse_failure')


def scenarios():
    source = {
        'nets': ['ground_main_plane', 'pad_left_Q1', 'pad_right_Q1'],
        'matrix_fF': [[200., -80., -80.], [-80., 129., -49.], [-80., -49., 129.]],
        'C_sigma_fF': 91., 'converged': True, 'percent_error': .1,
    }
    return {tag: dict(source) for tag in ('nom', 'ep', 'em')}


@pytest.mark.parametrize('status', [None, False, 'True'])
def test_any_unconverged_scene_is_rejected(status):
    data = scenarios()
    data['ep']['converged'] = status
    with pytest.raises(ValueError, match='Unconverged'):
        metrics_from_scenarios(data)


def test_matrix_frequency_and_signed_mhz_are_consistent():
    data = scenarios()
    data['ep']['matrix_fF'] = [[200., -80., -80.], [-80., 128., -48.], [-80., -48., 128.]]
    data['ep']['C_sigma_fF'] = 90.
    result = metrics_from_scenarios(data)
    assert result['D'] > 0
    assert result['signed_shift_MHz']['ep'] > 0
    assert result['signed_shift_MHz']['em'] == 0
    data['ep']['C_sigma_fF'] = 91.
    with pytest.raises(ValueError, match='matrix and summary differ'):
        metrics_from_scenarios(data)

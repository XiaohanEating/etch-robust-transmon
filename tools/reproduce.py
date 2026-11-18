"""Recompute archived results without Ansys, API access, or third-party packages."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text('utf-8-sig'))


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def capacitance(scene):
    nets, matrix = scene['nets'], scene['matrix_fF']
    g = next(i for i, net in enumerate(nets) if 'ground' in net.lower())
    left, right = [i for i in range(len(nets)) if i != g]
    c1, c2 = -matrix[left][g], -matrix[right][g]
    return 2.0 - matrix[left][right] + c1 * c2 / (c1 + c2)


def frequency(c):
    ec = 1.602176634e-19 ** 2 / (2 * c * 1e-15) / 6.62607015e-34 / 1e9
    ej = 2.067833848e-15 ** 2 / (4 * math.pi ** 2 * 12.31e-9) / 6.62607015e-34 / 1e9
    return math.sqrt(8 * ej * ec) - ec


def structural_frequency(c):
    ec = 19.36 / c
    return math.sqrt(8 * 13.28 * ec) - ec


def main():
    data = read('data/results.json')
    starts = read('robust_design_lab/results/cross_model_2026-11-08/run02/starts.json')
    groups, valid, scenes, attempts = defaultdict(list), defaultdict(int), 0, 0
    for row in data['cross']:
        run = read(row['source'])
        assert run['complete'] and run['attempts'] == len(run['records']) == 9
        attempts += run['attempts']
        valid[row['model']] += run['valid_new']
        metric = run['final']['metrics']
        assert 70 <= metric['C_sigma_fF'] <= 105 and 4.5 <= metric['f01_GHz'] <= 5.5
        fs = {}
        for tag in ['nom', 'ep', 'em']:
            scene = metric['scenarios'][tag]
            assert scene['converged'] is True and scene['percent_error'] == 0.1
            close(capacitance(scene), scene['C_sigma_fF'])
            fs[tag] = frequency(capacitance(scene))
            scenes += 1
        close(fs['nom'], metric['f01_GHz'])
        drift = max(abs(math.log(fs[k] / fs['nom'])) for k in ['ep', 'em'])
        close(drift, metric['D'])
        close(drift, row['metrics']['D'])
        best, parameters = starts[row['task']]['D'], None
        for rec in run['records']:
            m = rec.get('metrics')
            if rec.get('feasible') and m and m['D'] < best:
                best, parameters = m['D'], rec['candidate']
        close(best, drift)
        assert parameters == run['final']['parameters'] == row['parameters']
        reduction = 100 * (1 - drift / starts[row['task']]['D'])
        close(reduction, row['reduction'])
        groups[(row['model'], row['task'])].append(reduction)
    assert len(data['cross']) == 30 and attempts == 270 and scenes == 90
    assert all(len(v) == 5 for v in groups.values())
    rows = data['structure_rows']
    for row in rows.values():
        fs = {tag: structural_frequency(2 + terms['C12'] + terms['G']) for tag, terms in row['breakdown'].items()}
        close(fs['nom'], row['f0'])
        close(max(abs(math.log(fs[tag] / fs['nom'])) for tag in ['ep', 'em']), row['D'])
    held = defaultdict(list)
    for row in data['held_out']:
        close(1000 * (row['f_GHz'] - row['f0_GHz']), row['delta_f_MHz'])
        close(abs(math.log(row['f_GHz'] / row['f0_GHz'])), row['D'])
        held[row['object']].append(abs(row['delta_f_MHz']))
    history = defaultdict(list)
    for row in data['historical']:
        m = row['metrics']
        qualified = bool(m) and m.get('ok', True) and 70 <= m['C_sigma_fF'] <= 105 and 4.5 <= m['f01_GHz'] <= 5.5
        assert qualified == row['feasible']
        if row['task'] == 'A_dual_pad' and qualified:
            history[row['model']].append(m['D'])
    result = dict(cross_model_searches=30, attempts=attempts, recomputed_final_scenes=scenes,
        median_reduction_percent={f'{m}/{t}': statistics.median(v) for (m, t), v in groups.items()},
        valid_new=dict(valid), structural_reduction_percent={k: 100 * (1 - rows['champion_01']['D'] / rows[k]['D']) for k in ['idc_01', 'pad_01']},
        stricter_edited_vs_pad_percent=100 * (1 - rows['champion_005']['D'] / rows['pad_005']['D']),
        held_out_max_MHz={k: max(v) for k, v in held.items()},
        historical_qualified_counts={k: len(v) for k, v in history.items()},
        scope='Archived electromagnetic simulations and analytic predictions; no fabricated-device measurement claims.')
    destination = ROOT / 'verification/reproduction.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + chr(10), 'utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

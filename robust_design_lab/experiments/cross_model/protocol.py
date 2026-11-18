"""Fixed three-slot rounds, strict decoding, and restart without extra requests."""
import json
import math
from pathlib import Path
import time

from robust_design_lab.runtime.records import write_json
from .config import SETTINGS, TASKS, feasible, identity


def decode_slots(raw, current, box, count=3):
    """Keep original order; rejected, missing and malformed entries consume slots."""
    try:
        text = raw.strip()
        if text.startswith('```'):
            text = '\n'.join(x for x in text.splitlines() if not x.strip().startswith('```'))
        def reject_nonfinite(token):
            raise ValueError('Nonstandard JSON numeric constant: ' + token)
        props = json.loads(text, parse_constant=reject_nonfinite)
        if not isinstance(props, list):
            raise ValueError('Expected a JSON array')
    except (ValueError, TypeError, AttributeError) as exc:
        return [dict(error='response_parse_failure', detail=str(exc)[:160])] * count
    slots = []
    for index in range(count):
        if index >= len(props):
            slots.append(dict(error='missing_proposal'))
            continue
        proposal = props[index]
        try:
            if not isinstance(proposal, dict) or set(proposal) - {'changes', 'reason'}:
                raise ValueError('Expected changes and optional reason')
            changes = proposal['changes']
            if not isinstance(changes, dict) or not 1 <= len(changes) <= 2:
                raise ValueError('Change one or two parameters')
            candidate = dict(current)
            for key, value in changes.items():
                if key not in box or isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError('Unknown parameter or non-numeric value')
                if not math.isfinite(value) or not box[key][0] <= value <= box[key][1]:
                    raise ValueError('Nonfinite or out-of-range value')
                candidate[key] = round(float(value), 4)
            slots.append(dict(candidate=candidate, raw=proposal))
        except (ValueError, KeyError, TypeError) as exc:
            slots.append(dict(error='invalid_proposal', detail=str(exc), raw=proposal))
    return slots


def run_search(task, start_metrics, proposer, evaluator, output, run_identity):
    """proposer sees only this run; evaluator may reuse a deterministic oracle."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    descriptor = dict(run_identity=run_identity, task=task)
    descriptor_path = output / 'identity.json'
    if descriptor_path.exists():
        if json.loads(descriptor_path.read_text(encoding='utf-8')) != descriptor:
            raise ValueError('Run identity changed')
    else:
        write_json(descriptor_path, descriptor)
    current = dict(TASKS[task]['start'])
    incumbent = dict(parameters=current, metrics=start_metrics, feasible=feasible(start_metrics),
                     source='shared_start', identity=identity(current))
    if not incumbent['feasible']:
        raise ValueError('Shared start is not feasible')
    history, rounds = [], []
    seen = {identity(current)}
    valid_new = 0
    for rnd in range(SETTINGS['rounds']):
        round_dir = output / ('round_%02d' % rnd)
        round_dir.mkdir(exist_ok=True)
        response_file = round_dir / 'response.json'
        context = identity(dict(parameters=current, metrics=incumbent['metrics'], history=history))
        if response_file.exists():
            saved = json.loads(response_file.read_text(encoding='utf-8'))
            if saved['context'] != context:
                raise ValueError('Restored response context differs')
        else:
            t0 = time.perf_counter()
            try:
                response = proposer(rnd, dict(current), incumbent['metrics'], history, round_dir)
                saved = dict(context=context, response=response)
            except Exception as exc:
                saved = dict(context=context, response=None,
                             error='%s: %s' % (type(exc).__name__, str(exc)[:200]))
            saved['wall_s'] = time.perf_counter() - t0
            write_json(response_file, saved)
        response = saved['response'] or {}
        raw = response.get('content') or ''
        if response.get('finish_reason') == 'length':
            raw = ''
        slots = decode_slots(raw, current, TASKS[task]['box'], SETTINGS['per_round'])
        best = incumbent
        for index, slot in enumerate(slots):
            path = round_dir / ('candidate_%02d.json' % index)
            expected = identity(dict(context=context, slot=slot))
            if path.exists():
                record = json.loads(path.read_text(encoding='utf-8'))
                if record['input_identity'] != expected:
                    raise ValueError('Restored candidate differs')
            else:
                record = dict(round=rnd, index=index, input_identity=expected, **slot,
                              metrics=None, feasible=False, duplicate=False)
                if 'candidate' in slot:
                    key = identity(slot['candidate'])
                    record['duplicate'] = key in seen
                    t0 = time.perf_counter()
                    try:
                        result = evaluator(task, slot['candidate'])
                        record.update(metrics=result, feasible=feasible(result))
                    except Exception as exc:
                        record['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:250])
                    record['wall_s'] = time.perf_counter() - t0
                write_json(path, record)
            if record.get('candidate') is not None:
                key = identity(record['candidate'])
                if record['metrics'] is not None and key not in seen:
                    valid_new += 1
                seen.add(key)
                if record['feasible'] and record['metrics']['D'] < best['metrics']['D']:
                    best = dict(parameters=record['candidate'], metrics=record['metrics'],
                                feasible=True, source=str(path), identity=key)
            history.append(record)
            print(json.dumps(dict(task=task, run=output.name, round=rnd, slot=index,
                                  feasible=record['feasible'], error=record.get('error'),
                                  D=(record['metrics'] or {}).get('D')), ensure_ascii=False), flush=True)
        incumbent = best
        current = dict(best['parameters'])
        rounds.append(dict(round=rnd, attempts=len(slots), incumbent=incumbent))
        write_json(output / 'progress.json', dict(rounds=rounds, attempts=len(history)))
    result = dict(run_identity=run_identity, task=task, final=incumbent,
                  attempts=len(history), valid_new=valid_new,
                  n_duplicates=sum(x['duplicate'] for x in history),
                  n_invalid=sum(x.get('error') is not None and x.get('candidate') is None for x in history),
                  n_failed_solves=sum(x.get('candidate') is not None and x['metrics'] is None for x in history),
                  budget_exhausted=len(history) == 9,
                  complete=len(history) == 9 and valid_new > 0,
                  rounds=rounds, records=history)
    write_json(output / 'result.json', result)
    return result

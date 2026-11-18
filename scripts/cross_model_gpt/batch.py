"""Stage ten independent searches, then solve after the original twenty."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

from scripts.cross_model.provider import load_provider
from scripts.cross_model.extend import (AwaitingFieldSolve, defer_solver,
                                       extension_hashes, wait_for_original)
from robust_design_lab.experiments.purcell_qualification.batch import source_hashes
from robust_design_lab.experiments.cross_model.config import SETTINGS, TASKS, identity
from robust_design_lab.experiments.cross_model.oracle import Oracle
from robust_design_lab.experiments.cross_model.protocol import run_search
from robust_design_lab.experiments.cross_model.batch import summarize
from robust_design_lab.runtime.records import write_json
from .adapter import MODEL, PROFILE, payload_for, normalize, proposer

ROOT = Path(__file__).resolve().parents[2]


def adapter_hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


def assert_sources(manifest):
    if (source_hashes() != manifest['source_sha256']
            or extension_hashes() != manifest['shared_adapter_sha256']
            or adapter_hashes() != manifest['gpt_adapter_sha256']):
        raise ValueError('Archived sources changed; use a new batch')


def prepare(output, original, preflight, provider):
    parent = json.loads((original / 'manifest.json').read_text(encoding='utf-8'))
    core = source_hashes()
    if core != parent['source_sha256']:
        raise ValueError('Original core source changed')
    checks = {}
    for i in (0, 1):
        path = preflight / ('responses_image_%d.json' % i)
        row = json.loads(path.read_text(encoding='utf-8'))
        payload = row['request']
        parts = payload['input'][0]['content']
        text = next(x['text'] for x in parts if x['type'] == 'input_text')
        image_url = next(x['image_url'] for x in parts if x['type'] == 'input_image')
        if row['endpoint'] != provider.base_url or payload != payload_for(text, image_url):
            raise ValueError('Preflight configuration differs')
        response = normalize(row['raw_response'])
        if row['passed'] is not True or json.loads(response['content']) != row['expected']:
            raise ValueError('Paired image check failed')
        checks[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    common = extension_hashes()
    adapter = adapter_hashes()
    payload = dict(schema=1, parent_batch_identity=parent['batch_identity'],
                   source_sha256=core, shared_adapter_sha256=common, gpt_adapter_sha256=adapter,
                   tasks=TASKS, physical_and_search_settings=SETTINGS,
                   models=[MODEL], transport_profile=PROFILE, repeats=list(range(5)),
                   endpoint=provider.base_url, preflight_sha256=checks,
                   python=sys.executable, python_version=sys.version,
                   python_no_user_site=os.environ.get('PYTHONNOUSERSITE'),
                   comparison_scope='Equal candidate/field-solve budgets; model reasoning and token budgets differ.',
                   identity_scope='Relay requested/returned model identifier; underlying weights not independently verified.')
    manifest = dict(batch_identity=identity(payload), **payload)
    path = output / 'manifest.json'
    if path.exists():
        if identity(json.loads(path.read_text(encoding='utf-8'))) != identity(manifest):
            raise ValueError('GPT batch identity changed')
    else:
        with ZipFile(output / 'sources.zip', 'x', ZIP_DEFLATED) as archive:
            for name in {**core, **common, **adapter}:
                archive.write(ROOT / name, name)
        for name in checks:
            target = output / 'preflight' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((preflight / name).read_bytes())
        write_json(path, manifest)
    return manifest


def execute_searches(output, manifest, starts, provider, evaluator, stage):
    outcomes = []
    for repeat in range(5):
        for task in TASKS:
            assert_sources(manifest)
            run_id = identity(dict(batch=manifest['batch_identity'], model=MODEL, task=task, repeat=repeat))
            run_dir = output / 'runs' / MODEL / task / ('repeat_%02d' % repeat)
            result_path = run_dir / 'result.json'
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding='utf-8'))
                if result['run_identity'] != run_id:
                    raise ValueError('Existing result has a different identity')
                outcomes.append(dict(task=task, repeat=repeat, status='recorded', complete=result['complete']))
                continue
            start = time.perf_counter()
            try:
                result = run_search(task, starts[task], proposer(task, provider), evaluator, run_dir, run_id)
            except AwaitingFieldSolve:
                if not stage:
                    raise
                outcomes.append(dict(task=task, repeat=repeat, status='awaiting_field_solve', run=str(run_dir)))
                write_json(output / 'staged.json', outcomes)
                print(json.dumps(dict(staged_model=MODEL, task=task, repeat=repeat)), flush=True)
                continue
            timing = run_dir / 'timing_segments.json'
            segments = json.loads(timing.read_text(encoding='utf-8')) if timing.exists() else []
            segments.append(dict(wall_s=time.perf_counter() - start, stage=stage,
                                 completed_at_utc=datetime.now(timezone.utc).isoformat()))
            write_json(timing, segments)
            outcomes.append(dict(task=task, repeat=repeat, status='recorded', complete=result['complete']))
            summarize(output)
            print(json.dumps(dict(completed_model=MODEL, task=task, repeat=repeat,
                                  complete=result['complete'], D=result['final']['metrics']['D'])), flush=True)
        if stage:
            write_json(output / 'staged.json', outcomes)
    if stage:
        write_json(output / 'staged.json', outcomes)
    return outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--credentials', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--stage-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    original = args.original.resolve()
    provider = load_provider(args.credentials)
    lock = output / 'running.lock'
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, 'w') as f:
        f.write(str(os.getpid()))
    try:
        manifest = prepare(output, original, args.preflight, provider)
        oracle = Oracle(output / 'oracle', manifest['batch_identity'])
        starts = {task: oracle.import_start(task, ROOT / 'robust_design_lab/results/refs_b_2026-11-06')
                  for task in TASKS}
        write_json(output / 'starts.json', starts)
        if args.prepare_only:
            print('Sources, paired image checks and shared starts verified.', flush=True)
            return
        write_json(output / 'queue_status.json', dict(status='staging', pid=os.getpid()))
        outcomes = execute_searches(output, manifest, starts, provider, defer_solver, stage=True)
        if args.stage_only:
            write_json(output / 'queue_status.json', dict(status='staged', searches=len(outcomes), pid=os.getpid()))
            return
        wait_for_original(original, output)
        oracle.import_measurements(original / 'oracle')
        write_json(output / 'queue_status.json', dict(status='solving', pid=os.getpid()))
        execute_searches(output, manifest, starts, provider, oracle, stage=False)
        write_json(output / 'queue_status.json', dict(status='finished', pid=os.getpid()))
    except Exception as exc:
        write_json(output / 'orchestration_error.json', dict(type=type(exc).__name__, message=str(exc)[:500]))
        raise
    finally:
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()


if __name__ == '__main__':
    main()

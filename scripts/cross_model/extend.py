"""Extend a frozen experiment without modifying the active production sources."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

from robust_design_lab.runtime.records import write_json
from robust_design_lab.experiments.purcell_qualification.batch import source_hashes
from robust_design_lab.experiments.cross_model.config import SETTINGS, TASKS, identity
from robust_design_lab.experiments.cross_model.batch import render, prompt, summarize
from robust_design_lab.experiments.cross_model.oracle import Oracle
from robust_design_lab.experiments.cross_model.protocol import run_search
from .provider import load_provider, payload_for, chat

ROOT = Path(__file__).resolve().parents[2]
MODELS = ('glm-5.3', 'gpt-4.1-2025-04-14', 'claude-sonnet-4-5-20250929', 'gemini-2.5-flash')


def extension_hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


def prepare(output, original, preflight, provider, models):
    old = json.loads((original / 'manifest.json').read_text(encoding='utf-8'))
    hashes = source_hashes()
    if hashes != old['source_sha256']:
        raise ValueError('Frozen core differs from the running original batch')
    for model in models:
        for index in (0, 1):
            row = json.loads((preflight / model / ('image_%d.json' % index)).read_text(encoding='utf-8'))
            expected = payload_for(model, row['payload']['messages'])
            if not row['passed'] or row['endpoint'] != provider.base_url or row['payload'] != expected:
                raise ValueError('Preflight sampling/endpoint/check differs for ' + model)
    adapter = extension_hashes()
    preflight_hashes = {p.relative_to(preflight).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for model in models for p in sorted((preflight / model).glob('image_*.json'))}
    payload = dict(schema=1, parent_batch_identity=old['batch_identity'], source_sha256=hashes,
                   extension_sha256=adapter, tasks=TASKS, settings=SETTINGS, models=models,
                   endpoint=provider.base_url, provider='user_supplied_compatible_service',
                   preflight_sha256=preflight_hashes, python=sys.executable, python_version=sys.version,
                   transport_profiles={model: payload_for(model, []) for model in models},
                   model_identity_scope='requested/returned identifiers from relay; weights unverified')
    manifest = dict(batch_identity=identity(payload), **payload)
    path = output / 'manifest.json'
    if path.exists():
        if identity(json.loads(path.read_text(encoding='utf-8'))) != identity(manifest):
            raise ValueError('Extension manifest changed; use a new directory')
    else:
        with ZipFile(output / 'sources.zip', 'x', ZIP_DEFLATED) as archive:
            for name in {**hashes, **adapter}:
                archive.write(ROOT / name, name)
        for name in preflight_hashes:
            dest = output / 'preflight' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((preflight / name).read_bytes())
        write_json(path, manifest)
    return manifest


def proposer(model, task, provider):
    def call(rnd, current, metrics, history, output):
        path = output / 'geometry.png'
        render(task, current, path)
        text = prompt(task, current, metrics, history)
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': text},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
             base64.b64encode(path.read_bytes()).decode('ascii')}}]}]
        payload = payload_for(model, messages, SETTINGS['temperature'], SETTINGS['max_tokens'])
        write_json(output / 'request.json', dict(endpoint=provider.base_url, model=model, payload=payload,
                   parameters=current, image=str(path), image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   requested_at_utc=datetime.now(timezone.utc).isoformat(), service_seed=None))
        response = chat(provider, payload, SETTINGS['timeout_s'])
        response['received_at_utc'] = datetime.now(timezone.utc).isoformat()
        return response
    return call


class AwaitingFieldSolve(BaseException):
    """Stop staging after persisting the first response, before any physical solve."""


def defer_solver(*args):
    raise AwaitingFieldSolve()


def is_pid_alive(pid):
    if sys.platform == 'win32':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            if ctypes.get_last_error() == 5:
                raise RuntimeError('Cannot determine predecessor process status')
            return False
        try:
            status = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(status)):
                raise RuntimeError('Cannot read predecessor process status')
            return status.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_for_original(original, output):
    lock = original / 'running.lock'
    pid = int(lock.read_text()) if lock.exists() else None
    write_json(output / 'queue_status.json', dict(status='waiting_for_existing_solver', predecessor=str(original),
                                                 predecessor_pid=pid, pid=os.getpid()))
    while pid is not None and is_pid_alive(pid):
        time.sleep(5)
    if lock.exists():
        raise RuntimeError('Predecessor left its running lock; inspect before sharing Ansys')
    expected = {(model, task, repeat) for model in ('kimi-k2.5', 'qwen3-vl-plus')
                for task in TASKS for repeat in range(5)}
    actual = set()
    for path in original.glob('runs/*/*/repeat_*/result.json'):
        row = json.loads(path.read_text(encoding='utf-8'))
        if row['attempts'] != 9:
            raise RuntimeError('Predecessor did not exhaust its registered slots')
        actual.add((path.parents[2].name, row['task'], int(path.parent.name.split('_')[-1])))
    if actual != expected:
        raise RuntimeError('Predecessor stopped before all 20 registered searches were recorded')
    write_json(output / 'queue_status.json', dict(status='ready_to_solve', predecessor=str(original), pid=os.getpid()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--credentials', type=Path, required=True)
    ap.add_argument('--original', type=Path, required=True)
    ap.add_argument('--preflight', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--models', nargs='+', default=list(MODELS))
    ap.add_argument('--stage', action='store_true')
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--wait-for-original', action='store_true')
    args = ap.parse_args()
    output = args.output.resolve()
    original = args.original.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provider = load_provider(args.credentials)
    lock = output / 'running.lock'
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, 'w') as stream:
        stream.write(str(os.getpid()))
    try:
        manifest = prepare(output, original, args.preflight, provider, tuple(args.models))
        oracle = Oracle(output / 'oracle', manifest['batch_identity'])
        starts = {task: oracle.import_start(task, ROOT / 'robust_design_lab/results/refs_b_2026-11-06')
                  for task in TASKS}
        write_json(output / 'starts.json', starts)
        if args.prepare_only:
            print('Extension sources and shared starts verified; active core unchanged.', flush=True)
            return
        if not args.stage:
            if args.wait_for_original:
                wait_for_original(original, output)
            elif (original / 'running.lock').exists():
                raise RuntimeError('Existing solver batch is active; use --wait-for-original')
            oracle.import_measurements(original / 'oracle')
        staged = []
        for repeat in range(5):
            for task in TASKS:
                for model in args.models:
                    if source_hashes() != manifest['source_sha256'] or extension_hashes() != manifest['extension_sha256']:
                        raise ValueError('Frozen source changed')
                    run_dir = output / 'runs' / model / task / ('repeat_%02d' % repeat)
                    run_id = identity(dict(batch=manifest['batch_identity'], model=model, task=task, repeat=repeat))
                    if (run_dir / 'result.json').exists():
                        saved = json.loads((run_dir / 'result.json').read_text(encoding='utf-8'))
                        if saved['run_identity'] != run_id:
                            raise ValueError('Stored result identity differs')
                        continue
                    start_time = time.perf_counter()
                    try:
                        result = run_search(task, starts[task], proposer(model, task, provider),
                                            defer_solver if args.stage else oracle, run_dir, run_id)
                    except AwaitingFieldSolve:
                        staged.append(dict(model=model, task=task, repeat=repeat, response=str(run_dir / 'round_00/response.json')))
                        write_json(output / 'staged.json', staged)
                        print(json.dumps(dict(staged_model=model, task=task, repeat=repeat), ensure_ascii=False), flush=True)
                        continue
                    write_json(run_dir / 'timing.json', dict(wall_s=time.perf_counter() - start_time,
                               first_response_precomputed=(run_dir / 'round_00/response.json').exists()))
                    summarize(output)
                    print(json.dumps(dict(completed_model=model, task=task, repeat=repeat,
                                          D=result['final']['metrics']['D'], complete=result['complete'])), flush=True)
        write_json(output / 'queue_status.json', dict(status='first_responses_staged' if args.stage else 'finished',
                                                     staged_count=len(staged), pid=os.getpid()))
    except Exception as exc:
        write_json(output / 'orchestration_error.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()

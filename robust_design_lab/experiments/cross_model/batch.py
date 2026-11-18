"""Run independently sampled searches, storing every response before evaluation."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

from robust_design_lab.runtime.records import write_json
from robust_design_lab.vlm.aliyun import chat, load_credentials
from robust_design_lab.experiments.purcell_qualification.batch import source_hashes
from .config import SETTINGS, TASKS, identity, structure
from .oracle import Oracle
from .protocol import run_search

ROOT = Path(__file__).resolve().parents[2]


def render(task, parameters, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from robust_design_lab.geometry.novel_structures import nominal_pocket
    st = structure(task, parameters)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.fill(*nominal_pocket(st).exterior.xy, color='#e5e5e5')
    for key in ('pad_left', 'pad_right'):
        poly = st[key]
        ax.fill(*poly.exterior.xy, color='#2186c4', edgecolor='#164c75', linewidth=.6)
        for hole in poly.interiors:
            ax.fill(*hole.xy, color='#e5e5e5')
    ax.plot(*st['junction'].xy, color='#cf3434', linewidth=2)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def prompt(task, current, metrics, history):
    feedback = [dict(parameters=r.get('candidate'), feasible=r['feasible'], error=r.get('error'),
                     metrics={k: (r['metrics'] or {}).get(k) for k in ('D', 'C_sigma_fF', 'f01_GHz')})
                for r in history]
    return ('你是超导Transmon电容设计研究者。根据结构图、当前参数和自己的仿真历史，'
            '提出三个不同的候选，降低均匀刻蚀±0.5微米导致的最大绝对自然对数频漂D。'
            '每个候选相对于当前参数修改1至2项，全部数值单位为微米。'
            '图中蓝色为两电容岛，灰色为地口袋，红色为结。地口袋随金属同步刻蚀。'
            '固定Lj=12.31nH、Cj=2fF；名义CΣ必须在[70,105]fF且f01在[4.5,5.5]GHz。'
            '非法、重复、缺失提案或求解失败都占用一个候选名额，不另补。'
            '依据f≈sqrt(8EJEC)/h−EC/h，EC=e²/(2CΣ)；改变几何会同时改变名义C与刻蚀响应。'
            '请结合图像判断具体修改位置和尺寸，避免将单纯降低电容作为目标。'
            '\n任务：' + task + '\n参数范围：' + json.dumps(TASKS[task]['box']) +
            '\n当前参数：' + json.dumps(current) + '\n当前结果：' +
            json.dumps({k: metrics[k] for k in ('D', 'C_sigma_fF', 'f01_GHz', 'alpha_MHz')}) +
            '\n本次搜索历史：' + json.dumps(feedback, ensure_ascii=False) +
            '\n只返回长度为3的JSON数组，每项格式：'
            '{"changes":{"参数名":数值},"reason":"修改依据"}。三个候选各自作用于同一当前结构。')


def proposer_for(model, task, credentials):
    def propose(rnd, current, metrics, history, output):
        image = output / 'geometry.png'
        render(task, current, image)
        text = prompt(task, current, metrics, history)
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': text},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
             base64.b64encode(image.read_bytes()).decode('ascii')}}]}]
        write_json(output / 'request.json', dict(requested_model=model, round=rnd,
                   parameters=current, prompt=text, image=str(image),
                   image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                   temperature=SETTINGS['temperature'], max_tokens=SETTINGS['max_tokens'],
                   enable_thinking=False, stream=False, service_seed=None,
                   messages=messages))
        return chat(credentials, model, messages, max_tokens=SETTINGS['max_tokens'],
                    timeout=SETTINGS['timeout_s'], temperature=SETTINGS['temperature'])
    return propose


def prepare(output):
    hashes = source_hashes()
    payload = dict(schema=1, tasks=TASKS, settings=SETTINGS, source_sha256=hashes,
                   python=sys.executable, python_version=sys.version,
                   python_no_user_site=os.environ.get('PYTHONNOUSERSITE'))
    manifest = dict(batch_identity=identity(payload), **payload)
    path = output / 'manifest.json'
    if path.exists():
        if identity(json.loads(path.read_text(encoding='utf-8'))) != identity(manifest):
            raise ValueError('Sources/configuration changed; use a new output directory')
    else:
        with ZipFile(output / 'sources.zip', 'x', ZIP_DEFLATED) as archive:
            for name in hashes:
                archive.write(ROOT.parent / name, name)
        write_json(path, manifest)
    return manifest


def summarize(output):
    rows = []
    for path in sorted(output.glob('runs/*/*/repeat_*/result.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        rows.append(dict(model=path.parents[2].name, task=row['task'], repeat=path.parent.name,
                         D=row['final']['metrics']['D'], complete=row['complete'],
                         valid_new=row['valid_new'], n_invalid=row['n_invalid'],
                         n_failed_solves=row['n_failed_solves'], result=str(path)))
    write_json(output / 'summary.json', rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=['qwen3-vl-plus', 'kimi-k2.5'])
    parser.add_argument('--tasks', nargs='+', choices=list(TASKS), default=list(TASKS))
    parser.add_argument('--repeats', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--reuse-oracle', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'running.lock'
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, 'w') as stream:
        stream.write(str(os.getpid()))
    try:
        manifest = prepare(output)
        oracle = Oracle(output / 'oracle', manifest['batch_identity'])
        for model in args.models:
            for index in (0, 1):
                path = args.preflight / model.replace('/', '_') / ('image_%d.json' % index)
                if not path.exists() or json.loads(path.read_text(encoding='utf-8')).get('passed') is not True:
                    raise ValueError('Image preflight missing/failed for ' + model)
        starts = {task: oracle.import_start(task, ROOT / 'results/refs_b_2026-11-06') for task in args.tasks}
        write_json(output / 'starts.json', starts)
        if args.reuse_oracle:
            previous = json.loads((args.reuse_oracle.parent / 'manifest.json').read_text(encoding='utf-8'))
            for name, digest in manifest['source_sha256'].items():
                physical_source = (name.startswith(('src/qiskit_metal/', 'robust_design_lab/geometry/'))
                                   or name in ('robust_design_lab/experiments/q3d_shape_sweep.py',
                                               'robust_design_lab/experiments/verify_candidates.py',
                                               'robust_design_lab/experiments/cross_model/config.py'))
                if physical_source and previous['source_sha256'].get(name) != digest:
                    raise ValueError('Physical source changed; cannot import field results: ' + name)
            oracle.import_measurements(args.reuse_oracle)
        if args.prepare_only:
            print('Manifest, source archive and six starting geometry/matrix checks passed.', flush=True)
            return
        credentials = load_credentials()
        # Interleave models within each repeat, with no cross-model feedback.
        for repeat in args.repeats:
            if repeat not in range(5):
                raise ValueError('Repeat index must be in [0,5)')
            for task in args.tasks:
                for model in args.models:
                    if source_hashes() != manifest['source_sha256']:
                        raise ValueError('Sources changed while batch was running')
                    run_dir = output / 'runs' / model.replace('/', '_') / task / ('repeat_%02d' % repeat)
                    run_id = identity(dict(batch=manifest['batch_identity'], model=model, task=task, repeat=repeat))
                    if (run_dir / 'result.json').exists():
                        saved = json.loads((run_dir / 'result.json').read_text(encoding='utf-8'))
                        if saved['run_identity'] != run_id:
                            raise ValueError('Completed run identity differs')
                        continue
                    begin = time.perf_counter()
                    result = run_search(task, starts[task], proposer_for(model, task, credentials),
                                        oracle, run_dir, run_id)
                    write_json(run_dir / 'timing.json', dict(wall_s=time.perf_counter() - begin))
                    summarize(output)
                    print(json.dumps(dict(completed_model=model, task=task, repeat=repeat,
                                          complete=result['complete'], D=result['final']['metrics']['D'])), flush=True)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()

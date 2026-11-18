# -*- coding: utf-8 -*-
"""全量强基线对比调度器：5 数值基线 + VLM 臂 × 2 族 × 5 种子。

顺序：先完成 seed 0 的全部（族 × 臂），尽早拿到第一份完整对比；
再滚动 seed 1-4。每步独立子进程，runs.jsonl 去重可断点续跑；
单步失败记录后继续，不中断整个队列。

用法：
  python full_suite_schedule.py [--seeds 0 1 2 3 4]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "results" / "strong_baselines_2026-10-27" / "runs.jsonl"

PY = sys.executable  # Public artifact: use the active interpreter.
FAMILIES = ["A_dual_pad", "D_interdigit"]
BASELINES = ["coordinate_descent", "cma_es", "nelder_mead",
             "sobol_best", "bayesian_opt"]


def done_keys():
    if not RUNS.exists():
        return set()
    keys = set()
    with RUNS.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                keys.add((r["arm"], r["family"], r.get("seed")))
            except Exception:
                continue
    return keys


def run_step(cmd):
    env = {"PYTHONNOUSERSITE": "1", "PYTHONUTF8": "1"}
    import os
    full_env = dict(os.environ, **env)
    t0 = time.perf_counter()
    try:
        p = subprocess.run([PY, str(cmd[0])] + [str(c) for c in cmd[1:]],
                           cwd=str(ROOT.parent), env=full_env,
                           capture_output=True, text=True, timeout=14400)
    except subprocess.TimeoutExpired:
        # 单臂超 4h 视为失败：不写 runs.jsonl，下轮调度自动重试
        print(f"  ERR {time.perf_counter()-t0:6.0f}s TIMEOUT "
              f"{' '.join(str(c) for c in cmd)}")
        return False
    ok = p.returncode == 0
    tail = (p.stdout or "").strip().splitlines()[-1:] or [""]
    err = (p.stderr or "").strip().splitlines()[-3:]
    print(f"  {'OK ' if ok else 'ERR'} {time.perf_counter()-t0:6.0f}s "
          f"{' '.join(str(c) for c in cmd)} | {tail[0][:110]}")
    if not ok:
        for e in err:
            print(f"     ! {e[:160]}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    for seed in args.seeds:
        for family in FAMILIES:
            for baseline in BASELINES:
                if (baseline, family, seed) in done_keys():
                    print(f"跳过: {baseline} {family} seed={seed}")
                    continue
                run_step([HERE / "strong_baseline_run.py",
                          "--baseline", baseline,
                          "--family", family, "--seed", seed])
            # VLM 臂（每 seed 一次重复）
            if ("vlm", family, seed) in done_keys():
                print(f"跳过: vlm {family} seed={seed}")
                continue
            run_step([HERE / "vlm_arm_run.py",
                      "--family", family, "--repeat", seed])

    # 汇总
    print("\n==== 当前汇总 ====")
    by = {}
    for key in done_keys():
        by.setdefault((key[0], key[1]), []).append(key[2])
    for (arm, fam), seeds in sorted(by.items()):
        print(f"  {arm:18s} {fam:14s} 完成 seeds: {sorted(seeds)}")


if __name__ == "__main__":
    main()

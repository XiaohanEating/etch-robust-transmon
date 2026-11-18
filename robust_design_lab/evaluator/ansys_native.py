"""Probe installed AEDT through its built-in script interface."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from ..runtime.records import ROOT, provenance, write_json

AEDT_EXE = Path('D:/Program Files/AnsysEM/v221/Win64/ansysedt.exe')


def probe(output, timeout=90):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'native_probe_result.json'
    script_path = output / 'native_probe_script.py'
    script = '''import ScriptEnv
import json
import traceback
try:
    ScriptEnv.Initialize("Ansoft.ElectronicsDesktop.2022.1")
    report = {"connected": True, "version": str(oDesktop.GetVersion())}
except Exception:
    report = {"connected": False, "error": traceback.format_exc()}
with open(REPORT_PATH, "w") as stream:
    json.dump(report, stream, indent=2)
'''.replace('REPORT_PATH', repr(str(result_path)))
    script_path.write_text(script, encoding='utf-8')
    command = [str(AEDT_EXE), '-ng', '-RunScriptAndExit', str(script_path)]
    start = time.perf_counter()
    report = dict(command=command, provenance=provenance(), connected=False)
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                timeout=timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        report.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        if result_path.exists():
            report.update(json.loads(result_path.read_text(encoding='utf-8')))
        else:
            report['error'] = 'AEDT did not produce a native script result'
    except (OSError, subprocess.TimeoutExpired) as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    report['elapsed_s'] = time.perf_counter() - start
    write_json(output / 'native_probe.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'calibration_v2' / 'ansys')
    args = parser.parse_args()
    print(json.dumps(probe(args.output), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

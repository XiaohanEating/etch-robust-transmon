import argparse
import json
from pathlib import Path

from .batch import OBJECTS, SCENARIOS, build_manifest, run_batch


def main():
    parser = argparse.ArgumentParser(description="Common-readout Purcell qualification pilot")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--objects", nargs="+", choices=OBJECTS, default=OBJECTS)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=tuple(SCENARIOS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        manifest = build_manifest()
        print(json.dumps(dict(batch_sha256=manifest["batch_sha256"],
                              objects=list(manifest["objects"]), scenarios=list(SCENARIOS),
                              source_files=len(manifest["source_sha256"])), indent=2))
    else:
        result = run_batch(args.run_dir, args.objects, args.scenarios)
        print(json.dumps({k: {x: v[x] for x in ("complete", "D_f", "scenario_status")}
                          for k, v in result["objects"].items()}, indent=2))


if __name__ == "__main__":
    main()

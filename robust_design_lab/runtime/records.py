"""Atomic result records and explicit executable/configuration provenance."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from ..configs import frozen

ROOT = Path(__file__).resolve().parents[1]


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, 'item'):
        return clean_json(value.item())
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(clean_json(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def provenance():
    paths = sorted(p for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts)
    sources = {str(p.relative_to(ROOT)).replace(os.sep, '/'): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in paths}
    return dict(python=sys.executable, python_version=sys.version,
                python_no_user_site=os.environ.get('PYTHONNOUSERSITE'),
                frozen_config=frozen.frozen_hash(), source_sha256=sources,
                command=[sys.executable, *sys.argv], omp_threads=os.environ.get('OMP_NUM_THREADS'),
                mkl_threads=os.environ.get('MKL_NUM_THREADS'))

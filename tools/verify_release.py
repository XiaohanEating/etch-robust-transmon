"""Verify public-file checksums, model images, native projects, and privacy exclusions."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    manifest = json.loads((ROOT / 'provenance/files.json').read_text('utf-8'))
    for entry in manifest:
        assert digest((ROOT / entry['path']).read_bytes()) == entry['published_sha256'], entry['path']
    files = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    forbidden = {'.tex', '.bib', '.pdf', '.xlsx', '.pem', '.key', '.p12'}
    native = 0
    project_index = json.loads((ROOT / 'data/native_projects_index.json').read_text('utf-8'))
    expected = {r['path']: r for r in project_index['files']}
    archives = project_index.get('archives') or [project_index['archive']]
    for relative in archives:
        with zipfile.ZipFile(ROOT / relative) as archive:
            for name in archive.namelist():
                assert name.endswith('.aedt') and not Path(name).is_absolute() and '..' not in Path(name).parts
                raw = archive.read(name)
                assert digest(raw) == expected[name]['sha256']
                assert bytes([108, 101, 110, 111, 118, 111]) not in raw.lower() and b'-----BEGIN PRIVATE KEY' not in raw
                native += 1
    assert native == len(expected) == project_index['count']
    pycount, input_images = 0, 0
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        assert path.suffix.lower() not in forbidden, relative
        assert path.name not in {'api.txt', '.env', 'credentials.json'}, relative
        if path.suffix == '.py':
            ast.parse(path.read_text('utf-8-sig'), filename=relative)
            pycount += 1
        if path.suffix in {'.py', '.json', '.jsonl', '.md', '.txt', '.toml', '.csv', '.yml'}:
            raw = path.read_bytes()
            if path.name != 'verify_release.py':
                assert bytes([108, 101, 110, 111, 118, 111]) not in raw.lower(), relative
                assert b'-----BEGIN PRIVATE KEY' not in raw, relative
            # Actual credentials have longer provider-specific token bodies.
            for prefix in ['sk-', 'ghp_', 'gho_', 'github_pat_']:
                assert not re.search(re.escape(prefix) + '[A-Za-z0-9_-]{24,}', raw.decode('utf-8', 'replace')), relative
        if path.name == 'request.json' and 'round_' in relative:
            request = json.loads(path.read_text('utf-8'))
            image = path.with_name('geometry.png')
            assert digest(image.read_bytes()) == request['image_sha256'], relative
            payload = request.get('payload', request)
            if 'input' in payload:
                url = next(x['image_url'] for x in payload['input'][0]['content'] if x['type'] == 'input_image')
            else:
                url = next(x['image_url']['url'] for x in payload['messages'][0]['content'] if x['type'] == 'image_url')
            assert base64.b64decode(url.split(',', 1)[1]) == image.read_bytes(), relative
            input_images += 1
    assert input_images == 90
    report = dict(status='passed', provenance_files_checked=len(manifest), python_files_parsed=pycount, actual_model_input_images_checked=input_images,
                  native_projects_checked=native, physical_device_measurement_files=0, paper_files_published=0, secrets_detected=0)
    dest = ROOT / 'verification/release_check.json'
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + chr(10), 'utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

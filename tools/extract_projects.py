"""Restore verified native projects without replacing different local files."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    index = json.loads((ROOT / 'data/native_projects_index.json').read_text('utf-8'))
    expected = {row['path']: row['sha256'] for row in index['files']}
    count = 0
    for relative in index.get('archives') or [index['archive']]:
        with zipfile.ZipFile(ROOT / relative) as archive:
            for name in archive.namelist():
                destination = (ROOT / name).resolve()
                assert destination.is_relative_to(ROOT) and destination.suffix == '.aedt'
                raw = archive.read(name)
                assert hashlib.sha256(raw).hexdigest() == expected[name]
                if destination.exists():
                    assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected[name], str(destination)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(raw)
                count += 1
    print(f'Native projects verified and restored: {count}')


if __name__ == '__main__':
    main()

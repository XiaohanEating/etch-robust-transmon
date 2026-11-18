"""Independent providers share identical paired image checks and sampling limits."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time

from robust_design_lab.runtime.records import write_json
from .provider import load_provider, request, chat, payload_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--credentials', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--models', nargs='+', required=True)
    args = ap.parse_args()
    provider = load_provider(args.credentials)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'models.json', request(provider, '/models', timeout=30))
    prior_path = args.output / 'summary.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8')) if prior_path.exists() else []
    summary = [x for x in prior if x['model'] not in args.models]
    for model in args.models:
        rows = []
        for index, (filename, expected) in enumerate([
                ('a.png', dict(blue_circles=2, red_square_side='left')),
                ('b.png', dict(blue_circles=4, red_square_side='right'))]):
            path = args.output / model / ('image_%d.json' % index)
            image = args.images / filename
            text = ('Inspect the image. Return only a JSON object with blue_circles '
                    '(integer count) and red_square_side (left or right of image center).')
            messages = [{'role': 'user', 'content': [
                {'type': 'text', 'text': text},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                 base64.b64encode(image.read_bytes()).decode('ascii')}}]}]
            payload = payload_for(model, messages)
            if path.exists():
                row = json.loads(path.read_text(encoding='utf-8'))
                if row['payload'] != payload or row['endpoint'] != provider.base_url:
                    raise ValueError('Preflight identity changed')
                rows.append(row)
                continue
            row = dict(model=model, endpoint=provider.base_url, payload=payload,
                       image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                       expected=expected, passed=False)
            start = time.perf_counter()
            try:
                response = chat(provider, payload)
                row['response'] = response
                raw = (response['content'] or '').strip()
                if raw.startswith('```'):
                    raw = '\n'.join(x for x in raw.splitlines() if not x.strip().startswith('```'))
                parsed = json.loads(raw)
                row.update(parsed=parsed, passed=parsed == expected and response['finish_reason'] != 'length')
            except Exception as exc:
                row['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:500])
            row['wall_s'] = time.perf_counter() - start
            write_json(path, row)
            rows.append(row)
            print(json.dumps(dict(model=model, image=index, passed=row['passed'],
                                  response=row.get('response'), error=row.get('error')), ensure_ascii=False), flush=True)
        summary.append(dict(model=model, passed=all(r['passed'] for r in rows), checks=len(rows)))
        write_json(prior_path, summary)


if __name__ == '__main__':
    main()

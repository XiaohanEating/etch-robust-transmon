"""Verify that the configured model endpoint actually distinguishes images."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from robust_design_lab.runtime.records import write_json
from robust_design_lab.vlm.aliyun import chat, load_credentials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=['kimi-k2.5', 'glm-5.3'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    credentials = load_credentials()
    request = urllib.request.Request(credentials.base_url + '/models',
                                    headers={'Authorization': 'Bearer ' + credentials.api_key})
    with urllib.request.urlopen(request, timeout=25) as response:
        listing = json.load(response)
    write_json(args.output / 'available_models.json', listing)
    from PIL import Image, ImageDraw
    images = []
    # Neither file names nor prompt disclose the answer; only pixels do.
    for tag, count, side in [('a', 2, 'left'), ('b', 4, 'right')]:
        path = args.output / (tag + '.png')
        image = Image.new('RGB', (400, 300), 'white')
        draw = ImageDraw.Draw(image)
        for i in range(count):
            x = 35 + 90 * i
            draw.ellipse((x, 35, x + 45, 80), fill=(30, 90, 220))
        x = 50 if side == 'left' else 300
        draw.rectangle((x, 180, x + 45, 225), fill=(230, 40, 40))
        image.save(path)
        images.append((path, {'blue_circles': count, 'red_square_side': side}))
    summary_path = args.output / 'summary.json'
    prior = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else []
    summary = [r for r in prior if r['model'] not in args.models]
    for model in args.models:
        rows = []
        for index, (path, expected) in enumerate(images):
            dest = args.output / model.replace('/', '_') / ('image_%s.json' % index)
            if dest.exists():
                row = json.loads(dest.read_text(encoding='utf-8'))
                rows.append(row)
                continue
            text = ('Inspect the image. Return only a JSON object with blue_circles '
                    '(integer count) and red_square_side (left or right of image center).')
            image_url = 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')
            messages = [{'role': 'user', 'content': [
                {'type': 'text', 'text': text},
                {'type': 'image_url', 'image_url': {'url': image_url}}]}]
            row = dict(requested_model=model, prompt=text, image=str(path),
                       image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       expected=expected, passed=False, purpose='image-interface-check-only')
            start = time.perf_counter()
            try:
                response = chat(credentials, model, messages, max_tokens=512, timeout=90)
                row['response'] = response
                raw = (response.get('content') or '').strip()
                if raw.startswith('```'):
                    raw = '\n'.join(x for x in raw.splitlines() if not x.strip().startswith('```'))
                parsed = json.loads(raw)
                row.update(parsed=parsed, passed=parsed == expected)
            except Exception as exc:
                row['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:160])
            row['wall_s'] = time.perf_counter() - start
            write_json(dest, row)
            rows.append(row)
            print(json.dumps({'model': model, 'image': index, 'passed': row['passed'],
                              'response': row.get('response'), 'error': row.get('error')},
                             ensure_ascii=False), flush=True)
        summary.append(dict(model=model, passed=all(x['passed'] for x in rows), checks=len(rows)))
        write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

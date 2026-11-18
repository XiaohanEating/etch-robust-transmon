"""Responses transport with explicit reasoning and visible failure records."""
import base64
from datetime import datetime, timezone
import hashlib

from scripts.cross_model.provider import request
from robust_design_lab.runtime.records import write_json
from robust_design_lab.experiments.cross_model.batch import render, prompt

MODEL = 'gpt-6-astra'
PROFILE = dict(model=MODEL, reasoning_effort='xhigh', max_output_tokens=16384,
               timeout_s=600, image_detail='high', temperature=None, store=False,
               endpoint_route='/responses')


def payload_for(text, image_url):
    return dict(model=MODEL, input=[dict(role='user', content=[
        dict(type='input_text', text=text),
        dict(type='input_image', image_url=image_url, detail=PROFILE['image_detail'])])],
        reasoning={'effort': PROFILE['reasoning_effort']},
        max_output_tokens=PROFILE['max_output_tokens'], store=False)


def normalize(data):
    if data.get('model') != MODEL:
        raise ValueError('Returned model identifier differs from the selected GPT model')
    status = data.get('status')
    content = ''.join(part.get('text', '')
                      for item in data.get('output', []) if item.get('type') == 'message'
                      for part in item.get('content', []) if part.get('type') == 'output_text')
    limited = status == 'incomplete' and (data.get('incomplete_details') or {}).get('reason') == 'max_output_tokens'
    if status != 'completed' and not limited:
        raise ValueError('Response did not complete: ' + str(status))
    return dict(content=content, model=data.get('model'), request_id=data.get('id'),
                finish_reason='length' if limited else 'stop', usage=data.get('usage', {}),
                status=status, reasoning=data.get('reasoning'),
                incomplete_details=data.get('incomplete_details'))


def proposer(task, provider):
    def propose(rnd, current, metrics, history, output):
        image = output / 'geometry.png'
        render(task, current, image)
        image_bytes = image.read_bytes()
        text = prompt(task, current, metrics, history)
        payload = payload_for(text, 'data:image/png;base64,' + base64.b64encode(image_bytes).decode('ascii'))
        write_json(output / 'request.json', dict(endpoint=provider.base_url, payload=payload,
                   model=MODEL, round=rnd, parameters=current,
                   image_sha256=hashlib.sha256(image_bytes).hexdigest(),
                   requested_at_utc=datetime.now(timezone.utc).isoformat(),
                   service_seed=None, transport_profile=PROFILE))
        data = request(provider, '/responses', payload, timeout=PROFILE['timeout_s'])
        # Keep service metadata before validating it; never silently substitute a model.
        write_json(output / 'raw_response.json', data)
        return normalize(data)
    return propose

"""User-selected compatible endpoint; secrets are read locally and never archived."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request


@dataclass(frozen=True)
class Provider:
    base_url: str
    api_key: str = field(repr=False)
    source: str = ''


def load_provider(path):
    lines = [line.strip() for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError('Expected endpoint and key on two nonempty lines')
    match = re.search(r'https://[^\s]+', lines[0])
    if not match:
        raise ValueError('Expected HTTPS endpoint')
    base = match.group().rstrip('/')
    parsed = urllib.parse.urlsplit(base)
    if parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
        raise ValueError('Endpoint must not embed credentials or query data')
    key = lines[1]
    if re.match(r'(?i)^(api[_ -]?key|key|token)\s*[:=：]', key):
        key = re.split(r'[:=：]', key, maxsplit=1)[1].strip()
    if len(key) < 16 or any(c.isspace() for c in key):
        raise ValueError('Invalid credential format')
    return Provider(base, key, str(Path(path).resolve()))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _redact(value, key):
    if isinstance(value, str):
        return value.replace(key, '[REDACTED]')
    if isinstance(value, dict):
        return {k: _redact(v, key) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, key) for v in value]
    return value


def request(provider, route, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode('utf-8')
    req = urllib.request.Request(provider.base_url + route, data=data,
                                 headers={'Authorization': 'Bearer ' + provider.api_key,
                                          'Content-Type': 'application/json'})
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        # Preserve only the server's error fields, never headers or a request echo.
        try:
            body = json.loads(exc.read())
            error = body.get('error', {})
            message = error.get('message', '') if isinstance(error, dict) else str(error)
        except Exception:
            message = ''
        message = _redact(str(message), provider.api_key)
        # A server error can include image/request content; omit lengthy bodies.
        if len(message) > 500 or 'base64' in message:
            message = 'Response body omitted'
        raise RuntimeError('HTTP %s: %s' % (exc.code, message)) from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError('Provider transport failure: ' + type(exc).__name__) from None
    return _redact(result, provider.api_key)


def payload_for(model, messages, temperature=.6, max_tokens=1500):
    payload = dict(model=model, messages=messages, temperature=temperature,
                   max_tokens=max_tokens, stream=False)
    # Explicitly request no thinking for the model that exposes this setting.
    if model.startswith('gemini-2.5-flash'):
        payload['thinking_budget'] = 0
    if model.startswith('glm-'):
        payload['thinking'] = {'type': 'disabled'}
    return payload


def chat(provider, payload, timeout=120):
    data = request(provider, '/chat/completions', payload, timeout)
    choice = data['choices'][0]
    content = choice['message'].get('content')
    if isinstance(content, list):
        content = ''.join(item.get('text', '') for item in content if item.get('type') == 'text')
    return dict(content=content, model=data.get('model'), request_id=data.get('id'),
                finish_reason=choice.get('finish_reason'), usage=data.get('usage', {}),
                service_fingerprint=data.get('system_fingerprint'),
                reasoning_content=choice['message'].get('reasoning_content'))

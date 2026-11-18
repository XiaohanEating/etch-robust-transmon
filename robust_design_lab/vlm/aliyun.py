"""Read user-provided Model Studio credentials locally; never serialize secrets."""
import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import posixpath
import re
from zipfile import BadZipFile, ZipFile
from xml.etree import ElementTree as ET
import urllib.error
import urllib.parse
import urllib.request


class CredentialError(ValueError):
    pass


@dataclass(frozen=True)
class Credentials:
    api_key: str = field(repr=False)
    base_url: str
    source: str


def _validated_credentials(key, base, source):
    if len(key) < 16 or any(c.isspace() for c in key) or '***' in key:
        raise CredentialError('API key is incomplete or masked')
    if not base:
        raise CredentialError('Credential file has no OpenAI-compatible endpoint')
    parsed = urllib.parse.urlsplit(base)
    if (parsed.scheme != 'https' or not parsed.hostname
            or not parsed.hostname.endswith('.aliyuncs.com')
            or parsed.username or parsed.password or parsed.query):
        raise CredentialError('Expected a verified HTTPS Alibaba Cloud endpoint')
    base = base.rstrip('/')
    if base.endswith('/chat/completions'):
        base = base[:-len('/chat/completions')]
    return Credentials(key, base, source)


def read_csv(path):
    path = Path(path)
    for encoding in ('utf-8-sig', 'utf-16', 'gb18030'):
        try:
            raw = path.read_text(encoding=encoding)
            break
        except UnicodeError:
            continue
    else:
        raise CredentialError('Credential CSV encoding is not supported')
    rows = list(csv.reader(raw.splitlines()))
    pairs = {r[0].strip().lower(): r[1].strip() for r in rows if len(r) >= 2}
    key = pairs.get('apikey', pairs.get('api_key', ''))
    base = pairs.get('openaicompatible', pairs.get('base_url', ''))
    return _validated_credentials(key, base, path.name)


def read_xlsx(path):
    """Read key/endpoint pairs from a local XLSX without external packages."""
    path = Path(path)
    ns = {
        'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
        'rel': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
        'pkg': 'http://schemas.openxmlformats.org/package/2006/relationships',
    }
    try:
        with ZipFile(path) as archive:
            shared = []
            if 'xl/sharedStrings.xml' in archive.namelist():
                root = ET.fromstring(archive.read('xl/sharedStrings.xml'))
                shared = [''.join(t.text or '' for t in item.findall('.//main:t', ns))
                          for item in root.findall('main:si', ns)]
            workbook = ET.fromstring(archive.read('xl/workbook.xml'))
            relationships = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
            targets = {item.attrib['Id']: item.attrib['Target']
                       for item in relationships.findall('pkg:Relationship', ns)}

            def column_index(reference):
                match = re.match(r'([A-Z]+)', reference.upper())
                if not match:
                    return 0
                index = 0
                for char in match.group(1):
                    index = index * 26 + ord(char) - 64
                return index

            def cell_value(cell):
                kind = cell.attrib.get('t')
                if cell.find('main:f', ns) is not None:
                    return ''
                if kind == 'inlineStr':
                    return ''.join(t.text or '' for t in cell.findall('.//main:t', ns)).strip()
                value = cell.find('main:v', ns)
                if value is None or value.text is None:
                    return ''
                if kind == 's':
                    return shared[int(value.text)].strip()
                return value.text.strip()

            keys = []
            endpoints = []
            for sheet in workbook.findall('main:sheets/main:sheet', ns):
                target = targets[sheet.attrib['{'+ns['rel']+'}id']]
                sheet_path = (target.lstrip('/') if target.startswith('/')
                              else posixpath.normpath(posixpath.join('xl', target)))
                sheet_root = ET.fromstring(archive.read(sheet_path))
                for row in sheet_root.findall('.//main:sheetData/main:row', ns):
                    cells = {column_index(cell.attrib.get('r', '')): cell_value(cell)
                             for cell in row.findall('main:c', ns)}
                    for column, label in cells.items():
                        normalized = ''.join(char.casefold() for char in label if char.isalnum())
                        if not normalized:
                            continue
                        field = None
                        if 'apikey' in normalized or normalized in ('api密钥', '访问密钥'):
                            field = keys
                        elif any(token in normalized for token in
                                 ('openaicompatible', 'baseurl', 'endpoint', '接口地址', '兼容地址')):
                            field = endpoints
                        if field is None:
                            continue
                        value = next((value for col, value in sorted(cells.items())
                                      if col > column and value), '')
                        if not value:
                            next_row = sheet_root.find(
                                f".//main:sheetData/main:row[@r='{int(row.attrib['r']) + 1}']", ns)
                            if next_row is not None:
                                value = next((cell_value(cell) for cell in next_row.findall('main:c', ns)
                                              if column_index(cell.attrib.get('r', '')) == column), '')
                        if value:
                            field.append(value.strip())
    except (BadZipFile, KeyError, ValueError, ET.ParseError, OSError) as error:
        raise CredentialError('Could not safely parse the local XLSX credentials') from None

    if len(keys) != 1 or len(endpoints) != 1:
        raise CredentialError('XLSX must contain exactly one API key and one compatible endpoint')
    return _validated_credentials(keys[0], endpoints[0], path.name)


def read_credentials(path):
    path = Path(path)
    if path.suffix.casefold() == '.xlsx':
        return read_xlsx(path)
    return read_csv(path)


def load_credentials(directory=None):
    explicit = (os.environ.get('ROBUST_ALIYUN_CREDENTIALS')
                or os.environ.get('ROBUST_ALIYUN_CREDENTIALS_XLSX')
                or os.environ.get('ROBUST_ALIYUN_CREDENTIALS_CSV'))
    if explicit:
        return read_credentials(explicit)
    directory = Path.cwd() if directory is None else Path(directory)
    candidates = []
    for path in (*directory.glob('*.csv'), *directory.glob('*.xlsx')):
        try:
            candidates.append(read_credentials(path))
        except CredentialError:
            continue
    if len(candidates) != 1:
        raise CredentialError('Expected exactly one usable API credential file; set ROBUST_ALIYUN_CREDENTIALS')
    return candidates[0]


def chat(credentials, model, messages, max_tokens=1200, timeout=90, *, temperature=0):
    payload = dict(model=model, messages=messages, max_tokens=max_tokens,
                   temperature=temperature, stream=False, enable_thinking=False)
    request = urllib.request.Request(credentials.base_url + '/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Authorization': 'Bearer ' + credentials.api_key,
                 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        # Server responses may echo input; do not print bodies or authentication.
        raise RuntimeError('Model endpoint returned HTTP ' + str(error.code)) from None
    return dict(content=data['choices'][0]['message']['content'],
                model=data.get('model', model), usage=data.get('usage', {}),
                request_id=data.get('id'), finish_reason=data['choices'][0].get('finish_reason'))

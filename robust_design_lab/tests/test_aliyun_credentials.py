"""Credential workbook parsing must never expose secret values."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pytest

from robust_design_lab.vlm.aliyun import CredentialError, read_xlsx


def _write_xlsx(path: Path, api_key: str):
    with ZipFile(path, 'w') as archive:
        archive.writestr(
            'xl/workbook.xml',
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="credentials" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            'xl/_rels/workbook.xml.rels',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="worksheet"/>'
            '</Relationships>',
        )
        archive.writestr(
            'xl/worksheets/sheet1.xml',
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData><row r="1">'
            '<c r="A1" t="inlineStr"><is><t>apiKey</t></is></c>'
            f'<c r="B1" t="inlineStr"><is><t>{api_key}</t></is></c>'
            '</row><row r="2">'
            '<c r="A2" t="inlineStr"><is><t>OpenAICompatible</t></is></c>'
            '<c r="B2" t="inlineStr"><is><t>https://example.cn-beijing.maas.aliyuncs.com/v1</t></is></c>'
            '</row></sheetData></worksheet>',
        )


def test_reads_xlsx_credentials_without_repr_leaking_key():
    secret = 'local-test-key-0123456789abcdef'
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'credentials.xlsx'
        _write_xlsx(path, secret)
        credentials = read_xlsx(path)

    assert credentials.api_key == secret
    assert credentials.base_url == 'https://example.cn-beijing.maas.aliyuncs.com/v1'
    assert secret not in repr(credentials)


def test_rejects_incomplete_key_in_xlsx():
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'credentials.xlsx'
        _write_xlsx(path, 'x')
        with pytest.raises(CredentialError, match='incomplete or masked'):
            read_xlsx(path)

"""Evidence-only XLSX/Markdown projection. Standard library; no model imports."""
from __future__ import annotations

from collections import defaultdict
import json
import math
import statistics
from xml.sax.saxutils import escape
import zipfile



def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + str(key)
        if isinstance(item, dict):
            result.update(flatten(item, name + "."))
        else:
            result[name] = json.dumps(item, ensure_ascii=False) if isinstance(item, list) else item
    return result


def text(value):
    if value is None:
        return "unavailable"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def column(index):
    name = ""
    while index:
        index, digit = divmod(index - 1, 26)
        name = chr(65 + digit) + name
    return name


def workbook(path, sheets, *, presentation=False):
    """Small OOXML writer: explicit text cells prevent formula injection. Raw retains full strings."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    declarations = ['<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
                    '<Default Extension="xml" ContentType="application/xml"/>',
                    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
    names, relations = [], []
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        if presentation:
            declarations.append('<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>')
            archive.writestr("xl/styles.xml", f'<styleSheet xmlns="{ns}"><fonts count="2"><font><sz val="11"/><name val="Microsoft JhengHei"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Microsoft JhengHei"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF24576B"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment wrapText="1"/></xf><xf numFmtId="4" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')
        for number, (name, records) in enumerate(sheets.items(), 1):
            fields = list(dict.fromkeys(k for row in records for k in row)) or ["status"]
            if len(records) + 1 > 1048576 or len(fields) > 16384:
                raise ValueError("Excel row/column limit exceeded; raw evidence retained")
            rows = []
            for ri, values in enumerate([fields] + [[row.get(k) for k in fields] for row in records], 1):
                cells = []
                for ci, value in enumerate(values, 1):
                    ref = column(ci) + str(ri)
                    style = ' s="1"' if presentation and ri == 1 else ' s="2"' if presentation and isinstance(value, float) else ''
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        cells.append(f'<c r="{ref}"{style}><v>{value}</v></c>')
                    else:
                        original = text(value)
                        cleaned = "".join(c for c in original if c in "\t\n\r" or ord(c) >= 32)
                        # Excel's limit is UTF-16 code units, not Python characters.
                        cleaned = cleaned.encode("utf-16-le")[:64000].decode("utf-16-le", errors="ignore")
                        if cleaned != original:
                            cleaned += " [完整內容見raw]"
                        cells.append(f'<c r="{ref}"{style} t="inlineStr"><is><t xml:space="preserve">{escape(cleaned)}</t></is></c>')
                rows.append(f'<row r="{ri}">' + "".join(cells) + '</row>')
            document = (f'<worksheet xmlns="{ns}"><sheetViews><sheetView workbookViewId="0">'
                        '<pane ySplit="1" topLeftCell="A2" state="frozen"/></sheetView></sheetViews>'
                        + ('<cols>' + ''.join(f'<col min="{i}" max="{i}" width="{60 if any(k in field for k in ("答案", "問題", "限制", "內容")) else 24}" customWidth="1"/>' for i, field in enumerate(fields, 1)) + '</cols>' if presentation else '')
                        + '<sheetData>' + "".join(rows) + '</sheetData>'
                        f'<autoFilter ref="A1:{column(len(fields))}{len(records)+1}"/></worksheet>')
            archive.writestr(f"xl/worksheets/sheet{number}.xml", document)
            sheet_name = escape(name, {'"': '&quot;'})
            names.append(f'<sheet name="{sheet_name}" sheetId="{number}" r:id="rId{number}"/>')
            relations.append(f'<Relationship Id="rId{number}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{number}.xml"/>')
            declarations.append(f'<Override PartName="/xl/worksheets/sheet{number}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        archive.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' + "".join(declarations) + '</Types>')
        archive.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr("xl/workbook.xml", f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + "".join(names) + '</sheets></workbook>')
        if presentation:
            relations.append('<Relationship Id="styles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>')
        archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(relations) + '</Relationships>')
    # Structural packaging verification only; no claim of Excel application validation.
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("XLSX archive CRC verification failed")

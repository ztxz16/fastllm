#!/usr/bin/env python3
"""使用Python标准库生成简单的多工作表XLSX报告。"""

import re
from pathlib import Path
from xml.sax.saxutils import escape
import zipfile


def _column_name(index):
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def _sheet_name(value, used_names):
    value = re.sub(r"[\\/*?:\[\]]", "_", value).strip("'") or "Sheet"
    value = value[:31]
    candidate = value
    suffix = 2
    while candidate in used_names:
        marker = f"-{suffix}"
        candidate = value[:31 - len(marker)] + marker
        suffix += 1
    used_names.add(candidate)
    return candidate


def _cell_xml(reference, value, style=0):
    if isinstance(value, float) and style == 0:
        style = 2
    style_attr = f' s="{style}"' if style else ""
    if value is None:
        return (f'<c r="{reference}"{style_attr} t="inlineStr">'
                '<is><t>n/a</t></is></c>')
    if isinstance(value, bool):
        return f'<c r="{reference}"{style_attr} t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)):
        return f'<c r="{reference}"{style_attr}><v>{value}</v></c>'
    text = escape(str(value))
    preserve = ' xml:space="preserve"' if str(value).strip() != str(value) else ""
    return (f'<c r="{reference}"{style_attr} t="inlineStr"><is>'
            f'<t{preserve}>{text}</t></is></c>')


def write_xlsx(path, sheets, image_sheets=None):
    """把表格和PNG趋势图写成无需第三方依赖的XLSX工作簿。"""
    used_names = set()
    normalized = [
        (_sheet_name(name, used_names), headers, rows)
        for name, headers, rows in sheets
    ]
    normalized_images = []
    for name, images in image_sheets or []:
        paths = [Path(image) for image in images]
        normalized_images.append((
            len(normalized) + 1,
            paths,
        ))
        normalized.append((
            _sheet_name(name, used_names), ["趋势图文件"],
            [[path.name] for path in paths],
        ))

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for index in range(1, len(normalized) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    if normalized_images:
        content_types.append('<Default Extension="png" ContentType="image/png"/>')
    for drawing_index in range(1, len(normalized_images) + 1):
        content_types.append(
            f'<Override PartName="/xl/drawings/drawing{drawing_index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>')
    content_types.append('</Types>')

    workbook_sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, (name, _, _) in enumerate(normalized, 1))
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{workbook_sheets}</sheets></workbook>')

    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for index in range(1, len(normalized) + 1):
        workbook_rels.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>')
    workbook_rels.extend((
        f'<Relationship Id="rId{len(normalized) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>',
        '</Relationships>',
    ))

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>')
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<numFmts count="1"><numFmt numFmtId="164" formatCode="0.000"/></numFmts>'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font></fonts>'
        '<fills count="3"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF4472C4"/>'
        '<bgColor indexed="64"/></patternFill></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
        '<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
        '</cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/>'
        '</cellStyles></styleSheet>')

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "".join(content_types))
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        archive.writestr("xl/styles.xml", styles)

        image_sheet_map = {
            sheet_index: (drawing_index, paths)
            for drawing_index, (sheet_index, paths) in
            enumerate(normalized_images, 1)
        }
        media_index = 1
        for sheet_index, (_, headers, rows) in enumerate(normalized, 1):
            all_rows = [headers, *rows]
            widths = []
            for column_index, header in enumerate(headers):
                values = [header] + [
                    row[column_index] if column_index < len(row) else ""
                    for row in rows
                ]
                widths.append(min(
                    80, max(10, max(len(str(value)) for value in values) + 2)))
            columns = "".join(
                f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
                for index, width in enumerate(widths, 1))
            row_xml = []
            for row_index, row in enumerate(all_rows, 1):
                cells = "".join(
                    _cell_xml(
                        f"{_column_name(column_index)}{row_index}", value,
                        1 if row_index == 1 else 0)
                    for column_index, value in enumerate(row, 1))
                row_xml.append(f'<row r="{row_index}">{cells}</row>')
            last_cell = f"{_column_name(len(headers))}{len(all_rows)}"
            drawing = image_sheet_map.get(sheet_index)
            relationship_namespace = (
                ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
                if drawing else "")
            drawing_reference = '<drawing r:id="rId1"/>' if drawing else ""
            worksheet = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
                f'{relationship_namespace}>'
                '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
                'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
                f'<cols>{columns}</cols><sheetData>{"".join(row_xml)}</sheetData>'
                f'<autoFilter ref="A1:{last_cell}"/>{drawing_reference}</worksheet>')
            archive.writestr(f"xl/worksheets/sheet{sheet_index}.xml", worksheet)

            if not drawing:
                continue
            drawing_index, image_paths = drawing
            worksheet_rels = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" '
                f'Target="../drawings/drawing{drawing_index}.xml"/></Relationships>')
            archive.writestr(
                f"xl/worksheets/_rels/sheet{sheet_index}.xml.rels",
                worksheet_rels)

            anchors = []
            drawing_relationships = []
            for local_index, image_path in enumerate(image_paths, 1):
                if not image_path.is_file():
                    raise FileNotFoundError(f"趋势图不存在：{image_path}")
                start_row = 2 + (local_index - 1) * 42
                anchors.append(
                    '<xdr:twoCellAnchor>'
                    f'<xdr:from><xdr:col>0</xdr:col><xdr:colOff>0</xdr:colOff>'
                    f'<xdr:row>{start_row}</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from>'
                    f'<xdr:to><xdr:col>16</xdr:col><xdr:colOff>0</xdr:colOff>'
                    f'<xdr:row>{start_row + 38}</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:to>'
                    '<xdr:pic><xdr:nvPicPr>'
                    f'<xdr:cNvPr id="{local_index}" name="{escape(image_path.name)}"/>'
                    '<xdr:cNvPicPr/></xdr:nvPicPr><xdr:blipFill>'
                    f'<a:blip r:embed="rId{local_index}"/><a:stretch><a:fillRect/></a:stretch>'
                    '</xdr:blipFill><xdr:spPr><a:prstGeom prst="rect"><a:avLst/>'
                    '</a:prstGeom></xdr:spPr></xdr:pic><xdr:clientData/>'
                    '</xdr:twoCellAnchor>')
                drawing_relationships.append(
                    f'<Relationship Id="rId{local_index}" '
                    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                    f'Target="../media/image{media_index}.png"/>')
                archive.writestr(
                    f"xl/media/image{media_index}.png", image_path.read_bytes())
                media_index += 1

            drawing_xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" '
                'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                f'{"".join(anchors)}</xdr:wsDr>')
            drawing_rels = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                f'{"".join(drawing_relationships)}</Relationships>')
            archive.writestr(f"xl/drawings/drawing{drawing_index}.xml", drawing_xml)
            archive.writestr(
                f"xl/drawings/_rels/drawing{drawing_index}.xml.rels",
                drawing_rels)

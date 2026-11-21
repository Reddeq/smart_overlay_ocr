"""Native PDF typography with natural font widths and paragraph leading.

Source sizes are starting bounds, not a reason to compress glyphs or baselines.
Tables and text frames may grow; editable structure and text take precedence.
"""
from functools import lru_cache
from io import BytesIO
import os
import math
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from PIL import ImageFont

NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}


def _tag(name):
    return "{" + NS["a"] + "}" + name


@lru_cache(maxsize=1)
def _windows_fonts():
    if os.name != "nt":
        return {}
    import winreg
    result = {}
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            key = winreg.OpenKey(hive, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts")
        except FileNotFoundError:
            continue
        with key:
            for index in range(winreg.QueryInfoKey(key)[1]):
                name, filename, _ = winreg.EnumValue(key, index)
                path = Path(filename)
                if not path.is_absolute():
                    path = Path(os.environ["WINDIR"]) / "Fonts" / path
                result[name.removesuffix(" (TrueType)").casefold()] = str(path)
    return result


@lru_cache(maxsize=256)
def _font(face, size, bold=False, italic=False):
    suffix = (" Bold" if bold else "") + (" Italic" if italic else "")
    registered = _windows_fonts().get((face + suffix).casefold())
    for candidate in (registered, face + suffix + ".ttf"):
        if candidate:
            try:
                return ImageFont.truetype(candidate, size), face
            except OSError:
                pass
    for substitute, filename in (("Calibri", "calibri.ttf"), ("Arial", "arial.ttf"),
                                 ("DejaVu Sans", "DejaVuSans.ttf")):
        try:
            registered = _windows_fonts().get((substitute + suffix).casefold(), filename)
            return ImageFont.truetype(registered, size), substitute
        except OSError:
            pass
    raise RuntimeError(f"Cannot measure table font {face!r}; install it or a supported fallback font")


def _parse_slide(data):
    # ElementTree otherwise renames namespaces and drops declarations referenced
    # only by prefix-valued attributes (e.g. mc:Ignorable).
    namespaces = dict(item for _, item in ET.iterparse(BytesIO(data), events=("start-ns",)))
    for prefix, uri in namespaces.items():
        ET.register_namespace(prefix, uri)
    root = ET.fromstring(data)
    used = {name[1:].split("}", 1)[0] for element in root.iter()
            for name in (element.tag, *element.attrib) if name.startswith("{")}
    for prefix, uri in namespaces.items():
        if uri not in used:
            root.set("xmlns" + (":" + prefix if prefix else ""), uri)
    return root


def _same_baseline_paragraphs(body, layout, paragraphs, width, pdf_scale):
    """Separate PDF spans with zero baseline advance are not new text rows."""
    if layout and len(layout["lines"]) == len(paragraphs):
        for index in range(len(paragraphs) - 2, -1, -1):
            if layout["lines"][index].get("line_spacing") is None:
                following = paragraphs[index + 1]
                advances = [run.get("advance") for run in layout["lines"][index + 1]["runs"]]
                tab_position = None
                if len(paragraphs) == 2 and advances and all(value is not None for value in advances):
                    tab_position = max(0, width - sum(advances) * pdf_scale)
                    properties = paragraphs[index].find("a:pPr", NS)
                    tabs = ET.SubElement(properties, _tag("tabLst"))
                    ET.SubElement(tabs, _tag("tab"), {"pos": str(round(tab_position * 12700)), "algn": "l"})
                for run_index, run in enumerate(following.findall("a:r", NS)):
                    if run_index == 0:
                        text = run.find("a:t", NS)
                        text.text = ("\t" if tab_position is not None else " ") + (text.text or "")
                    paragraphs[index].insert(len(paragraphs[index]) - 1, run)
                body.remove(following)
    return body.findall("a:p", NS)


def _paragraph(paragraph, substitutions):
    properties = paragraph.find("a:pPr", NS)
    if properties is None:
        properties = ET.Element(_tag("pPr"))
        paragraph.insert(0, properties)
    spacing = properties.find("a:lnSpc", NS)
    if spacing is not None:
        properties.remove(spacing)
    runs = paragraph.findall("a:r", NS)
    if not any((run.findtext("a:t", default="", namespaces=NS)).strip() for run in runs):
        # Empty edit placeholders have no source glyph size to preserve. Their
        # otherwise inherited Arial 12 can impose a minimum on tiny PDF rows.
        end = paragraph.find("a:endParaRPr", NS)
        if end is not None:
            end.set("sz", "100")
        for run in runs:
            run.find("a:rPr", NS).set("sz", "100")
    measured = []
    for run in runs:
        rp = run.find("a:rPr", NS)

        face = rp.find("a:latin", NS).get("typeface")
        _, actual = _font(face, int(rp.get("sz")), rp.get("b") == "1", rp.get("i") == "1")
        if actual != face:
            substitutions.add((face, actual))
        for font_tag in ("latin", "ea", "cs"):
            element = rp.find("a:" + font_tag, NS)
            if element is not None:
                element.set("typeface", actual)
        if int(rp.get("spc", "0")) < 0:
            rp.set("spc", "0")
        measured.append(rp)
    end = paragraph.find("a:endParaRPr", NS)
    if end is not None and measured:
        for font_tag in ("latin", "ea", "cs"):
            element = end.find("a:" + font_tag, NS)
            if element is not None:
                element.set("typeface", measured[-1].find("a:" + font_tag, NS).get("typeface"))


def _cell(xmlcell, cell, obj, scale, pdf_scale, substitutions):
    props = xmlcell.find("a:tcPr", NS)
    body = xmlcell.find("a:txBody", NS)
    if body is None:
        return
    body_properties = body.find("a:bodyPr", NS)
    ET.SubElement(body_properties, _tag("noAutofit"))
    available = sum(obj["col_widths"][cell["col"]:cell["col"] + cell["colspan"]]) * scale
    available -= (int(props.get("marL", 0)) + int(props.get("marR", 0))) / 12700
    layout = cell.get("native_layout")
    lines = (layout.get("paragraphs") if layout.get("paragraphs") is not None else layout["lines"]) if layout else []
    if layout and "\n".join("".join(run["text"] for run in line["runs"]) for line in lines) != cell["text"]:
        layout = None  # An edited cell must not recover stale PDF line layout.
    paragraphs = body.findall("a:p", NS)
    if layout and layout.get("paragraphs") is None:
        paragraphs = _same_baseline_paragraphs(body, layout, paragraphs, available, pdf_scale)
    for paragraph in paragraphs:
        _paragraph(paragraph, substitutions)


def _natural_text_height(body, width):
    """Conservative word wrapping using the actual exported font metrics."""
    height = 0.
    for paragraph in body.findall("a:p", NS):
        words = []
        word = []
        for run in paragraph.findall("a:r", NS):
            props = run.find("a:rPr", NS)
            font, _ = _font(props.find("a:latin", NS).get("typeface"),
                            int(props.get("sz", "1200")), props.get("b") == "1", props.get("i") == "1")
            leading = sum(font.getmetrics()) / 100
            for char in run.findtext("a:t", default="", namespaces=NS):
                metric = (font.getlength(char) / 100, leading)
                if char.isspace():
                    if word:
                        words.append((False, word))
                        word = []
                    words.append((True, [metric]))
                else:
                    word.append(metric)
        if word:
            words.append((False, word))
        x = line_height = gap = 0.
        for whitespace, word in words:
            if whitespace:
                gap += word[0][0]
                continue
            word_width = sum(item[0] for item in word)
            if x and x + gap + word_width > width:
                height += line_height
                x = line_height = 0.
            elif x:
                x += gap
            gap = 0.
            for advance, leading in word:
                if x and x + advance > width:
                    height += line_height
                    x = line_height = 0.
                x += advance
                line_height = max(line_height, leading)
        height += line_height or 1.
    # Office and FreeType use slightly different leading/rounding. Reserve
    # space rather than squeezing glyphs. The user can refine layout later.
    return height * 1.2 + 2


def _text(shape, substitutions):
    body = shape.find("p:txBody", NS)
    if body is None:
        return
    for paragraph in body.findall("a:p", NS):
        _paragraph(paragraph, substitutions)
    extent = shape.find("p:spPr/a:xfrm/a:ext", NS)
    if extent is not None:
        props = body.find("a:bodyPr", NS)
        horizontal = sum(int(props.get(key, "0")) for key in ("lIns", "rIns")) / 12700
        vertical = sum(int(props.get(key, "0")) for key in ("tIns", "bIns")) / 12700
        width = max(1., int(extent.get("cx")) / 12700 - horizontal)
        required = _natural_text_height(body, width) + vertical
        extent.set("cy", str(max(int(extent.get("cy")), math.ceil(required * 12700))))


def fixed_native_pdf_layout(path: Path, payload):
    pages=payload.get("slides",[payload])
    by_slide={f"ppt/slides/slide{number}.xml":page for number,page in enumerate(pages,1)
              if page.get("native_pdf_background_mode")=="full_original"
              and page.get("native_pdf_overlay_mode")=="editable"}
    substitutions = set()
    with ZipFile(path) as archive:
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]
    with ZipFile(path, "w") as archive:
        for info, data in entries:
            if info.filename in by_slide:
                page=by_slide[info.filename]
                scale=page["settings"]["inches_per_pixel"]*72
                pdf_scale=scale*page["native_pdf_zoom"]
                # Names are slide-local; even identical names on separate
                # pages must use that page's own cells and paragraph layout.
                objects={obj["id"]:obj for obj in page["objects"]}
                root = _parse_slide(data)
                for frame in root.findall(".//p:graphicFrame", NS):
                    properties = frame.find("p:nvGraphicFramePr/p:cNvPr", NS)
                    assert properties is not None, "Generated graphic frame has no identity"
                    name = properties.attrib["name"]
                    obj = objects.get(name)
                    if obj is None:
                        continue
                    rows = frame.findall(".//a:tbl/a:tr", NS)
                    for cell in obj["cells"]:
                        xmlcell = rows[cell["row"]].findall("a:tc", NS)[cell["col"]]
                        _cell(xmlcell, cell, obj, scale, pdf_scale, substitutions)
                for shape in root.findall(".//p:sp", NS):
                    properties = shape.find("p:nvSpPr/p:cNvPr", NS)
                    obj = objects.get(properties.get("name")) if properties is not None else None
                    if obj is not None and obj["type"] == "text":
                        _text(shape, substitutions)
                data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            archive.writestr(info, data)
    if substitutions:
        details = ", ".join(f"{source} → {target}" for source, target in sorted(substitutions))
        warnings.warn("Native PDF font substitution: " + details +
                      ". Point sizes are unchanged; text is not compressed to source bounds.",
                      RuntimeWarning, stacklevel=2)
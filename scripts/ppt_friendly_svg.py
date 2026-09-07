#!/usr/bin/env python3
"""Rewrite matplotlib SVGs so PowerPoint can treat labels as real text.

Matplotlib defaults to glyph outlines (`svg.fonttype=path`). Office also
ignores CSS `style=` on `<text>`, quoted font stacks like STIX/DejaVu, and
`rotate(-0 ...)`. This converts comments+outlines or CSS text into:

    <text font-family="Arial" font-size="9" fill="#2c3338" x="..." y="...">...</text>
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
PPT_FONT = "Arial"
FONT_SCALE = 100.0  # matplotlib.textpath.TextToPath.FONT_SCALE

ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)
ET.register_namespace("dc", "http://purl.org/dc/elements/1.1/")
ET.register_namespace("cc", "http://creativecommons.org/ns#")
ET.register_namespace("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")

_Q = {
    "g": f"{{{SVG_NS}}}g",
    "text": f"{{{SVG_NS}}}text",
    "tspan": f"{{{SVG_NS}}}tspan",
    "defs": f"{{{SVG_NS}}}defs",
    "path": f"{{{SVG_NS}}}path",
    "use": f"{{{SVG_NS}}}use",
    "clipPath": f"{{{SVG_NS}}}clipPath",
    "style": f"{{{SVG_NS}}}style",
}

_SUP = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")
_SUB = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
_TF = re.compile(r"(translate|rotate|scale)\(([^)]*)\)")
_HREF = f"{{{XLINK_NS}}}href"


def configure_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.formatter.use_mathtext": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        }
    )


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _is_comment(el: ET.Element) -> bool:
    return el.tag is ET.Comment


def _nums(blob: str) -> list[float]:
    return [float(x) for x in re.split(r"[,\s]+", blob.strip()) if x]


def parse_transform(value: str | None) -> dict:
    out: dict = {"translate": (0.0, 0.0), "rotate": None, "scale": None}
    if not value:
        return out
    for kind, raw in _TF.findall(value):
        nums = _nums(raw)
        if kind == "translate":
            out["translate"] = (nums[0], nums[1] if len(nums) > 1 else 0.0)
        elif kind == "rotate":
            out["rotate"] = tuple(nums)
        elif kind == "scale":
            out["scale"] = (nums[0], nums[1] if len(nums) > 1 else nums[0])
    return out


def parse_style(value: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (value or "").split(";"):
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        out[key.strip()] = val.strip()
    return out


def to_sup(text: str) -> str:
    chars = []
    for ch in text.replace("−", "-"):
        if ch == ".":
            chars.append("·")
        else:
            chars.append(ch.translate(_SUP))
    return "".join(chars)


def to_sub(text: str) -> str:
    chars = []
    for ch in text.replace("−", "-"):
        if ch == ".":
            chars.append(".")
        else:
            chars.append(ch.translate(_SUB))
    return "".join(chars)


def math_to_plain(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1]
    for cmd in (
        r"\mathdefault",
        r"\mathrm",
        r"\mathbf",
        r"\mathit",
        r"\textrm",
        r"\textsf",
        r"\text",
        r"\left",
        r"\right",
        r"\operatorname",
    ):
        text = text.replace(cmd, "")
    text = text.replace(r"\,", "").replace(r"\;", " ").replace(r"\ ", " ")
    text = text.replace(r"\times", "×").replace(r"\cdot", "·")
    text = text.replace(r"\minus", "-")
    text = re.sub(r"\^{([^{}]+)}", lambda m: to_sup(m.group(1)), text)
    text = re.sub(r"_{([^{}]+)}", lambda m: to_sub(m.group(1)), text)
    text = re.sub(r"\^(\d)", lambda m: to_sup(m.group(1)), text)
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    return text


def _font_size_from_scale(scale: tuple[float, float] | None, fallback: float | None) -> float | None:
    if scale is None:
        return fallback
    return abs(scale[0]) * FONT_SCALE


def _font_size_from_style(style: dict[str, str], fallback: float | None) -> float | None:
    raw = style.get("font-size")
    if not raw:
        return fallback
    try:
        return float(re.sub(r"[a-zA-Z%]+", "", raw))
    except ValueError:
        return fallback


def _fill_from_style(style: dict[str, str], fallback: str = "#2c3338") -> str:
    return style.get("fill") or fallback


def _looks_like_mpl_text(rest: list[ET.Element]) -> bool:
    if len(rest) != 1:
        return False
    child = rest[0]
    name = _local(child.tag)
    if name == "text":
        return True
    if name != "g":
        return False
    tf = child.get("transform") or ""
    if "scale(" in tf:
        return True
    kids = [_local(k.tag) for k in child]
    return any(k in {"text", "use", "defs", "path"} for k in kids)


def _first_text(el: ET.Element) -> ET.Element | None:
    if _local(el.tag) == "text":
        return el
    for child in el:
        found = _first_text(child)
        if found is not None:
            return found
    return None


def _ppt_text_element(
    *,
    content: str,
    x: float,
    y: float,
    size: float,
    fill: str,
    anchor: str | None,
    rotate: tuple[float, ...] | None,
    weight: str | None = None,
) -> ET.Element:
    text = ET.Element(_Q["text"])
    text.text = content
    text.set("x", f"{x:.6g}")
    text.set("y", f"{y:.6g}")
    text.set("font-family", PPT_FONT)
    text.set("font-size", f"{size:.4g}")
    text.set("fill", fill)
    if anchor and anchor != "start":
        text.set("text-anchor", anchor)
    if weight and weight not in {"400", "normal"}:
        text.set("font-weight", weight)
    angle = None
    if rotate:
        angle = rotate[0]
        if abs(angle) < 1e-6:
            angle = None
    if angle is not None:
        text.set("transform", f"rotate({angle:.4g}, {x:.6g}, {y:.6g})")
    return text


def _convert_mpl_group(el: ET.Element) -> bool:
    children = list(el)
    if not children or not _is_comment(children[0]) or not _looks_like_mpl_text(children[1:]):
        return False
    label = math_to_plain(children[0].text or "")
    if not label:
        return False
    inner = children[1]
    style = parse_style(inner.get("style"))
    tf = parse_transform(inner.get("transform"))
    node = _first_text(inner)
    node_style = parse_style(node.get("style") if node is not None else None)
    style = {**style, **node_style}
    x, y = tf["translate"]
    if node is not None:
        if node.get("x"):
            x += float(node.get("x"))
        if node.get("y"):
            y += float(node.get("y"))
        ntf = parse_transform(node.get("transform"))
        if ntf["translate"] != (0.0, 0.0):
            x += ntf["translate"][0]
            y += ntf["translate"][1]
        if ntf["rotate"] and not tf["rotate"]:
            tf["rotate"] = ntf["rotate"]
    size = _font_size_from_scale(tf["scale"], None)
    if size is None:
        descendant = None
        for item in inner.iter():
            sz = _font_size_from_style(parse_style(item.get("style")), None)
            if sz is None:
                continue
            descendant = sz if descendant is None else max(descendant, sz)
        size = descendant
    size = _font_size_from_style(style, size) or 9.0
    fill = _fill_from_style(style)
    anchor = style.get("text-anchor")
    weight = style.get("font-weight")
    rotate = tf["rotate"]
    if rotate and len(rotate) == 1:
        rotate = (rotate[0], x, y)
    replacement = _ppt_text_element(
        content=label,
        x=x,
        y=y,
        size=size,
        fill=fill,
        anchor=anchor,
        rotate=rotate,
        weight=weight,
    )
    attrib = dict(el.attrib)
    el.clear()
    el.attrib.update(attrib)
    el.append(replacement)
    return True


def _restyle_existing_text(el: ET.Element) -> None:
    if _local(el.tag) != "text":
        return
    style = parse_style(el.get("style"))
    if style:
        size = _font_size_from_style(style, None)
        if size is not None:
            el.set("font-size", f"{size:.4g}")
        fill = style.get("fill")
        if fill:
            el.set("fill", fill)
        anchor = style.get("text-anchor")
        if anchor:
            el.set("text-anchor", anchor)
        weight = style.get("font-weight")
        if weight and weight not in {"400", "normal"}:
            el.set("font-weight", weight)
        opacity = style.get("opacity")
        if opacity:
            el.set("opacity", opacity)
        el.attrib.pop("style", None)
    el.set("font-family", PPT_FONT)
    tf = el.get("transform")
    if tf:
        parsed = parse_transform(tf)
        rot = parsed["rotate"]
        if rot and abs(rot[0]) < 1e-6:
            el.attrib.pop("transform", None)
        elif rot:
            x = float(el.get("x") or (rot[1] if len(rot) > 1 else 0.0))
            y = float(el.get("y") or (rot[2] if len(rot) > 2 else 0.0))
            el.set("transform", f"rotate({rot[0]:.4g}, {x:.6g}, {y:.6g})")
    # Merge leftover tspans into one string when they are just split glyphs.
    tspans = [c for c in list(el) if _local(c.tag) == "tspan"]
    if tspans and not (el.text or "").strip():
        parts = []
        max_size = 0.0
        for span in tspans:
            st = parse_style(span.get("style"))
            sz = _font_size_from_style(st, max_size or None) or 0.0
            max_size = max(max_size, sz)
        for span in tspans:
            st = parse_style(span.get("style"))
            sz = _font_size_from_style(st, max_size) or max_size
            chunk = "".join(span.itertext())
            if max_size and sz < 0.85 * max_size:
                parts.append(to_sup(chunk))
            else:
                parts.append(chunk)
            el.remove(span)
        el.text = "".join(parts)
        if max_size:
            el.set("font-size", f"{max_size:.4g}")


def _collect_used_ids(root: ET.Element) -> set[str]:
    used: set[str] = set()
    for el in root.iter():
        href = el.get(_HREF) or el.get("href") or ""
        if href.startswith("#"):
            used.add(href[1:])
        for key, val in el.attrib.items():
            if "url(#" in val:
                used.update(re.findall(r"url\(#([^)]+)\)", val))
    return used


def _prune_unused_defs(root: ET.Element) -> None:
    used = _collect_used_ids(root)
    for defs in root.iter(_Q["defs"]):
        for child in list(defs):
            cid = child.get("id")
            if cid and cid not in used and _local(child.tag) in {"path", "clipPath", "g"}:
                defs.remove(child)


def _repair_script_fontsize(root: ET.Element) -> None:
    """If conversion picked the superscript tspan size, restore the body size."""
    marks = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    for el in root.iter(_Q["text"]):
        content = "".join(el.itertext())
        if not any(ch in content for ch in marks):
            continue
        try:
            size = float(el.get("font-size") or "0")
        except ValueError:
            continue
        if abs(size - 5.6) < 0.05:
            el.set("font-size", "8")
        elif abs(size - 7) < 0.05:
            el.set("font-size", "10")


def sanitize_svg_tree(root: ET.Element) -> int:
    converted = 0
    for el in list(root.iter()):
        if _convert_mpl_group(el):
            converted += 1
    for el in root.iter(_Q["text"]):
        _restyle_existing_text(el)
    _repair_script_fontsize(root)
    _prune_unused_defs(root)
    return converted


def sanitize_svg_file(path: Path) -> int:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(path, parser=parser)
    root = tree.getroot()
    converted = sanitize_svg_tree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True, method="xml")
    return converted


def configure_and_save(fig, path: Path, **savefig_kw) -> Path:
    configure_matplotlib()
    path = Path(path)
    fig.savefig(path, **savefig_kw)
    if path.suffix.lower() == ".svg":
        sanitize_svg_file(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Make matplotlib SVGs PowerPoint-editable")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.svg")))
        else:
            files.append(path)
    if not files:
        print("no svg files", file=sys.stderr)
        return 1
    for path in files:
        n = sanitize_svg_file(path)
        text_n = path.read_text(encoding="utf-8").count("<text")
        print(f"{path.name}: converted_groups={n} text_nodes={text_n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

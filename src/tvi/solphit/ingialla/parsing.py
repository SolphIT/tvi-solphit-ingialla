from __future__ import annotations
from pathlib import Path
from lxml import etree
import re

WIKI_REDIRECT_RE = re.compile(r"^\s*#REDIRECT", re.IGNORECASE)

def extract_title_and_text(page_xml_path: Path) -> tuple[str, str, bool]:
    title, text, is_redirect = None, None, False
    try:
        context = etree.iterparse(str(page_xml_path), events=("end",))
        for _, elem in context:
            tag = elem.tag
            if tag.endswith("title"):
                title = elem.text or ""
            elif tag.endswith("text"):
                text = elem.text or ""
            elif tag.endswith("redirect"):
                is_redirect = True
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        if not is_redirect and text and WIKI_REDIRECT_RE.match(text):
            is_redirect = True
    except Exception:
        root = etree.parse(str(page_xml_path)).getroot()
        t = root.find("./{*}title"); title = (t.text if t is not None else "") or ""
        te = root.find(".//{*}text"); text = (te.text if te is not None else "") or ""
        if root.find("./{*}redirect") is not None or WIKI_REDIRECT_RE.match(text or ""):
            is_redirect = True
    return (title or "").strip(), (text or ""), is_redirect

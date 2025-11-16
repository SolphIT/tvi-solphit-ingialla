from __future__ import annotations
import re

def simple_wikitext_clean(s: str) -> str:
    """
    Cleans basic Wikipedia wikitext for chunking/embedding.
    - Removes comments, templates, file/image links, refs, HTML tags.
    - Simplifies section headers and links.
    """
    if not s:
        return ""

    # Remove comments
    s = re.sub(r"<!--.*?-->", " ", s, flags=re.DOTALL)

    # Remove templates {{...}}
    s = re.sub(r"\{\{[^\{\}]*\}\}", " ", s)

    # IMPORTANT: remove file/image links BEFORE generic link simplification
    # e.g. [[File:x.jpg]] or [[Image:y.png]]
    s = re.sub(r"\[\[(?:File|Image):[^\]\n]+\]\]", " ", s, flags=re.IGNORECASE)

    # Replace [[A|B]] with B
    s = re.sub(r"\[\[([^\]\n]+)\|([^\]\n]+)\]\]", r"\2", s)

    # Replace [[A]] with A
    s = re.sub(r"\[\[([^\]\n]+)\]\]", r"\1", s)

    # Simplify section headers — allow leading whitespace
    s = re.sub(r"^\s*=+\s*(.*?)\s*=+\s*$", r"\1\n", s, flags=re.MULTILINE)

    # Remove <ref>...</ref> and <ref ... />
    s = re.sub(r"<ref[^>]*>.*?</ref>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<ref[^>]*/>", " ", s, flags=re.IGNORECASE)

    # Remove all other HTML/XML tags
    s = re.sub(r"<[^>]+>", " ", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s
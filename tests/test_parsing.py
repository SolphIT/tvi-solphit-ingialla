from pathlib import Path
import types
import pytest

# Module under test
from tvi.solphit.ingialla import parsing as parsing_mod
from tvi.solphit.ingialla.parsing import extract_title_and_text


def _write_xml(path: Path, *, title: str, text: str = "", redirect: bool = False) -> None:
    """
    Create a minimal MediaWiki-like page XML structure:
      <page>
        <title>...</title>
        [<redirect title="Target" />]
        <revision><text>...</text></revision>
      </page>
    """
    redirect_tag = '<redirect title="Target"/>' if redirect else ""
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<page>
  <title>{title}</title>
  {redirect_tag}
  <revision>
    <text>{text}</text>
  </revision>
</page>
"""
    path.write_text(xml, encoding="utf-8")


# --- TRY-PATH: happy case (no redirect) -------------------------------------

def test_try_path_extracts_title_and_text(tmp_path: Path):
    xml = tmp_path / "page.xml"
    _write_xml(xml, title="T", text="Body", redirect=False)

    title, text, is_redirect = extract_title_and_text(xml)

    assert title == "T"
    assert "Body" in text
    assert is_redirect is False


# --- TRY-PATH: redirect detected by text regex ------------------------------

def test_try_path_redirect_detected_by_text_regex(tmp_path: Path):
    xml = tmp_path / "redir_text.xml"
    _write_xml(xml, title="R", text="#REDIRECT [[Target]]", redirect=False)

    title, text, is_redirect = extract_title_and_text(xml)

    assert title == "R"
    assert is_redirect is True
    # text was set; regex path sets redirect without needing a <redirect/> element
    assert text.startswith("#REDIRECT")


# --- FALLBACK-PATH: normal page (no redirect element, no redirect text) -----

def test_fallback_path_no_redirect(monkeypatch, tmp_path: Path):
    xml = tmp_path / "fallback_page.xml"
    _write_xml(xml, title="T", text="Body", redirect=False)

    # Force the try/streaming path to raise so the code takes the except/fallback branch.
    def boom(*a, **k):
        raise RuntimeError("forced iterparse error")

    monkeypatch.setattr(parsing_mod.etree, "iterparse", boom)

    title, text, is_redirect = extract_title_and_text(xml)

    assert title == "T"
    assert "Body" in text
    assert is_redirect is False


# --- FALLBACK-PATH: redirect detected by <redirect/> element ----------------

def test_fallback_path_redirect_element(monkeypatch, tmp_path: Path):
    xml = tmp_path / "fallback_redirect.xml"
    _write_xml(xml, title="R", text="Anything", redirect=True)

    # Force fallback path again
    def boom(*a, **k):
        raise RuntimeError("forced iterparse error")

    monkeypatch.setattr(parsing_mod.etree, "iterparse", boom)

    title, text, is_redirect = extract_title_and_text(xml)

    assert title == "R"
    assert is_redirect is True
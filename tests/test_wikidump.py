# tests/test_wikidump_all.py

import os
import builtins
from pathlib import Path

import pytest

from tvi.solphit.ingialla import wikidump as wd
from tvi.solphit.ingialla.wikidump import safe_filename, get_article_path


# ----------------------------
# safe_filename() coverage
# ----------------------------

@pytest.mark.parametrize(
    "raw",
    [
        "CON",
        "con",              # mixed case
        "CON.txt",          # with extension
        "PrN",              # mixed case
        "PRN.txt",
        "AUX",
        "aux.TXT",
        "NUL",
        "nul.dat",
        "COM1",
        "com1.txt",
        "COM9",
        "LPT1",
        "lpt1.md",
        "LPT9",
    ],
)
def test_safe_filename_reserved_branch_always_hits_assignment(raw):
    """
    Explicitly cover the Windows-reserved branch multiple times.
    This guarantees execution of the line: base = f"{base}_file".
    """
    name = safe_filename(raw)
    # The reserved-case path appends "_file" to the base before hashing.
    assert "_file" in name
    # Must have "-<hash>" suffix of at least 8 hex chars
    assert len(name.split("-")[-1]) >= 8


def test_safe_filename_trimming_and_hash_suffix():
    # trailing spaces/dots trimmed; still suffixed
    name2 = safe_filename("Name... ")
    assert not name2.endswith(" ") and not name2.endswith(".")
    assert len(name2.split("-")[-1]) >= 8


def test_safe_filename_general_cleaning_and_hash_len():
    # Content with illegal chars and long content gets cleaned and suffixed
    raw = 'Illegals:*?"<>|/\\  name.. '
    name = safe_filename(raw)
    # Allow alnum, space, dot, underscore, hyphen and the "-<hash>" suffix chars
    assert all(
        c.isalnum() or c in " ._-{}"
        for c in name.replace("-", "").replace(name.split("-")[-1], "{}")
    ), "unexpected characters"
    # Contains "-<hash8>" suffix
    assert len(name.split("-")[-1]) >= 8


# ----------------------------
# get_article_path() coverage
# ----------------------------

def test_get_article_path_shortening_branch_targets_filename_not_full_path(tmp_path: Path):
    """
    When max_path is very small relative to the hashed folder path, the function
    shortens the *filename* via usable_len, but cannot guarantee the full absolute
    path is <= max_path (folder may already exceed it).
    We assert the filename shortening and directory creation.
    """
    title = "A" * 300  # very long
    out = get_article_path(str(tmp_path), title, ext=".xml", max_path=60)
    p = Path(out)

    # Folder hierarchy exists
    assert p.parent.exists()

    # Verify filename shortening honored usable_len >= 24
    fname_no_ext = p.stem
    assert len(fname_no_ext) <= 24  # shortened name per usable_len logic
    # Do not assert on total len(out), since folder length can exceed max_path.


# ----------------------------
# extract_articles() coverage
# ----------------------------

def _dump_two_pages() -> str:
    return """<?xml version="1.0" encoding="utf-8"?>
<mediawiki>
  <page><title>KEEP</title><revision><text>One</text></revision></page>
  <page><title>SKIP</title><revision><text>Two</text></revision></page>
</mediawiki>
"""


def _tiny_dump(ns: str = "http://example.org") -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<mediawiki xmlns="{ns}">
  <page><title>Alpha</title><revision><text>Text A</text></revision></page>
  <page><title>Beta</title><revision><text>Text B</text></revision></page>
</mediawiki>
"""


def test_extract_articles_saves_pages(tmp_path: Path, monkeypatch):
    xml = tmp_path / "dump.xml"
    xml.write_text(_tiny_dump(), encoding="utf-8")
    out_dir = tmp_path / "out"

    # Fake ES facade functions
    called = {"ensure": 0, "mark": []}

    def fake_ensure_articles_index():
        called["ensure"] += 1
        class _E: ...
        return _E()

    def fake_already_split(es, path: str) -> bool:
        return False

    def fake_mark_split_done(es, path: str, title: str):
        called["mark"].append((Path(path).name, title))

    monkeypatch.setattr(wd, "ensure_articles_index", fake_ensure_articles_index)
    monkeypatch.setattr(wd, "already_split", fake_already_split)
    monkeypatch.setattr(wd, "mark_split_done", fake_mark_split_done)

    saved = wd.extract_articles(str(xml), str(out_dir), max_pages=None)

    assert saved == 2
    assert len(called["mark"]) == 2
    assert any("Alpha" in t for _, t in called["mark"])
    assert any("Beta" in t for _, t in called["mark"])

    files = list(out_dir.rglob("*.xml"))
    assert len(files) == 2


def test_extract_articles_respects_max_pages(tmp_path: Path, monkeypatch):
    xml = tmp_path / "dump.xml"
    xml.write_text(_tiny_dump(), encoding="utf-8")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(wd, "ensure_articles_index", lambda: object())
    monkeypatch.setattr(wd, "already_split", lambda *a, **k: False)
    monkeypatch.setattr(wd, "mark_split_done", lambda *a, **k: None)

    saved = wd.extract_articles(str(xml), str(out_dir), max_pages=1)
    assert saved == 1


def test_extract_articles_handles_already_split(tmp_path: Path, monkeypatch):
    xml = tmp_path / "dump.xml"
    xml.write_text(_dump_two_pages(), encoding="utf-8")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(wd, "ensure_articles_index", lambda: object())
    calls = {"mark": []}
    monkeypatch.setattr(wd, "mark_split_done", lambda es, p, t: calls["mark"].append(t))

    def fake_already(es, path):
        return "SKIP" in path

    monkeypatch.setattr(wd, "already_split", fake_already)

    saved = wd.extract_articles(str(xml), str(out_dir))

    assert saved == 1
    assert calls["mark"] == ["KEEP"]


def test_extract_articles_handles_write_oserror(tmp_path: Path, monkeypatch):
    xml = tmp_path / "dump.xml"
    xml.write_text(_tiny_dump(), encoding="utf-8")
    out_dir = tmp_path / "out"

    # ES stubs
    monkeypatch.setattr(wd, "ensure_articles_index", lambda: object())
    monkeypatch.setattr(wd, "already_split", lambda *a, **k: False)
    monkeypatch.setattr(wd, "mark_split_done", lambda *a, **k: None)

    real_open = builtins.open

    def flaky_open(path, mode="r", *a, **k):
        # Fail only on article writes, not on reading the XML file itself
        if "wb" in mode and str(path) != str(xml):
            raise OSError("disk full")
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", flaky_open)

    saved = wd.extract_articles(str(xml), str(out_dir))
    assert saved == 0  # both writes failed -> OSError branch taken
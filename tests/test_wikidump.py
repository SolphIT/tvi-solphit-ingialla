from pathlib import Path
from tvi.solphit.ingialla import wikidump as wd

def _tiny_dump(ns: str = "http://example.org") -> str:
    return f"""<mediawiki xmlns="{ns}">
      <page><title>Alpha</title><revision><text>Text A</text></revision></page>
      <page><title>Beta</title><revision><text>Text B</text></revision></page>
    </mediawiki>"""

def test_extract_articles_saves_pages(tmp_path, monkeypatch):
    xml = tmp_path / "dump.xml"
    xml.write_text(_tiny_dump(), encoding="utf-8")
    out_dir = tmp_path / "out"

    # Fake ES facade functions
    called = {"ensure": 0, "mark": []}

    def fake_ensure_articles_index():
        called["ensure"] += 1
        class _E: pass
        return _E()

    def fake_already_split(es, path: str) -> bool:
        # Never split yet; test that we write both pages
        return False

    def fake_mark_split_done(es, path: str, title: str):
        called["mark"].append((Path(path).name, title))

    monkeypatch.setattr(wd, "ensure_articles_index", fake_ensure_articles_index)
    monkeypatch.setattr(wd, "already_split", fake_already_split)
    monkeypatch.setattr(wd, "mark_split_done", fake_mark_split_done)

    saved = wd.extract_articles(str(xml), str(out_dir), max_pages=None)

    # Two pages saved, two calls to mark_split_done, files exist
    assert saved == 2
    assert len(called["mark"]) == 2
    assert any("Alpha" in t for _, t in called["mark"])
    assert any("Beta" in t for _, t in called["mark"])

    # Files really got written
    files = list(out_dir.rglob("*.xml"))
    assert len(files) == 2

def test_extract_articles_respects_max_pages(tmp_path, monkeypatch):
    xml = tmp_path / "dump.xml"
    xml.write_text(_tiny_dump(), encoding="utf-8")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(wd, "ensure_articles_index", lambda: object())
    monkeypatch.setattr(wd, "already_split", lambda *a, **k: False)
    monkeypatch.setattr(wd, "mark_split_done", lambda *a, **k: None)

    saved = wd.extract_articles(str(xml), str(out_dir), max_pages=1)
    assert saved == 1
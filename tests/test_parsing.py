from pathlib import Path
from tvi.solphit.ingialla.parsing import extract_title_and_text

def test_extract_title_and_text(tmp_path: Path):
    xml = tmp_path / "p.xml"
    xml.write_text(
        '<page><title>T</title><revision><text>Body</text></revision></page>',
        encoding="utf-8"
    )
    title, text, is_redirect = extract_title_and_text(xml)
    assert title == "T"
    assert "Body" in text
    assert is_redirect is False
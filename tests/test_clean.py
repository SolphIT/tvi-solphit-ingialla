from tvi.solphit.ingialla.clean import simple_wikitext_clean

def test_simple_wikitext_clean_basic():
    s = """<!-- comment --> {{T}} [[A|B]] [[C]] [[File:x.jpg]]
    == Section == <ref>r</ref> <ref name="x"/> <div>H</div>"""
    out = simple_wikitext_clean(s)
    # minimal sanity: comments/templates/refs/tags removed, links simplified
    assert "comment" not in out
    assert "{" not in out and "}" not in out
    assert "File:" not in out
    assert "B" in out and "C" in out
    assert "Section" in out
    assert "<" not in out and ">" not in out
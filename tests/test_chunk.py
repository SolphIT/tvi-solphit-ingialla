from tvi.solphit.ingialla.chunk import chunk_text

def test_chunk_text_overlap():
    text = "abcdefghij"
    chunks = chunk_text(text, chunk_size=4, overlap=2)
    assert chunks[0] == "abcd"
    assert chunks[1] == "cdef"
    assert chunks[2] == "efgh"
    assert chunks[-1].endswith("ij")
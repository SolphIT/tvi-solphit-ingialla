import pytest
from tvi.solphit.ingialla.chunk import chunk_text


# --- Core behavior -----------------------------------------------------------

def test_chunk_text_empty_short_circuits():
    assert chunk_text("", 100, 10) == []


def test_chunk_text_overlap_normal():
    # 10 chars, chunk=4, overlap=2 => ["abcd","cdef","efgh","ghij"]
    text = "abcdefghij"
    chunks = chunk_text(text, chunk_size=4, overlap=2)
    assert chunks == ["abcd", "cdef", "efgh", "ghij"]


def test_chunk_text_overlap_larger_than_chunk_size_forces_step_1():
    # overlap >= chunk_size forces step=1 (max(1, chunk_size - overlap) => 1)
    chunks = chunk_text("abcdefghij", chunk_size=4, overlap=10)
    assert chunks[0] == "abcd"
    assert chunks[1] == "bcde"
    assert chunks[-1].endswith("ij")


# --- Additional edges to close remaining branch coverage ---------------------

def test_chunk_text_negative_overlap_increases_step():
    """
    Negative overlap -> step = chunk_size - overlap > chunk_size.
    This hits the branch where max(1, chunk_size - overlap) selects a value > 1
    in a different way than the normal overlap case.
    """
    text = "abcdefghij"  # len=10
    # chunk_size=4, overlap=-2 => step = 4 - (-2) = 6
    # i=0 -> "abcd"; i=6 -> "ghij"; stops after second chunk.
    chunks = chunk_text(text, chunk_size=4, overlap=-2)
    assert chunks == ["abcd", "ghij"]


def test_chunk_text_exact_fit_no_partial_and_break_condition():
    """
    Exact division: overlap=0, len(text) divisible by chunk_size.
    Ensures the final-iteration `if i + chunk_size >= len(text): break` branch is exercised.
    """
    text = "abcdefgh"  # len=8
    chunks = chunk_text(text, chunk_size=4, overlap=0)  # step = 4
    assert chunks == ["abcd", "efgh"]
    # No trailing partial, and loop exits via the >= break condition.


def test_chunk_text_zero_chunk_size_hits_empty_chunk_branch():
    """
    chunk_size = 0 => text[i:i+0] == '' (empty), so `if chunk:` is False.
    This covers the otherwise-missed false branch of `if chunk: chunks.append(...)`.
    The loop terminates naturally via range(...), no break needed.
    """
    text = "abcde"
    chunks = chunk_text(text, chunk_size=0, overlap=2)
    assert chunks == []  # nothing appended since every slice is empty
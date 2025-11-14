from tvi.solphit.ingialla import hello, cosine_similarity

def test_hello_returns_message():
    msg = hello("Tobias")
    assert "Ingialla says hello to Tobias" in msg

def test_cosine_similarity_basic():
    a = [1, 0, 0]
    b = [1, 0, 0]
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-9
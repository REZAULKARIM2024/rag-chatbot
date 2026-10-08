import ingest


def test_short_text_is_single_chunk():
    assert ingest.chunk_text("hello world") == ["hello world"]


def test_empty_text_gives_no_chunks():
    assert ingest.chunk_text("   \n  ") == []


def test_whitespace_is_normalized():
    assert ingest.chunk_text("a   b\n\nc") == ["a b c"]


def test_chunks_respect_size_and_overlap():
    text = "x" * 2000
    chunks = ingest.chunk_text(text, chunk_size=800, overlap=150)
    assert all(len(c) <= 800 for c in chunks)
    assert len(chunks) >= 3
    # consecutive chunks share `overlap` characters
    assert chunks[0][-150:] == chunks[1][:150]


def test_all_content_is_covered():
    words = " ".join(f"w{i}" for i in range(500))
    chunks = ingest.chunk_text(words, chunk_size=200, overlap=40)
    joined = " ".join(chunks)
    assert "w0" in chunks[0] and "w499" in joined

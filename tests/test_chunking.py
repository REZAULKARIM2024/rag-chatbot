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


# ---- sentence-aware chunking -------------------------------------------------

SENTENCES = (
    "REST API tests check status codes. They also check headers and bodies! "
    "Is a 404 returned for a missing resource? Yes, it should be. "
    "Contract tests agree on message shape. Pact is a common tool."
)


def test_sentence_chunks_never_cut_a_sentence():
    chunks = ingest.chunk_text(SENTENCES, chunk_size=90, overlap=30, strategy="sentence")
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 90
        assert chunk[-1] in ".!?"          # ends on a sentence boundary
        assert chunk[0].isupper()          # starts on a sentence start


def test_sentence_chunks_cover_all_sentences():
    chunks = ingest.chunk_text(SENTENCES, chunk_size=90, overlap=30, strategy="sentence")
    joined = " ".join(chunks)
    for sentence in ingest.split_sentences(SENTENCES):
        assert sentence in joined


def test_sentence_chunks_overlap_by_whole_sentences():
    chunks = ingest.chunk_text(SENTENCES, chunk_size=90, overlap=45, strategy="sentence")
    shared = [s for s in ingest.split_sentences(SENTENCES) if s in chunks[0] and s in chunks[1]]
    assert shared, "consecutive chunks should share at least one whole sentence"


def test_sentence_chunker_handles_one_long_sentence():
    long_sentence = "word " * 400  # no sentence boundary, longer than chunk_size
    chunks = ingest.chunk_text(long_sentence, chunk_size=200, overlap=40, strategy="sentence")
    assert all(len(c) <= 200 for c in chunks) and len(chunks) > 1


def test_paragraph_break_splits_sentences():
    assert ingest.split_sentences("Title Line\n\nFirst sentence. Second one.") == [
        "Title Line", "First sentence.", "Second one."]


def test_short_and_empty_text_for_sentence_chunker():
    assert ingest.chunk_text("Hello world.", strategy="sentence") == ["Hello world."]
    assert ingest.chunk_text("  \n ", strategy="sentence") == []


def test_unknown_strategy_raises():
    import pytest
    with pytest.raises(ValueError):
        ingest.chunk_text("text", strategy="bogus")

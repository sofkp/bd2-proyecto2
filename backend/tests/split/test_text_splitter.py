from backend.src.split.split_text import SplitText

def test_text_splitter_creates_paragraph_chunks_with_indices():
    splitter = SplitText(min_chars=0, max_chars=200)
    text = "Primer parrafo del documento.\n\nSegundo parrafo para buscar."

    chunks = splitter.split_text(text, document_id="doc1")

    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "doc1_text_0"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["content"] == "Primer parrafo del documento."
    assert chunks[0]["modality"] == "text"
    assert chunks[0]["metadata"]["num_chars"] == len(chunks[0]["content"])

    assert chunks[1]["chunk_id"] == "doc1_text_1"
    assert chunks[1]["chunk_index"] == 1
    assert chunks[1]["content"] == "Segundo parrafo para buscar."

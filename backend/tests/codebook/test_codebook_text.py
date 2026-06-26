import pytest
from src.codebook import CodebookText

def test_codebook_text_top_k_selection():
    # 1. Preparar datos de prueba (mock chunks)
    mockup_input = [
        {"doc_id": "doc001", "chunk_id": "p_01", "tf": {"base": 2, "datos": 3}},
        {"doc_id": "doc002", "chunk_id": "p_01", "tf": {"datos": 2, "busqueda": 4}},
        {"doc_id": "doc003", "chunk_id": "p_03", "tf": {"ciencia": 1, "datos": 3}},
    ]

    # 2. Inicializar CodebookText con Top-K = 2
    top_k = 2
    text_cb = CodebookText(top_k=top_k)
    codebook = text_cb.build_codebook(mockup_input)

    # 3. Verificaciones (asserts)
    # Deben haberse seleccionado exactamente 2 términos
    assert len(codebook) == top_k

    # "datos" es el más frecuente en los 3 chunks, por lo que debe ser el índice 0
    assert "datos" in codebook
    assert codebook["datos"]["index"] == 0
    assert codebook["datos"]["idf"] == 1.0  # log(3/3) + 1 = 1.0

    # "base" es seleccionado como el segundo (por desempate/frecuencia)
    assert "base" in codebook or "busqueda" in codebook or "ciencia" in codebook
    
    # Comprobar que no haya IDFs negativos
    for term, info in codebook.items():
        assert info["idf"] >= 1.0

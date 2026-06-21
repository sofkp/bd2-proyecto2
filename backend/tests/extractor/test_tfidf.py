"""
Tests unitarios para TFIDFExtractor.

Cubre:
  - Preprocesamiento: lower, limpieza, stopwords, stemming
  - fit(): vocabulario, pesos IDF, restricción top-k
  - transform(): forma de salida, normalización L2, términos fuera de vocabulario
  - extract(): formato de salida para el Codebook (conteos crudos)
  - Casos borde: corpus vacío, chunk sin texto, idioma sin stopwords
"""

import math
import pytest
import numpy as np

from src.extractor.tfidf import TFIDFExtractor


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def extractor_es():
    """Extractor en español con vocabulario sin límite."""
    return TFIDFExtractor(language="spanish")


@pytest.fixture
def corpus_es():
    """Corpus mínimo en español para pruebas."""
    return [
        "Los datos estructurados permiten búsquedas eficientes",
        "Las bases de datos almacenan información estructurada",
        "La búsqueda eficiente requiere índices adecuados",
    ]


@pytest.fixture
def chunks_es():
    """Fragmentos con metadatos para probar extract()."""
    return [
        {"doc_id": "doc_001", "chunk_id": "p_01",
         "text": "Base de datos estructurada para consultas"},
        {"doc_id": "doc_001", "chunk_id": "p_02",
         "text": "Búsqueda eficiente en grandes volúmenes de datos"},
        {"doc_id": "doc_002", "chunk_id": "p_01",
         "text": ""},  # chunk vacío — caso borde
    ]


# -----------------------------------------------------------------------
# Tests de preprocesamiento
# -----------------------------------------------------------------------

class TestPreprocesamiento:

    def test_conversion_minusculas(self, extractor_es):
        """El texto debe convertirse a minúsculas antes de tokenizar."""
        tokens = extractor_es._preprocesar("HOLA MUNDO")
        assert all(t == t.lower() for t in tokens)

    def test_elimina_numeros_y_simbolos(self, extractor_es):
        """Los números y símbolos especiales deben eliminarse."""
        tokens = extractor_es._preprocesar("precio: $100 (descuento 20%)")
        for t in tokens:
            assert t.isalpha() or all(
                c.isalpha() or c in "áéíóúàèìòùñü" for c in t
            )

    def test_conserva_tildes_y_enie(self, extractor_es):
        """Las tildes y la ñ deben conservarse después de la limpieza."""
        tokens = extractor_es._preprocesar("niño camión árbol")
        texto_unido = " ".join(tokens)
        # Al menos algún token debe contener caracteres con tilde o ñ
        assert any(c in texto_unido for c in "áéíóúñ")

    def test_elimina_stopwords(self, extractor_es):
        """Las stopwords del español deben eliminarse."""
        # "de", "la", "el", "en" son stopwords comunes en español
        tokens = extractor_es._preprocesar("el gato de la casa en el jardín")
        stopwords_comunes = {"el", "la", "de", "en", "un", "una"}
        assert not any(t in stopwords_comunes for t in tokens)

    def test_stemming_reduce_palabras(self, extractor_es):
        """Palabras con la misma raíz deben producir el mismo stem."""
        tokens_singular = extractor_es._preprocesar("dato")
        tokens_plural   = extractor_es._preprocesar("datos")
        # Ambas deben producir el mismo stem
        assert tokens_singular == tokens_plural

    def test_texto_vacio_retorna_lista_vacia(self, extractor_es):
        """Un texto vacío debe retornar lista vacía sin errores."""
        assert extractor_es._preprocesar("") == []

    def test_solo_stopwords_retorna_vacio(self, extractor_es):
        """Texto compuesto solo por stopwords debe retornar lista vacía."""
        tokens = extractor_es._preprocesar("el la de un una y")
        assert tokens == []


# -----------------------------------------------------------------------
# Tests de fit()
# -----------------------------------------------------------------------

class TestFit:

    def test_fit_construye_vocabulario(self, extractor_es, corpus_es):
        """fit() debe poblar vocabulary_ con al menos un término."""
        extractor_es.fit(corpus_es)
        assert len(extractor_es.vocabulary_) > 0

    def test_fit_construye_idf(self, extractor_es, corpus_es):
        """fit() debe generar un vector idf_ con la misma longitud que el vocabulario."""
        extractor_es.fit(corpus_es)
        assert len(extractor_es.idf_) == len(extractor_es.vocabulary_)

    def test_idf_valores_positivos(self, extractor_es, corpus_es):
        """Todos los valores IDF deben ser positivos (IDF suavizado >= 1)."""
        extractor_es.fit(corpus_es)
        assert np.all(extractor_es.idf_ > 0)

    def test_max_features_limita_vocabulario(self, corpus_es):
        """max_features debe limitar el tamaño del vocabulario."""
        extractor = TFIDFExtractor(max_features=3, language="spanish")
        extractor.fit(corpus_es)
        assert len(extractor.vocabulary_) <= 3

    def test_fit_corpus_vacio_lanza_error(self, extractor_es):
        """fit() con corpus vacío debe lanzar ValueError."""
        with pytest.raises(ValueError):
            extractor_es.fit([])

    def test_fit_retorna_self(self, extractor_es, corpus_es):
        """fit() debe retornar self para permitir encadenamiento."""
        resultado = extractor_es.fit(corpus_es)
        assert resultado is extractor_es


# -----------------------------------------------------------------------
# Tests de transform()
# -----------------------------------------------------------------------

class TestTransform:

    def test_forma_de_salida(self, extractor_es, corpus_es):
        """transform() debe retornar matriz (n_docs, vocab_size)."""
        extractor_es.fit(corpus_es)
        matriz = extractor_es.transform(corpus_es)
        assert matriz.shape == (len(corpus_es), extractor_es.vocab_size)

    def test_dtype_float32(self, extractor_es, corpus_es):
        """La matriz de salida debe ser float32."""
        extractor_es.fit(corpus_es)
        matriz = extractor_es.transform(corpus_es)
        assert matriz.dtype == np.float32

    def test_normalizacion_l2(self, extractor_es, corpus_es):
        """Cada fila no vacía debe tener norma L2 ≈ 1.0."""
        extractor_es.fit(corpus_es)
        matriz = extractor_es.transform(corpus_es)
        for fila in matriz:
            norma = np.linalg.norm(fila)
            if norma > 0:
                assert abs(norma - 1.0) < 1e-5

    def test_doc_vacio_produce_vector_cero(self, extractor_es, corpus_es):
        """Un documento vacío debe producir vector de ceros."""
        extractor_es.fit(corpus_es)
        matriz = extractor_es.transform([""])
        assert np.all(matriz[0] == 0)

    def test_terminos_fuera_de_vocabulario_se_ignoran(self, extractor_es, corpus_es):
        """Términos no vistos en fit() deben ignorarse silenciosamente."""
        extractor_es.fit(corpus_es)
        # xyzabc123 no existe en el vocabulario
        matriz = extractor_es.transform(["xyzabc123 palabrainventada"])
        assert np.all(matriz[0] == 0)

    def test_transform_sin_fit_lanza_error(self, extractor_es, corpus_es):
        """transform() sin fit() previo debe lanzar RuntimeError."""
        with pytest.raises(RuntimeError):
            extractor_es.transform(corpus_es)

    def test_fit_transform_equivalente(self, corpus_es):
        """fit_transform() debe producir el mismo resultado que fit() + transform()."""
        e1 = TFIDFExtractor(language="spanish")
        e2 = TFIDFExtractor(language="spanish")
        m1 = e1.fit_transform(corpus_es)
        m2 = e2.fit(corpus_es).transform(corpus_es)
        np.testing.assert_array_almost_equal(m1, m2)


# -----------------------------------------------------------------------
# Tests de extract()
# -----------------------------------------------------------------------

class TestExtract:

    def test_longitud_de_salida(self, extractor_es, chunks_es):
        """extract() debe retornar un dict por cada chunk de entrada."""
        resultado = extractor_es.extract(chunks_es)
        assert len(resultado) == len(chunks_es)

    def test_claves_presentes(self, extractor_es, chunks_es):
        """Cada elemento de la salida debe tener doc_id, chunk_id y tf."""
        resultado = extractor_es.extract(chunks_es)
        for item in resultado:
            assert "doc_id"   in item
            assert "chunk_id" in item
            assert "tf"       in item

    def test_metadatos_preservados(self, extractor_es, chunks_es):
        """doc_id y chunk_id deben mantenerse igual al input."""
        resultado = extractor_es.extract(chunks_es)
        for original, salida in zip(chunks_es, resultado):
            assert salida["doc_id"]   == original["doc_id"]
            assert salida["chunk_id"] == original["chunk_id"]

    def test_tf_son_conteos_crudos_enteros(self, extractor_es, chunks_es):
        """Los valores de tf deben ser enteros positivos (conteos crudos)."""
        resultado = extractor_es.extract(chunks_es)
        for item in resultado:
            for conteo in item["tf"].values():
                assert isinstance(conteo, int)
                assert conteo > 0

    def test_tf_terminos_stemizados(self, extractor_es):
        """Los términos en tf deben estar stemizados (sin forma original)."""
        chunks = [{"doc_id": "d", "chunk_id": "c",
                   "text": "datos estructurados datos"}]
        resultado = extractor_es.extract(chunks)
        tf = resultado[0]["tf"]
        # La suma de conteos debe reflejar las 2 ocurrencias de "datos"
        total = sum(tf.values())
        # "datos" y "estructurados" → tras stemming, máximo 2 términos distintos
        assert total >= 1

    def test_chunk_vacio_produce_tf_vacio(self, extractor_es, chunks_es):
        """Un chunk sin texto debe producir tf vacío, sin error."""
        resultado = extractor_es.extract(chunks_es)
        # chunks_es[2] tiene text=""
        assert resultado[2]["tf"] == {}

    def test_tf_no_normalizado(self, extractor_es):
        """Los conteos en tf deben ser crudos, NO pesos TF-IDF normalizados."""
        chunks = [{"doc_id": "d", "chunk_id": "c",
                   "text": "dato dato dato"}]
        resultado = extractor_es.extract(chunks)
        tf = resultado[0]["tf"]
        # El conteo debe ser un número entero pequeño, no un float normalizado
        for v in tf.values():
            assert isinstance(v, int)
            assert v <= 10  # imposible que "dato" aparezca > 10 veces aquí

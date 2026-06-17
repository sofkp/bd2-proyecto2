"""
Extractor de características TF-IDF para fragmentos de texto.

Pipeline
--------
  fit(corpus)       → construye el vocabulario y los pesos IDF a partir de todos los fragmentos
  transform(corpus) → retorna la matriz TF-IDF normalizada con L2

Fórmula utilizada
-----------------
  TF(t, d)   = 1 + log(count(t, d))   si count > 0, de lo contrario 0
  IDF(t)     = log((1 + N) / (1 + df(t))) + 1     (IDF suavizado)
  tfidf(t,d) = TF(t,d) * IDF(t)       luego normalizado con L2 por documento

donde N = número de documentos, df(t) = documentos que contienen el término t.

Nota sobre la tokenización
--------------------------
Este extractor espera fragmentos de texto ya divididos (strings planos). Realiza
únicamente tokenización básica por espacios + conversión a minúsculas. La eliminación
de stopwords, stemming y restricción del vocabulario (top-k) son responsabilidad
del módulo Codebook, para que cada módulo tenga una única responsabilidad.
"""

import math
from collections import Counter
from typing import Optional

import numpy as np

from .base import BaseExtractor


class TFIDFExtractor(BaseExtractor):
    """Calcula vectores de características TF-IDF para una colección de fragmentos de texto."""

    def __init__(self, max_features: Optional[int] = None) -> None:
        """
        Args:
            max_features: Conservar solo los *max_features* términos más frecuentes
                en el vocabulario. ``None`` conserva todos los términos.
        """
        self.max_features = max_features

        # Poblados por fit()
        self.vocabulary_: dict[str, int] = {}   # término → índice de columna
        self.idf_: np.ndarray = np.array([])    # forma (vocab_size,)
        self._fitted: bool = False


    # BaseExtractor interface

    def fit(self, corpus: list[str]) -> "TFIDFExtractor":
        """
        Construye el vocabulario y los pesos IDF a partir del *corpus*.

        Args:
            corpus: Lista de strings de texto (uno por fragmento).

        Returns:
            self
        """
        if not corpus:
            raise ValueError("corpus must not be empty")

        n_docs = len(corpus)
        tokenized = [self._tokenize(doc) for doc in corpus]

        # Frecuencia de documento: cuántos docs contienen cada término
        df: Counter = Counter()
        for tokens in tokenized:
            df.update(set(tokens))  # set → contar cada término una sola vez por doc

        # Opcionalmente restringir el vocabulario a los top-max_features términos
        if self.max_features is not None:
            selected = [term for term, _ in df.most_common(self.max_features)]
        else:
            selected = sorted(df.keys())

        self.vocabulary_ = {term: idx for idx, term in enumerate(selected)}

        # IDF suavizado: log((1+N)/(1+df(t))) + 1
        self.idf_ = np.array(
            [math.log((1 + n_docs) / (1 + df[term])) + 1.0 for term in selected],
            dtype=np.float32,
        )

        self._fitted = True
        return self

    def transform(self, corpus: list[str]) -> np.ndarray:
        """
        Calcula los vectores TF-IDF para cada documento del *corpus*.

        Args:
            corpus: Lista de strings de texto.

        Returns:
            Matriz de forma ``(len(corpus), vocab_size)`` con dtype float32.
            Cada fila está normalizada con L2 (vector unitario). Los vectores
            nulos permanecen como cero.
        """
        if not self._fitted:
            raise RuntimeError("Llamar a fit() antes de transform()")

        vocab_size = len(self.vocabulary_)
        matrix = np.zeros((len(corpus), vocab_size), dtype=np.float32)

        for i, doc in enumerate(corpus):
            tokens = self._tokenize(doc)
            if not tokens:
                continue

            tf_counts = Counter(tokens)

            for term, count in tf_counts.items():
                if term not in self.vocabulary_:
                    continue
                j = self.vocabulary_[term]
                # TF logarítmico: 1 + log(conteo) — evita valores negativos
                tf = 1.0 + math.log(count)
                matrix[i, j] = tf * self.idf_[j]

            # Normalizar la fila con L2
            norm = np.linalg.norm(matrix[i])
            if norm > 0.0:
                matrix[i] /= norm

        return matrix

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_feature_names(self) -> list[str]:
        """Retorna los términos en orden de índice de vocabulario."""
        return [t for t, _ in sorted(self.vocabulary_.items(), key=lambda x: x[1])]

    @property
    def vocab_size(self) -> int:
        """Número de términos en el vocabulario ajustado."""
        return len(self.vocabulary_)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Conversión a minúsculas y división por espacios. Filtra tokens vacíos."""
        return [t for t in text.lower().split() if t]
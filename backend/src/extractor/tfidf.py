import math
import re
from collections import Counter
from typing import Optional

import numpy as np

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem.snowball import SnowballStemmer
    from nltk.tokenize import word_tokenize
    _NLTK_DISPONIBLE = True
except ImportError:
    _NLTK_DISPONIBLE = False

from .base import BaseExtractor


def _asegurar_recursos_nltk() -> None:
    """Descarga los recursos NLTK necesarios si no están disponibles localmente."""
    if not _NLTK_DISPONIBLE:
        return
    recursos = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords",    "stopwords"),
    ]
    for ruta, nombre in recursos:
        try:
            nltk.data.find(ruta)
        except LookupError:
            nltk.download(nombre, quiet=True)


class TFIDFExtractor(BaseExtractor):
    """Extrae características TF-IDF para una colección de fragmentos de texto."""

    def __init__(
        self,
        max_features: Optional[int] = None,
        language: str = "spanish",
        custom_stopwords: Optional[set] = None,
    ) -> None:
        """
        input:
            max_features: Conservar solo los *max_features* términos más frecuentes
                en el vocabulario. ``None`` conserva todos los términos.
            language: Idioma para stopwords y stemming. Valores válidos: "spanish",
                "english", y cualquier idioma soportado por SnowballStemmer de NLTK.
            custom_stopwords: Conjunto adicional de palabras a eliminar,
                fusionado con las stopwords de NLTK.
        """
        _asegurar_recursos_nltk()

        self.max_features = max_features
        self.language = language

        # Cargar stopwords y stemmer según el idioma configurado
        self._stopwords: set[str] = self._cargar_stopwords(language, custom_stopwords)
        self._stemmer = self._cargar_stemmer(language)

        # Poblados por fit()
        self.vocabulary_: dict[str, int] = {}   # término → índice de columna
        self.idf_: np.ndarray = np.array([])    # forma (vocab_size,)
        self._fitted: bool = False


    # Interfaz BaseExtractor
    def fit(self, corpus: list[str]) -> "TFIDFExtractor":
        """
        Aprende el vocabulario y los pesos IDF a partir del corpus completo.

        Debe llamarse una sola vez antes de transform(). Internamente:
          1. Preprocesa cada doc (lower → limpieza → tokenización →
             stopwords → stemming) para obtener los tokens finales
          2. Calcula df(t): en cuántos documentos aparece cada término
          3. Si max_features está definido, conserva solo los top-k términos
             más frecuentes (los más informativos a nivel colección)
          4. Asigna un índice de columna a cada término → vocabulary_
          5. Calcula el IDF suavizado para cada término:
               IDF(t) = log((1 + N) / (1 + df(t))) + 1
             donde N = total de documentos. El +1 evita división por cero
             y suaviza la penalización a términos muy frecuentes.

        input: corpus: Lista de strings de texto, uno por fragmento.
        output: self — permite encadenar: extractor.fit(corpus).transform(corpus)
        """
        if not corpus:
            raise ValueError("El corpus no puede estar vacío")

        n_docs = len(corpus)
        tokenized = [self._preprocesar(doc) for doc in corpus]

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
        Calcula la matriz TF-IDF usando el vocabulario aprendido en fit().

        Por cada documento aplica el siguiente cálculo:
          1. Preprocesa el texto con _preprocesar() → tokens stemizados
          2. Cuenta las ocurrencias de cada token → tf_counts
          3. Para cada término en el vocabulario calcula:
               TF(t,d)    = 1 + log(count(t,d))  ← el +1 evita log(0)
               tfidf(t,d) = TF(t,d) × IDF(t)     ← IDF aprendido en fit()
          4. Normaliza el vector con norma L2 → vector unitario,
             lo que permite usar producto punto como similitud coseno
        input:
            corpus: Lista de strings. Pueden ser documentos no vistos en fit();
                    los términos fuera del vocabulario se ignoran.
        output:
            Matriz numpy de forma ``(len(corpus), vocab_size)`` con dtype float32.
            Documentos vacíos o sin términos en el vocabulario quedan como
            vector de ceros.
        """
        if not self._fitted:
            raise RuntimeError("Llamar a fit() antes de transform()")

        vocab_size = len(self.vocabulary_)
        matrix = np.zeros((len(corpus), vocab_size), dtype=np.float32)

        for i, doc in enumerate(corpus):
            tokens = self._preprocesar(doc)
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

    def extract(self, chunks: list[dict]) -> list[dict]:
        """
        Extrae conteos TF crudos por fragmento en el formato que espera el Codebook.

        Es el punto de entrada del pipeline Split → Extractor → Codebook.
        A diferencia de transform(), NO calcula IDF ni normaliza: solo cuenta
        cuántas veces aparece cada término (ya stemizado) en cada fragmento.

        Flujo interno:
          1. Por cada chunk llama a _preprocesar(text)  → tokens stemizados
          2. Cuenta frecuencias con Counter(tokens)     → conteos crudos
          3. Empaqueta resultado con doc_id y chunk_id  → dict de salida

        input:
            chunks: Lista de dicts. Cada dict debe tener:
                - "doc_id"   (str): identificador del documento padre
                - "chunk_id" (str): identificador único del fragmento
                - "text"     (str): contenido textual del fragmento
        output:
            Lista de dicts con la estructura que consume el Codebook:
            [
              {
                "doc_id":   "doc_001",
                "chunk_id": "p_01",
                "tf": { "bas": 2, "dat": 3 }  ← término stemizado : conteo crudo
              },
              ...
            ]
            Fragmentos vacíos tras el preprocesamiento se incluyen con "tf": {}
            para que el Codebook no pierda la referencia al chunk.
        """
        resultado: list[dict] = []

        for chunk in chunks:
            tokens = self._preprocesar(chunk["text"])
            tf = dict(Counter(tokens))  # conteos crudos sin normalizar
            resultado.append({
                "doc_id":   chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "tf":       tf,
            })

        return resultado


    # Auxiliares
    def _preprocesar(self, texto: str) -> list[str]:
        """
        Convierte un texto plano en lista de tokens listos para indexar.

        Pasos en orden:
            1. Minúsculas
            2. Eliminar caracteres no alfabéticos (conserva tildes y ñ)
            3. Tokenización con NLTK; fallback a split() si no está disponible
            4. Eliminar stopwords y tokens de un solo carácter
            5. Stemming con SnowballStemmer de NLTK (si está disponible)
        """
        # 1. Conversión a minúsculas
        texto = texto.lower()

        # 2. Eliminar caracteres no alfabéticos (conserva tildes y ñ)
        texto = re.sub(r'[^a-záéíóúàèìòùâêîôûäëïöüñ\s]', '', texto)

        # 3. Tokenización con NLTK; fallback a split si no está disponible
        if _NLTK_DISPONIBLE:
            try:
                tokens = word_tokenize(texto, language=self.language)
            except LookupError:
                tokens = texto.split()
        else:
            tokens = texto.split()

        # 4. Eliminar stopwords y tokens de un solo carácter
        tokens = [t for t in tokens if len(t) > 1 and t not in self._stopwords]

        # 5. Stemming — reduce cada palabra a su raíz léxica
        if self._stemmer is not None:
            tokens = [self._stemmer.stem(t) for t in tokens]

        return tokens

    def get_feature_names(self) -> list[str]:
        """Retorna los términos en orden de índice de vocabulario."""
        return [t for t, _ in sorted(self.vocabulary_.items(), key=lambda x: x[1])]

    @property
    def vocab_size(self) -> int:
        """Número de términos en el vocabulario ajustado."""
        return len(self.vocabulary_)


    # Métodos de clase para carga de recursos NLTK
    @staticmethod
    def _cargar_stopwords(language: str, custom: Optional[set]) -> set[str]:
        """
        Construye el conjunto de stopwords uniendo NLTK y las personalizadas.

        Carga las stopwords del idioma con nltk.corpus.stopwords.words().
        Si el idioma no está disponible, continúa con conjunto vacío sin
        lanzar excepción. Las custom_stopwords se agregan con unión de conjuntos.
        """
        palabras: set[str] = set()
        if _NLTK_DISPONIBLE:
            try:
                palabras = set(nltk_stopwords.words(language))
            except OSError:
                pass  # si el idioma no está disponible, continuar sin stopwords
        if custom:
            palabras |= custom
        return palabras

    @staticmethod
    def _cargar_stemmer(language: str):
        """
        Crea un SnowballStemmer de NLTK para el idioma configurado.

        Idiomas soportados: arabic, danish, dutch, english, finnish, french,
        german, hungarian, italian, norwegian, portuguese, romanian, russian,
        spanish, swedish.
        Retorna None si NLTK no está instalado o el idioma no es soportado;
        en ese caso _preprocesar() omite el paso de stemming silenciosamente.
        """
        if not _NLTK_DISPONIBLE:
            return None
        try:
            return SnowballStemmer(language)
        except Exception:
            return None
from abc import ABC, abstractmethod
from typing import Any


class BaseExtractor(ABC):
    """Clase base abstracta para los extractores TF-IDF, SIFT y MFCC."""

    @abstractmethod
    def fit(self, data: list) -> "BaseExtractor":
        """
        Aprende los parámetros a partir de *data*.

        Para extractores sin estado (SIFT, MFCC) esto es un no-op que
        solo retorna self, de modo que el pipeline fit → transform funciona
        de forma uniforme.

        input:  data: Entradas crudas (strings de texto, patches de imagen, ventanas de audio).

        output:  self — permite encadenar: extractor.fit(data).transform(data)
        """
        ...

    @abstractmethod
    def transform(self, data: list) -> Any:
        """
        Extrae características de *data*.

        input: data: Mismo formato que el pasado a fit().

        output: Representación de características (numpy array, lista de arrays, etc.).
               El tipo concreto de retorno está documentado en cada subclase.
        """
        ...

    def fit_transform(self, data: list) -> Any:
        """Ajusta con *data* y lo transforma inmediatamente."""
        return self.fit(data).transform(data)

    def extract(self, data: list) -> Any:
        """
        Interfaz unificada de extracción agnóstica a modalidad.

        Cada subclase implementa este método según su modalidad:
          - TFIDFExtractor: recibe lista de dicts {doc_id, chunk_id, text}
                            → lista de dicts con conteos TF crudos
          - SIFTExtractor:  recibe lista de patches de imagen
                            → matriz numpy (N, 128) con todos los descriptores apilados
          - MFCCExtractor:  recibe lista de ventanas de audio
                            → matriz numpy (M, n_mfcc) con todos los frames apilados

        input: data: Entradas en el formato específico de cada modalidad.
        output: Representación de características lista para el módulo Codebook.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} debe implementar el método extract()"
        )

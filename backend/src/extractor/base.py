"""
Clase base para todos los extractores de características.

Todos los extractores siguen la misma interfaz:
  fit(data)       → aprende los parámetros (vocabulario, estadísticas, etc.)
  transform(data) → calcula las representaciones de características
  fit_transform   → atajo conveniente (fit seguido de transform)
"""

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

        Args:
            data: Entradas crudas (strings de texto, patches de imagen, ventanas de audio).

        Returns:
            self — permite encadenar: extractor.fit(data).transform(data)
        """
        ...

    @abstractmethod
    def transform(self, data: list) -> Any:
        """
        Extrae características de *data*.

        Args:
            data: Mismo formato que el pasado a fit().

        Returns:
            Representación de características (numpy array, lista de arrays, etc.).
            El tipo concreto de retorno está documentado en cada subclase.
        """
        ...

    def fit_transform(self, data: list) -> Any:
        """Ajusta con *data* y lo transforma inmediatamente."""
        return self.fit(data).transform(data)

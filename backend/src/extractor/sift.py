"""
Extractor de características SIFT para patches de imagen.

SIFT (Scale-Invariant Feature Transform) detecta puntos clave locales y produce
descriptores de 128 dimensiones invariantes a escala y rotación. Estos
descriptores luego son cuantizados en palabras visuales por el módulo Codebook.

Pipeline
--------
  fit(patches)       → no-op (SIFT no tiene parámetros entrenables)
  transform(patches) → lista de arrays de descriptores, uno por patch

Forma de salida
---------------
  transform() retorna una lista de numpy arrays. El array para el patch i tiene
  forma (n_keypoints_i, 128) con dtype float32. Si no se encuentran puntos clave
  en un patch (ej. región en blanco), se retorna un array vacío de forma (0, 128)
  para que el código posterior siempre espere la misma estructura.

Dependencias
------------
  opencv-contrib-python >= 4.5  (provee cv2.SIFT_create)
"""

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-contrib-python is required for SIFT extraction. "
        "Install it with: pip install opencv-contrib-python"
    ) from exc

from .base import BaseExtractor


class SIFTExtractor(BaseExtractor):
    """Extrae descriptores SIFT de una lista de patches de imagen."""

    DESCRIPTOR_DIM = 128  # fijo por el algoritmo SIFT

    def __init__(
        self,
        n_features: int = 0,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        sigma: float = 1.6,
    ) -> None:
        """
        Args:
            n_features: Número máximo de puntos clave a conservar por patch.
                0 significa sin límite (conservar todos).
            n_octave_layers: Número de capas por octava en la pirámide gaussiana.
            contrast_threshold: Filtra puntos clave de bajo contraste.
            edge_threshold: Filtra puntos clave similares a bordes.
            sigma: Sigma de la gaussiana aplicada a la imagen de entrada.
        """
        self._sift = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma,
        )


    # Interfaz BaseExtractor

    def fit(self, data: list) -> "SIFTExtractor":
        """No-op — SIFT no tiene parámetros entrenables. Retorna self."""
        return self

    def transform(self, patches: list[np.ndarray]) -> list[np.ndarray]:
        """
        Extrae descriptores SIFT de patches de imagen.

        Args:
            patches: Lista de numpy arrays. Cada array puede ser:
                - Escala de grises: forma (H, W) o (H, W, 1), dtype uint8
                - Color BGR: forma (H, W, 3), dtype uint8

        Returns:
            Lista de matrices de descriptores. El item i tiene forma
            ``(n_keypoints_i, 128)`` con dtype float32.
            Retorna forma ``(0, 128)`` para patches sin puntos clave.
        """
        result: list[np.ndarray] = []

        for patch in patches:
            gray = self._to_gray(patch)
            _, descriptors = self._sift.detectAndCompute(gray, None)

            if descriptors is None:
                descriptors = np.empty((0, self.DESCRIPTOR_DIM), dtype=np.float32)
            else:
                descriptors = descriptors.astype(np.float32)

            result.append(descriptors)

        return result


    # Auxiliares
   

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convierte la imagen a escala de grises de 8 bits si es necesario."""
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3 and image.shape[2] == 1:
            return image[:, :, 0]
        return image

    def keypoints_and_descriptors(
        self, patch: np.ndarray
    ) -> tuple[list, np.ndarray]:
        """
        Retorna los puntos clave OpenCV crudos *y* los descriptores para un solo patch.

        Útil para visualización y depuración.

        Args:
            patch: Array de una sola imagen.

        Returns:
            (keypoints, descriptors) donde descriptors tiene forma (N, 128).
        """
        gray = self._to_gray(patch)
        kps, desc = self._sift.detectAndCompute(gray, None)
        if desc is None:
            desc = np.empty((0, self.DESCRIPTOR_DIM), dtype=np.float32)
        return kps, desc.astype(np.float32)

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-contrib-python es requerido para la extracción SIFT. "
        "Instálalo con: pip install opencv-contrib-python"
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
        input:
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
 Extrae descriptores SIFT de una lista de patches (uno por patch),
   manteniendo su correspondencia.
    Proceso por patch:
        1. Convierte a escala de grises si es BGR o tiene canal extra
        2. Ejecuta detectAndCompute() de OpenCV → detecta keypoints y
           calcula sus descriptores de 128 dimensiones
        3. Si el patch no tiene keypoints (región plana/en blanco),
           retorna array vacío (0, 128) en lugar de None para que el
           código posterior siempre reciba la misma estructura
        -Entrada: Lista de patches (Gris o BGR).
        -Salida: Lista de matrices de descriptores, cada una con forma
        (n_keypoints, 128).
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

    def extract(self, patches: list[np.ndarray]) -> np.ndarray:
        """
        Extrae y apila todos los descriptores SIFT en una sola matriz,
        formato requerido por K-Means para construir el vocabulario visual (Codebook).

        Flujo interno:
        1.Llama a transform(patches) -> obtiene lista de arrays (n_ki, 128).
        2.Filtra los arrays vacíos (patches sin keypoints).
        3.Apila verticalmente con np.vstack() -> matriz final (N, 128).
        -Entrada: Lista de patches (imágenes en BGR o gris).
        -Salida: Matriz única de forma (N, 128), donde N es la suma total de
        keypoints de la colección. Si no hay keypoints, retorna (0, 128).
      """
        # Obtener descriptores por patch (lista de arrays)
        descriptors_por_patch = self.transform(patches)

        # Conservar solo los patches que tuvieron keypoints
        no_vacios = [d for d in descriptors_por_patch if d.shape[0] > 0]

        if not no_vacios:
            return np.empty((0, self.DESCRIPTOR_DIM), dtype=np.float32)

        # Apilar todos en una sola matriz N × 128
        return np.vstack(no_vacios).astype(np.float32)


    # Auxiliares

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """
        Convierte la imagen a escala de grises de 8 bits si es necesario.

        SIFT opera sobre imágenes en escala de grises. Esta función acepta
        BGR (3 canales), canal único con dim extra (H,W,1), o ya en gris (H,W)
        y normaliza el formato para detectAndCompute().
        """
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3 and image.shape[2] == 1:
            return image[:, :, 0]
        return image

    def keypoints_and_descriptors(
        self, patch: np.ndarray
    ) -> tuple[list, np.ndarray]:
        """
        Retorna los objetos KeyPoint de OpenCV y los descriptores de un solo patch.
        Útil para visualización y depuración: los KeyPoint contienen
        posición, escala y orientación de cada punto, permitiendo dibujarlos
          con cv2.drawKeypoints().
        - Entrada: Array de una sola imagen (BGR o escala de grises, uint8).
        - Salida: Tupla (keypoints, descriptors), donde descriptors tiene forma (N, 128).
        """
        gray = self._to_gray(patch)
        kps, desc = self._sift.detectAndCompute(gray, None)
        if desc is None:
            desc = np.empty((0, self.DESCRIPTOR_DIM), dtype=np.float32)
        return kps, desc.astype(np.float32)


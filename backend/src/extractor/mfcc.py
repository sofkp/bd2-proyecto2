

import numpy as np

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "librosa es requerido para la extracción MFCC. "
        "Instálalo con: pip install librosa"
    ) from exc

from .base import BaseExtractor


class MFCCExtractor(BaseExtractor):
    """Extrae vectores MFCC de dimensión fija de ventanas de audio."""

    def __init__(
        self,
        n_mfcc: int = 13,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        pre_emphasis: float = 0.97,
    ) -> None:
        """
        Args:
            n_mfcc: Número de coeficientes MFCC a calcular.
                El vector de salida tendrá dimensión 2 * n_mfcc (media + std).
            sample_rate: Frecuencia de muestreo del audio en Hz.
            n_fft: Tamaño de la ventana FFT en muestras.
            hop_length: Desplazamiento entre frames consecutivos en muestras.
                Controla la superposición: hop_length < n_fft genera overlap.
            n_mels: Número de bandas en el banco de filtros Mel.
            pre_emphasis: Coeficiente del filtro de pre-énfasis (0 desactiva).
                Típicamente entre 0.95 y 0.99.
        """
        if n_mfcc < 1:
            raise ValueError("n_mfcc debe ser >= 1")
        if n_mfcc > n_mels:
            raise ValueError("n_mfcc no puede ser mayor que n_mels")
        if sample_rate < 1:
            raise ValueError("sample_rate debe ser >= 1")
        if n_fft < 1:
            raise ValueError("n_fft debe ser >= 1")
        if hop_length < 1:
            raise ValueError("hop_length debe ser >= 1")
        if not (0.0 <= pre_emphasis < 1.0):
            raise ValueError("pre_emphasis debe estar en [0.0, 1.0)")

        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.pre_emphasis = pre_emphasis

        # Dimensión del vector de salida: media + std de cada coeficiente
        self.feature_dim: int = 2 * n_mfcc

    # ------------------------------------------------------------------
    # Interfaz BaseExtractor
    # ------------------------------------------------------------------

    def fit(self, data: list) -> "MFCCExtractor":
        """No-op — MFCC no tiene parámetros entrenables. Retorna self."""
        return self

    def transform(self, windows: list[np.ndarray]) -> list[np.ndarray]:
        """
        Extrae un vector MFCC de dimensión fija por cada ventana de audio.

        Args:
            windows: Lista de numpy arrays de audio. Cada array debe ser:
                - Mono: forma (n_muestras,), dtype float32 o float64.
                - Los valores deben estar normalizados en el rango [-1.0, 1.0].

        Returns:
            Lista de vectores de características. El item i tiene forma
            ``(2 * n_mfcc,)`` con dtype float32: primeros n_mfcc valores
            son las medias, los siguientes n_mfcc son las desviaciones estándar.
            Retorna vector de ceros ``(2 * n_mfcc,)`` para ventanas vacías
            o demasiado cortas para generar al menos un frame.
        """
        resultado: list[np.ndarray] = []

        for ventana in windows:
            vector = self._extraer_vector(ventana)
            resultado.append(vector)

        return resultado

    # ------------------------------------------------------------------
    # Auxiliares
    # ------------------------------------------------------------------

    def _extraer_vector(self, ventana: np.ndarray) -> np.ndarray:
        """
        Procesa una sola ventana de audio y retorna su vector MFCC.

        Args:
            ventana: Array 1D de audio float32/float64.

        Returns:
            Vector de forma (2 * n_mfcc,) con dtype float32.
        """
        vector_nulo = np.zeros(self.feature_dim, dtype=np.float32)

        # Validar que la ventana tenga suficientes muestras para un frame
        if ventana is None or len(ventana) < self.n_fft:
            return vector_nulo

        # Asegurar tipo float32 para librosa
        audio = ventana.astype(np.float32)

        # 1. Pre-énfasis: amplifica frecuencias altas para mejorar la SNR
        if self.pre_emphasis > 0.0:
            audio = np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])

        # 2. Calcular coeficientes MFCC frame a frame
        try:
            mfcc_frames = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
            # mfcc_frames tiene forma (n_mfcc, n_frames)
        except Exception:
            # Si librosa falla (audio demasiado corto, NaN, etc.) retornar ceros
            return vector_nulo

        if mfcc_frames.shape[1] == 0:
            return vector_nulo

        # 3. Agregar a lo largo del eje temporal: media y desviación estándar
        media = mfcc_frames.mean(axis=1)   # forma (n_mfcc,)
        std = mfcc_frames.std(axis=1)      # forma (n_mfcc,)

        # 4. Concatenar en un vector de dimensión fija
        vector = np.concatenate([media, std]).astype(np.float32)
        return vector

    def nombres_caracteristicas(self) -> list[str]:
        """
        Retorna los nombres de las características en orden de índice.

        Los primeros n_mfcc son medias, los siguientes n_mfcc son desviaciones
        estándar. Útil para inspección y depuración.

        Returns:
            Lista de strings, ej. ['mfcc_0_media', ..., 'mfcc_0_std', ...]
        """
        nombres = (
            [f"mfcc_{i}_media" for i in range(self.n_mfcc)]
            + [f"mfcc_{i}_std" for i in range(self.n_mfcc)]
        )
        return nombres

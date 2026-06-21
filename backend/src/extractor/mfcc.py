
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
    """Extrae coeficientes MFCC de ventanas de audio."""

    def __init__(
        self,
        n_mfcc: int = 20,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        pre_emphasis: float = 0.97,
    ) -> None:
        """
        input:
            n_mfcc: Número de coeficientes MFCC por frame.
                - transform() devuelve vectores agregados de dimensión 2 * n_mfcc
                - extract()    devuelve cada frame individual → columnas = n_mfcc
                Default 20 (rango recomendado para K-Means acústico: 13–20).
            sample_rate: Frecuencia de muestreo del audio en Hz.
            n_fft: Tamaño de la ventana FFT en muestras.
                Determina la resolución frecuencial: más grande = más detalle,
                pero requiere más muestras por ventana.
            hop_length: Desplazamiento entre frames consecutivos en muestras.
                hop_length < n_fft genera overlap entre frames consecutivos.
            n_mels: Número de bandas en el banco de filtros Mel.
                Debe ser >= n_mfcc.
            pre_emphasis: Coeficiente del filtro paso-alto aplicado antes de
                la FFT. Amplifica frecuencias altas para mejorar la SNR.
                0 desactiva el filtro. Rango típico: 0.95–0.99.
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

        # Dimensión del vector agregado que devuelve transform(): media + std
        self.feature_dim: int = 2 * n_mfcc


    # Interfaz BaseExtractor

    def fit(self, data: list) -> "MFCCExtractor":
        """No-op — MFCC no tiene parámetros entrenables. Retorna self."""
        return self

    def transform(self, windows: list[np.ndarray]) -> list[np.ndarray]:
        """
        Extrae un vector MFCC de dimensión fija por cada ventana, manteniendo
        su correspondencia original.
        Flujo interno:
        1. Pre-énfasis: Amplifica las frecuencias altas del audio.
        2. Extrae MFCCs con librosa obteniendo una matriz de (n_mfcc, n_frames).
        3. Calcula la media y desviación estándar de cada coeficiente a lo largo de los frames.
        4. Concatena ambos resultados en un solo vector de forma (2 * n_mfcc,).
        - Entrada: Lista de ventanas de audio mono.
        - Salida: Lista de vectores de forma (2 * n_mfcc,).
                    Ventanas inválidas devuelven un vector de ceros.
                """
        return [self._extraer_vector_agregado(v) for v in windows]

    def extract(self, windows: list[np.ndarray]) -> np.ndarray:
        """
        Extrae y apila todos los frames MFCC en una sola matriz para entrenar
        K-Means (Codebook), devolviendo cada frame como una fila individual.
        Flujo interno:
        1. Aplica pre-énfasis a cada ventana.
        2. Calcula MFCCs con librosa y transpone a (n_frames, n_mfcc).
        3. Apila todos los frames con np.vstack() en una matriz (M, n_mfcc).
        -Entrada: Lista de ventanas de audio mono.
        -Salida: Matriz final de forma (M, n_mfcc), donde M es el total de frames.
                Retorna (0, n_mfcc) si está vacía.
                """
        todos_los_frames: list[np.ndarray] = []

        for ventana in windows:
            frames = self._extraer_frames(ventana)
            if frames.shape[0] > 0:
                todos_los_frames.append(frames)

        if not todos_los_frames:
            return np.empty((0, self.n_mfcc), dtype=np.float32)

        # Apilar todos los frames de todas las ventanas → (M, n_mfcc)
        return np.vstack(todos_los_frames).astype(np.float32)


    # Auxiliares

    def _aplicar_pre_enfasis(self, audio: np.ndarray) -> np.ndarray:
        """
        Aplica el filtro de pre-énfasis: y[n] = x[n] - coef * x[n-1].

        Amplifica las frecuencias altas del espectro antes de la FFT, lo que
        mejora la relación señal-ruido (SNR) en bandas de alta frecuencia y
        produce coeficientes MFCC más discriminativos.
        Si pre_emphasis == 0 devuelve el audio sin modificar.
        """
        if self.pre_emphasis > 0.0:
            return np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])
        return audio

    def _extraer_frames(self, ventana: np.ndarray) -> np.ndarray:
        """
        Calcula los frames MFCC de una ventana sin agregar.

        Ejecuta el pipeline completo (pre-énfasis → FFT → banco Mel → log → DCT)
        y devuelve la matriz de frames transpuesta para que cada fila sea un
        vector de coeficientes. Esta salida es la que consume extract().

        input:
            ventana: Array 1D de audio float32/float64.

        output:
            Matriz de forma ``(n_frames, n_mfcc)`` con dtype float32.
            Retorna ``(0, n_mfcc)`` si la ventana es inválida o muy corta.
        """
        vacio = np.empty((0, self.n_mfcc), dtype=np.float32)

        if ventana is None or len(ventana) < self.n_fft:
            return vacio

        audio = self._aplicar_pre_enfasis(ventana.astype(np.float32))

        try:
            # mfcc_frames: forma (n_mfcc, n_frames)
            mfcc_frames = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
        except Exception:
            return vacio

        if mfcc_frames.shape[1] == 0:
            return vacio

        # Transponer → (n_frames, n_mfcc): cada fila es un frame
        return mfcc_frames.T.astype(np.float32)

    def _extraer_vector_agregado(self, ventana: np.ndarray) -> np.ndarray:
        """
        Agrega los frames de una ventana en un vector de dimensión fija.

        Calcula la media y desviación estándar de cada coeficiente a lo largo
        del eje temporal y las concatena: [media_0..media_k, std_0..std_k].
        Esta representación compacta es la que devuelve transform().
        input: ventana: Array 1D de audio float32/float64.
        output:
            Vector de forma ``(2 * n_mfcc,)`` con dtype float32.
            Retorna vector de ceros si la ventana es inválida.
        """
        frames = self._extraer_frames(ventana)  # (n_frames, n_mfcc)

        if frames.shape[0] == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        media = frames.mean(axis=0)  # (n_mfcc,)
        std   = frames.std(axis=0)   # (n_mfcc,)
        return np.concatenate([media, std]).astype(np.float32)

    def nombres_caracteristicas(self) -> list[str]:
        """
        Retorna los nombres de las características del vector agregado (transform).

        Útil para etiquetar columnas en análisis o depuración.
        Los primeros n_mfcc son medias, los siguientes n_mfcc son desviaciones.

        output:
            Lista de strings, ej. ['mfcc_0_media', ..., 'mfcc_19_std']
        """
        return (
            [f"mfcc_{i}_media" for i in range(self.n_mfcc)]
            + [f"mfcc_{i}_std"   for i in range(self.n_mfcc)]
        )


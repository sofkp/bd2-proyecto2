"""
Tests unitarios para MFCCExtractor.

Cubre:
  - __init__(): validaciones de parámetros
  - fit(): no-op, retorna self
  - transform(): forma, dtype, vector agregado media+std, ventana corta
  - extract(): matriz apilada M×n_mfcc, ventana corta, lista vacía
  - _aplicar_pre_enfasis(): longitud conservada, sin énfasis
  - _extraer_frames(): forma (n_frames, n_mfcc), ventana inválida
"""

import numpy as np
import pytest

# Saltar todos los tests si librosa no está instalado
librosa = pytest.importorskip("librosa", reason="librosa no instalado")

from src.extractor.mfcc import MFCCExtractor


# -----------------------------------------------------------------------
# Helpers para generar audio sintético
# -----------------------------------------------------------------------

SR = 22050  # frecuencia de muestreo estándar


def audio_senoidal(duracion_s: float = 1.0, sr: int = SR, freq: float = 440.0) -> np.ndarray:
    """Genera una señal senoidal mono float32 normalizada en [-1, 1]."""
    t = np.linspace(0, duracion_s, int(sr * duracion_s), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def audio_ruido(n_muestras: int = SR) -> np.ndarray:
    """Genera ruido blanco float32 normalizado."""
    return (np.random.randn(n_muestras) * 0.1).astype(np.float32)


def audio_corto(n_muestras: int = 100) -> np.ndarray:
    """Señal demasiado corta para generar un frame (< n_fft)."""
    return np.zeros(n_muestras, dtype=np.float32)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def extractor():
    """MFCCExtractor con n_mfcc=20 y parámetros por defecto."""
    return MFCCExtractor(n_mfcc=20, sample_rate=SR)


@pytest.fixture
def ventanas():
    """Lista de 3 ventanas de audio de 1 segundo cada una."""
    return [audio_senoidal(), audio_ruido(), audio_senoidal(freq=880.0)]


# -----------------------------------------------------------------------
# Tests de __init__() — validaciones
# -----------------------------------------------------------------------

class TestInit:

    def test_n_mfcc_default_es_20(self):
        """El valor por defecto de n_mfcc debe ser 20."""
        e = MFCCExtractor()
        assert e.n_mfcc == 20

    def test_feature_dim_es_dos_por_n_mfcc(self, extractor):
        """feature_dim debe ser exactamente 2 * n_mfcc."""
        assert extractor.feature_dim == 2 * extractor.n_mfcc

    def test_n_mfcc_menor_que_uno_lanza_error(self):
        with pytest.raises(ValueError):
            MFCCExtractor(n_mfcc=0)

    def test_n_mfcc_mayor_que_n_mels_lanza_error(self):
        with pytest.raises(ValueError):
            MFCCExtractor(n_mfcc=200, n_mels=128)

    def test_sample_rate_invalido_lanza_error(self):
        with pytest.raises(ValueError):
            MFCCExtractor(sample_rate=0)

    def test_pre_emphasis_fuera_de_rango_lanza_error(self):
        with pytest.raises(ValueError):
            MFCCExtractor(pre_emphasis=1.0)

    def test_pre_emphasis_cero_valido(self):
        """pre_emphasis=0.0 debe ser válido (desactiva el filtro)."""
        e = MFCCExtractor(pre_emphasis=0.0)
        assert e.pre_emphasis == 0.0


# -----------------------------------------------------------------------
# Tests de fit()
# -----------------------------------------------------------------------

class TestFit:

    def test_fit_retorna_self(self, extractor, ventanas):
        """fit() debe retornar self (es un no-op)."""
        assert extractor.fit(ventanas) is extractor

    def test_fit_no_cambia_parametros(self, extractor, ventanas):
        """fit() no debe modificar n_mfcc ni ningún parámetro del extractor."""
        n_mfcc_antes = extractor.n_mfcc
        extractor.fit(ventanas)
        assert extractor.n_mfcc == n_mfcc_antes


# -----------------------------------------------------------------------
# Tests de transform()
# -----------------------------------------------------------------------

class TestTransform:

    def test_longitud_de_salida(self, extractor, ventanas):
        """transform() debe retornar un vector por cada ventana."""
        resultado = extractor.transform(ventanas)
        assert len(resultado) == len(ventanas)

    def test_forma_vector_2_por_n_mfcc(self, extractor, ventanas):
        """Cada vector de salida debe tener forma (2 * n_mfcc,)."""
        resultado = extractor.transform(ventanas)
        for vec in resultado:
            assert vec.shape == (2 * extractor.n_mfcc,)

    def test_dtype_float32(self, extractor, ventanas):
        """Los vectores de salida deben ser float32."""
        resultado = extractor.transform(ventanas)
        for vec in resultado:
            assert vec.dtype == np.float32

    def test_primeros_n_mfcc_son_medias(self, extractor):
        """
        Los primeros n_mfcc valores del vector deben ser las medias de los
        coeficientes (no todos cero para audio con contenido).
        """
        ventana = [audio_senoidal()]
        vec = extractor.transform(ventana)[0]
        medias = vec[:extractor.n_mfcc]
        # Para señal senoidal con contenido, las medias no deben ser todas cero
        assert not np.all(medias == 0)

    def test_ventana_corta_retorna_ceros(self, extractor):
        """Una ventana más corta que n_fft debe retornar vector de ceros."""
        vec = extractor.transform([audio_corto()])[0]
        assert np.all(vec == 0)

    def test_lista_vacia_retorna_lista_vacia(self, extractor):
        """transform([]) debe retornar lista vacía."""
        assert extractor.transform([]) == []


# -----------------------------------------------------------------------
# Tests de extract()
# -----------------------------------------------------------------------

class TestExtract:

    def test_retorna_numpy_array(self, extractor, ventanas):
        """extract() debe retornar un numpy.ndarray."""
        resultado = extractor.extract(ventanas)
        assert isinstance(resultado, np.ndarray)

    def test_forma_m_por_n_mfcc(self, extractor, ventanas):
        """La matriz debe tener exactamente n_mfcc columnas."""
        resultado = extractor.extract(ventanas)
        assert resultado.ndim == 2
        assert resultado.shape[1] == extractor.n_mfcc

    def test_dtype_float32(self, extractor, ventanas):
        """La matriz apilada debe ser float32."""
        resultado = extractor.extract(ventanas)
        assert resultado.dtype == np.float32

    def test_filas_es_suma_de_frames(self, extractor, ventanas):
        """
        El número de filas de extract() debe ser igual a la suma de frames
        de todas las ventanas (lo que _extraer_frames reporta individualmente).
        """
        n_total = sum(
            extractor._extraer_frames(v).shape[0] for v in ventanas
        )
        resultado = extractor.extract(ventanas)
        assert resultado.shape[0] == n_total

    def test_ventana_corta_excluida(self, extractor):
        """
        Una ventana demasiado corta no debe contribuir filas a la matriz.
        """
        frames_normal = extractor._extraer_frames(audio_senoidal()).shape[0]
        resultado = extractor.extract([audio_senoidal(), audio_corto()])
        # Solo la ventana normal aportó frames
        assert resultado.shape[0] == frames_normal

    def test_lista_vacia_retorna_vacio(self, extractor):
        """extract([]) debe retornar array vacío (0, n_mfcc)."""
        resultado = extractor.extract([])
        assert resultado.shape == (0, extractor.n_mfcc)

    def test_todas_ventanas_cortas_retorna_vacio(self, extractor):
        """Si todas las ventanas son cortas, retorna (0, n_mfcc)."""
        resultado = extractor.extract([audio_corto(), audio_corto()])
        assert resultado.shape == (0, extractor.n_mfcc)


# -----------------------------------------------------------------------
# Tests de _aplicar_pre_enfasis()
# -----------------------------------------------------------------------

class TestPreEnfasis:

    def test_conserva_longitud(self, extractor):
        """El audio filtrado debe tener la misma longitud que el original."""
        audio = audio_senoidal()
        filtrado = extractor._aplicar_pre_enfasis(audio)
        assert len(filtrado) == len(audio)

    def test_sin_enfasis_no_modifica(self):
        """Con pre_emphasis=0.0 el audio debe retornarse sin cambios."""
        e = MFCCExtractor(pre_emphasis=0.0)
        audio = audio_senoidal()
        resultado = e._aplicar_pre_enfasis(audio)
        np.testing.assert_array_equal(resultado, audio)

    def test_con_enfasis_modifica_audio(self, extractor):
        """Con pre_emphasis > 0, el audio filtrado debe diferir del original."""
        audio = audio_senoidal()
        filtrado = extractor._aplicar_pre_enfasis(audio)
        assert not np.allclose(audio, filtrado)


# -----------------------------------------------------------------------
# Tests de _extraer_frames()
# -----------------------------------------------------------------------

class TestExtraerFrames:

    def test_forma_n_frames_por_n_mfcc(self, extractor):
        """Los frames deben tener forma (n_frames, n_mfcc)."""
        frames = extractor._extraer_frames(audio_senoidal())
        assert frames.ndim == 2
        assert frames.shape[1] == extractor.n_mfcc

    def test_ventana_corta_retorna_vacio(self, extractor):
        """Una ventana muy corta debe retornar (0, n_mfcc)."""
        frames = extractor._extraer_frames(audio_corto())
        assert frames.shape == (0, extractor.n_mfcc)

    def test_none_retorna_vacio(self, extractor):
        """Pasar None debe retornar (0, n_mfcc) sin lanzar excepción."""
        frames = extractor._extraer_frames(None)
        assert frames.shape == (0, extractor.n_mfcc)

    def test_dtype_float32(self, extractor):
        """Los frames deben ser float32."""
        frames = extractor._extraer_frames(audio_senoidal())
        assert frames.dtype == np.float32

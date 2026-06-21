"""
Tests unitarios para SIFTExtractor.

Cubre:
  - fit(): no-op, retorna self
  - transform(): forma de salida, dtype, patches sin keypoints
  - extract(): matriz apilada N×128, patches sin keypoints, lista vacía
  - _to_gray(): conversión BGR→gris, canal extra, ya en gris
  - keypoints_and_descriptors(): estructura de retorno
"""

import numpy as np
import pytest

# Saltar todos los tests si OpenCV no está instalado
cv2 = pytest.importorskip("cv2", reason="opencv-contrib-python no instalado")

from src.extractor.sift import SIFTExtractor


# -----------------------------------------------------------------------
# Helpers para generar imágenes sintéticas
# -----------------------------------------------------------------------

def imagen_con_textura(alto: int = 64, ancho: int = 64) -> np.ndarray:
    """
    Genera una imagen BGR uint8 con textura suficiente para detectar keypoints.
    Usa un patrón de tablero de ajedrez que garantiza bordes detectables.
    """
    img = np.zeros((alto, ancho, 3), dtype=np.uint8)
    tam_celda = 8
    for i in range(0, alto, tam_celda):
        for j in range(0, ancho, tam_celda):
            if (i // tam_celda + j // tam_celda) % 2 == 0:
                img[i:i+tam_celda, j:j+tam_celda] = 255
    return img


def imagen_uniforme(alto: int = 64, ancho: int = 64) -> np.ndarray:
    """Imagen completamente gris — no tiene keypoints detectables."""
    return np.full((alto, ancho, 3), 128, dtype=np.uint8)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def extractor():
    """SIFTExtractor con configuración por defecto."""
    return SIFTExtractor()


@pytest.fixture
def patches_con_textura():
    """Lista de 3 patches con textura suficiente para keypoints."""
    return [imagen_con_textura() for _ in range(3)]


@pytest.fixture
def patches_uniformes():
    """Lista de 2 patches sin textura — no producen keypoints."""
    return [imagen_uniforme() for _ in range(2)]


# -----------------------------------------------------------------------
# Tests de fit()
# -----------------------------------------------------------------------

class TestFit:

    def test_fit_retorna_self(self, extractor, patches_con_textura):
        """fit() debe retornar self (es un no-op)."""
        assert extractor.fit(patches_con_textura) is extractor

    def test_fit_no_modifica_estado(self, extractor, patches_con_textura):
        """fit() no debe modificar ningún atributo del extractor."""
        extractor.fit(patches_con_textura)
        # No hay atributos entrenables que verificar; solo confirmar que no lanza
        assert True


# -----------------------------------------------------------------------
# Tests de transform()
# -----------------------------------------------------------------------

class TestTransform:

    def test_longitud_de_salida(self, extractor, patches_con_textura):
        """transform() debe retornar una lista con un array por patch."""
        resultado = extractor.transform(patches_con_textura)
        assert len(resultado) == len(patches_con_textura)

    def test_descriptores_tienen_128_columnas(self, extractor, patches_con_textura):
        """Cada array de descriptores debe tener exactamente 128 columnas."""
        resultado = extractor.transform(patches_con_textura)
        for desc in resultado:
            assert desc.shape[1] == 128

    def test_dtype_float32(self, extractor, patches_con_textura):
        """Los descriptores deben ser float32."""
        resultado = extractor.transform(patches_con_textura)
        for desc in resultado:
            assert desc.dtype == np.float32

    def test_patch_sin_keypoints_retorna_array_vacio(self, extractor, patches_uniformes):
        """Un patch sin keypoints debe retornar array (0, 128), no None."""
        resultado = extractor.transform(patches_uniformes)
        for desc in resultado:
            assert desc.shape == (0, 128)

    def test_acepta_imagen_en_escala_de_grises(self, extractor):
        """transform() debe aceptar imágenes en escala de grises (H, W)."""
        gris = cv2.cvtColor(imagen_con_textura(), cv2.COLOR_BGR2GRAY)
        resultado = extractor.transform([gris])
        assert len(resultado) == 1
        assert resultado[0].shape[1] == 128

    def test_acepta_imagen_bgr(self, extractor, patches_con_textura):
        """transform() debe aceptar imágenes BGR (H, W, 3)."""
        resultado = extractor.transform(patches_con_textura)
        assert len(resultado) == len(patches_con_textura)


# -----------------------------------------------------------------------
# Tests de extract()
# -----------------------------------------------------------------------

class TestExtract:

    def test_retorna_numpy_array(self, extractor, patches_con_textura):
        """extract() debe retornar un numpy.ndarray."""
        resultado = extractor.extract(patches_con_textura)
        assert isinstance(resultado, np.ndarray)

    def test_forma_n_por_128(self, extractor, patches_con_textura):
        """La matriz debe tener exactamente 128 columnas."""
        resultado = extractor.extract(patches_con_textura)
        assert resultado.ndim == 2
        assert resultado.shape[1] == 128

    def test_dtype_float32(self, extractor, patches_con_textura):
        """La matriz apilada debe ser float32."""
        resultado = extractor.extract(patches_con_textura)
        assert resultado.dtype == np.float32

    def test_filas_es_suma_de_keypoints(self, extractor, patches_con_textura):
        """
        El número de filas de extract() debe ser igual a la suma de
        keypoints de todos los patches (lo que transform() reporta).
        """
        descriptores_por_patch = extractor.transform(patches_con_textura)
        n_total = sum(d.shape[0] for d in descriptores_por_patch)
        matriz = extractor.extract(patches_con_textura)
        assert matriz.shape[0] == n_total

    def test_patches_sin_keypoints_retorna_vacio(self, extractor, patches_uniformes):
        """Si ningún patch tiene keypoints, retorna array vacío (0, 128)."""
        resultado = extractor.extract(patches_uniformes)
        assert resultado.shape == (0, 128)

    def test_lista_vacia_retorna_vacio(self, extractor):
        """extract([]) debe retornar array vacío (0, 128)."""
        resultado = extractor.extract([])
        assert resultado.shape == (0, 128)

    def test_mezcla_con_y_sin_keypoints(self, extractor):
        """
        Una lista con patches con y sin keypoints debe apilar solo los
        descriptores de los patches que sí tienen keypoints.
        """
        patches = [imagen_con_textura(), imagen_uniforme(), imagen_con_textura()]
        desc_por_patch = extractor.transform(patches)
        n_esperado = sum(d.shape[0] for d in desc_por_patch)
        resultado = extractor.extract(patches)
        assert resultado.shape[0] == n_esperado


# -----------------------------------------------------------------------
# Tests de _to_gray()
# -----------------------------------------------------------------------

class TestToGray:

    def test_bgr_a_gris(self):
        """Una imagen BGR (H,W,3) debe convertirse a (H,W)."""
        bgr = imagen_con_textura()
        gris = SIFTExtractor._to_gray(bgr)
        assert gris.ndim == 2

    def test_ya_en_gris_sin_cambio(self):
        """Una imagen (H,W) ya en gris debe retornarse sin modificar."""
        gris = np.zeros((32, 32), dtype=np.uint8)
        resultado = SIFTExtractor._to_gray(gris)
        assert resultado.shape == (32, 32)

    def test_canal_extra_hwc1(self):
        """Una imagen (H,W,1) debe retornar (H,W) eliminando la dimensión extra."""
        img = np.zeros((32, 32, 1), dtype=np.uint8)
        resultado = SIFTExtractor._to_gray(img)
        assert resultado.shape == (32, 32)


# -----------------------------------------------------------------------
# Tests de keypoints_and_descriptors()
# -----------------------------------------------------------------------

class TestKeypointsAndDescriptors:

    def test_retorna_tupla(self, extractor):
        """Debe retornar una tupla (keypoints, descriptors)."""
        resultado = extractor.keypoints_and_descriptors(imagen_con_textura())
        assert isinstance(resultado, tuple)
        assert len(resultado) == 2

    def test_descriptors_128_cols(self, extractor):
        """Los descriptores deben tener 128 columnas."""
        _, desc = extractor.keypoints_and_descriptors(imagen_con_textura())
        assert desc.shape[1] == 128

    def test_patch_sin_keypoints_desc_vacio(self, extractor):
        """Un patch sin keypoints debe retornar descriptores (0, 128)."""
        _, desc = extractor.keypoints_and_descriptors(imagen_uniforme())
        assert desc.shape == (0, 128)

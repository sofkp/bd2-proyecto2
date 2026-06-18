import pytest
import numpy as np
from src.codebook import VectorCodebook

def test_vector_codebook_kmeans_shapes():
    # 1. Preparar datos simulados de descriptores locales (100 descriptores de 128 dimensiones)
    np.random.seed(42)
    mockup_sift_features = np.random.rand(100, 128)
    
    # 2. Inicializar VectorCodebook con 5 clusters
    n_clusters = 5
    visual_cb = VectorCodebook(n_clusters=n_clusters)
    visual_centroids = visual_cb.build_codebook(mockup_sift_features)

    # 3. Verificaciones (asserts)
    # Los centroides deben tener la forma (n_clusters, n_features)
    assert visual_centroids.shape == (n_clusters, 128)
    
    # Verificar que los datos no estén vacíos o sean nulos
    assert not np.isnan(visual_centroids).any()

def test_vector_codebook_kmeans_audio():
    # 1. Preparar datos de audio (50 ventanas de 13 coeficientes MFCC)
    np.random.seed(42)
    mockup_mfcc_features = np.random.rand(50, 13)
    
    # 2. Inicializar VectorCodebook con 3 clusters
    n_clusters = 3
    audio_cb = VectorCodebook(n_clusters=n_clusters)
    audio_centroids = audio_cb.build_codebook(mockup_mfcc_features)

    # 3. Verificaciones (asserts)
    assert audio_centroids.shape == (n_clusters, 13)
    assert not np.isnan(audio_centroids).any()

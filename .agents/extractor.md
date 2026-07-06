# Extractor — Contexto para IA

> Integrante: KarolayTamayoH
> Rama: feat/karolay-extractor
> Lee primero AGENTS.md antes de continuar

## 1. Qué implementar
- **Texto**: TF-IDF (tokenización + stopwords + stemming + vectorización)
- **Imagen**: SIFT (detección y descripción de keypoints)
- **Audio**: MFCC (coeficientes cepstrales en frecuencia Mel)
- **Evaluación**: comparativas experimentales GIN vs índice invertido y pgvector vs codebook

## 2. Input que recibes (viene de softkp — Split)
```python
List[dict] = [
    {
        "chunk_id": str,
        "content": Any,        # texto: str | imagen: np.array | audio: np.array
        "modality": str,       # "text" | "image" | "audio"
        "metadata": dict
    }
]
```

## 3. Output que debes devolver (va a LuixRom — Codebook)
```python
List[dict] = [
    {
        "chunk_id": str,
        "modality": str,
        "features": np.ndarray
        # texto: vector TF-IDF (ya tokenizado + sin stopwords + stemmed)
        # imagen: descriptores SIFT por patch (N x 128)
        # audio: matriz MFCC por ventana (N x n_mfcc)
    }
]
```

## 4. Algoritmos a implementar

### Texto — TF-IDF
```python
# Recibes: chunk de texto como str de softkp
# Tu tarea:
# 1. Tokenizar el texto
# 2. Convertir a minúsculas y eliminar puntuación
# 3. Eliminar stopwords
# 4. Aplicar stemming o lemmatización
# 5. Calcular TF-IDF sobre la colección completa
# 6. Devolver vector TF-IDF por chunk
```

### Imagen — SIFT
```python
# Recibes: patch de imagen como np.ndarray de softkp
# Tu tarea:
# 1. Convertir imagen a escala de grises
# 2. Detectar keypoints con SIFT
# 3. Calcular descriptores de 128 dimensiones por keypoint
# 4. Devolver matriz de descriptores (N x 128) por patch
```

### Audio — MFCC
```python
# Recibes: ventana de audio como np.ndarray de softkp
# Tu tarea:
# 1. Aplicar pre-énfasis a la señal
# 2. Calcular coeficientes MFCC (default: 13 coeficientes)
# 3. Normalizar los coeficientes
# 4. Devolver matriz MFCC por ventana
```

### Evaluación experimental + Comparativas
```python
# Tu tarea:
# 1. GIN/GiST vs índice invertido (hanksvi):
#    - Ejecutar mismas consultas en ambos
#    - Medir latencia, memoria, precisión
# 2. pgvector vs codebook+histogramas (LuixRom):
#    - Ejecutar mismas consultas en ambos
#    - Medir latencia, precisión, memoria
# 3. Cargas: 1K, 10K, 100K chunks
# 4. Generar gráficos comparativos
# 5. Documentar trade-offs y conclusiones
```

## 5. Archivos a crear
```plaintext
backend/src/extractor/
├── __init__.py
├── text_extractor.py      # TF-IDF
├── image_extractor.py     # SIFT
└── audio_extractor.py     # MFCC

backend/tests/extractor/
├── __init__.py
├── test_text_extractor.py
├── test_image_extractor.py
└── test_audio_extractor.py

experiments/
├── gin_vs_index.py        # Comparativa texto
├── pgvector_vs_codebook.py # Comparativa imagen/audio
└── results/               # Gráficos y tablas
```

## 6. Librerías recomendadas
- `nltk` → tokenización, stopwords, stemming
- `scikit-learn` → TF-IDF (TfidfVectorizer)
- `opencv-python` → SIFT
- `librosa` → MFCC
- `numpy` → operaciones matriciales
- `matplotlib` → gráficos comparativos
- `psycopg2` → consultas PostgreSQL para comparativas

## 7. Parámetros importantes
- TF-IDF: `max_features` configurable (default 5120)
- SIFT: `nfeatures` configurable (default 512)
- MFCC: `n_mfcc` configurable (default 13)
- Comparativas: cargas 1K, 10K, 100K chunks
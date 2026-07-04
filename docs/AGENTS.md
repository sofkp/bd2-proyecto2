# AGENTS.md — Contexto compartido

> Lee este archivo completo antes de usar cualquier IA.
> Luego lee tu archivo específico en `.agents/tu-modulo.md`

## 1. Qué es el proyecto
Sistema multimodal de recuperación y búsqueda que indexa y busca
contenido de texto, imágenes y audio bajo una arquitectura unificada:
Split → Extractor → Codebook → Índice Invertido.

## 2. Stack tecnológico
- Backend: Python + FastAPI
- Frontend: Next.js (TypeScript + Tailwind)
- Base de datos: PostgreSQL + pgvector
- Contenedores: Docker Compose
- Tests: pytest

## 3. Estructura de carpetas
```plaintext
bd2-proyecto2/
├── frontend/                  # Next.js — LuixRom
├── backend/
│   ├── src/
│   │   ├── split/             # softkp
│   │   ├── extractor/         # KarolayTamayoH
│   │   ├── codebook/          # LuixRom
│   │   └── index/             # hanksvi
│   ├── api/                   # FastAPI endpoints
│   └── tests/                 # Tests por módulo
├── db/                        # PostgreSQL schemas
├── experiments/               # Benchmarks y comparativas
├── .agents/                   # Contexto por módulo para IA
├── docker-compose.yml         # softkp
├── init.sh                    # Inicializar proyecto
├── run_tests.sh               # Correr todos los tests
├── AGENTS.md                  # Este archivo
├── CONSTITUTION.md            # Reglas del equipo
└── FLUJO.md                   # Guía de trabajo
```

## 4. Contratos entre módulos

### Split → Extractor
```python
# Split devuelve
List[dict] = [
    {
        "chunk_id": str,
        "content": Any,      # texto: str | imagen: np.array | audio: np.array
        "modality": str,     # "text" | "image" | "audio"
        "metadata": dict
    }
]
```

### Extractor → Codebook
```python
# Extractor devuelve
List[dict] = [
    {
        "chunk_id": str,
        "modality": str,
        "features": np.ndarray  # TF-IDF vector | SIFT descriptors | MFCC matrix
    }
]
```

### Codebook → Retrieval
```python
# Codebook devuelve
List[dict] = [
    {
        "chunk_id": str,
        "modality": str,
        "histogram": np.ndarray  # vector de frecuencias de codewords
    }
]
```

### Retrieval → API
```python
# Retrieval devuelve
List[dict] = [
    {
        "chunk_id": str,
        "score": float,
        "metadata": dict
    }
]
```

## 5. Librerías recomendadas
> Puedes usar cualquier librería adicional siempre que
> respetes los contratos de salida definidos en la sección 4.

- numpy, scipy, scikit-learn
- nltk (texto)
- opencv-python (imagen)
- librosa (audio)
- sqlalchemy (PostgreSQL)
- fastapi, uvicorn (API)

## 6. Convenciones de código
- Funciones en snake_case: `build_codebook()`, `extract_features()`
- Clases en PascalCase: `TextSplitter`, `SIFTExtractor`
- Archivos máximo 100 líneas
- Docstring en cada función pública
- Type hints obligatorios

## 7. Índice de archivos por módulo
| Módulo | Integrante | Archivo de contexto |
|---|---|---|
| Split | softkp | `.agents/split.md` |
| Extractor | KarolayTamayoH | `.agents/extractor.md` |
| Codebook | LuixRom | `.agents/codebook.md` |
| Retrieval | hanksvi | `.agents/retrieval.md` |


## 8. Datasets disponibles

Los datos NO están en el repo. Cada integrante los descarga localmente con
`./scripts/download_data.sh <nombre>`. No commitear nada de `data/full/`.

| Nombre | Modalidad principal | Cómo descargarlo |
|---|---|---|
| AG News | Texto | `./scripts/download_data.sh agnews` |
| FMA 100K WAV | Audio | `./scripts/download_data.sh fma-audio` o `docker compose --profile datasets run --rm datasets` |
| Fashion200K | Imagen | `./scripts/download_data.sh fashion200k` |

### Estructura local esperada (generada por el script)
```plaintext
data/
├── samples/   # en git — muestra chica para tests
└── full/      # en .gitignore — datos reales, solo local
    ├── agnews/
    ├── fashion200k/
    ├── .kaggle_cache/
    └── (kaggle datasets aquí)
```

### Para los módulos
- **softkp (split):** los tres datasets como input
- **KarolayTamayoH (extractor):** idem
- **LuixRom (codebook):** usa output del extractor, no lee datasets directo
- **hanksvi (retrieval):** usa output del codebook, no lee datasets directo

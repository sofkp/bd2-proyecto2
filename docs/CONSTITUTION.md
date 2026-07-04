# Constitution — BD2 Proyecto 2

## 1. Ramas
- `main` → solo código estable y revisado
- `dev` → rama de integración
- `feat/<modulo>-<descripcion>` → nuevas funcionalidades
- `fix/<modulo>-<descripcion>` → correcciones

## 2. Formato de commits
[modulo] tipo: descripcion corta
**Módulos:** `init`, `split`, `extractor`, `codebook`, `index`, `api`, `frontend`, `db`, `experiments`, `docs`

**Tipos:** `feat`, `fix`, `test`, `refactor`, `docs`, `chore`

**Ejemplos:**
- `[split] feat: implement text chunking by paragraph`
- `[extractor] feat: add TF-IDF vectorizer`
- `[frontend] feat: add image search UI`
- `[index] fix: fix SPIMI memory overflow`

## 3. Pull Requests
- Todo merge a `dev` requiere al menos 1 review
- El PR debe referenciar su issue: `Closes #12`
- No hacer merge de tu propio PR

## 4. Issues
- Cada tarea tiene su issue en GitHub Projects
- Asignarse el issue antes de empezar a codear
- Cerrar el issue con el PR correspondiente

markdown## 5. Responsabilidades
| Integrante | Módulo principal | Detalle |
|---|---|---|
| LuixRom | Codebook + Frontend | Top-K palabras + K-Means visual + K-Means acústico + UI Next.js |
| hanksvi | Retrieval e Índices + Apps | SPIMI + índice invertido + histogramas visuales y acústicos + App 1 + App 2 |
| KarolayTamayoH | Feature Extraction + Evaluación | TF-IDF + SIFT + MFCC + Evaluación experimental + Comparativas GIN/pgvector |
| softkp | Split + Infra | División texto/imagen/audio (párrafos, patches, sliding windows) + Docker + PostgreSQL |

## 6. Reglas generales
- Archivos máximo 100 líneas (excepto configuración)
- No pushear directo a `main` ni a `dev`
- Correr `run_tests.sh` antes de cada PR
- Un integrante = un módulo = commits verificables

## 7. Datasets
Los datasets NO se versionan en git. Se descargan localmente con:

```bash
./scripts/download_data.sh agnews          # texto
./scripts/download_data.sh fma-audio       # audio crudo WAV para MFCC
./scripts/download_data.sh fashion200k     # imagen de ropa
```

Requiere token de Kaggle una vez: Kaggle → Settings → API → Create New Token → guardar en `~/.kaggle/kaggle.json`

| Dataset | Modalidad | Fuente |
|---|---|---|
| AG News | Texto | HuggingFace: `fancyzhx/ag_news` |
| FMA 100K WAV | Audio | Kaggle: `noahbadoa/fma-dataset-100k-music-wav-files` |
| Fashion200K | Imagen | HuggingFace: `Marqo/fashion200k` |

`data/full/` está en `.gitignore`. Solo `data/samples/` se versiona.

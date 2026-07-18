# Informe Técnico Final

## Sistema Multimodal De Recuperación Y Búsqueda

Este documento describe la versión final del proyecto. Para una presentación más breve del repositorio, instrucciones de ejecución y resumen general, revisar el [README principal](../README.md).

## 1. Objetivo

El objetivo del proyecto fue construir un sistema de recuperación de información capaz de trabajar con tres modalidades: texto, imágenes y audio. En lugar de implementar un motor totalmente distinto para cada tipo de dato, se diseñó un flujo común basado en:

```text
Preprocesamiento -> Extractor -> Codebook -> Índice -> Ranking
```

Cada modalidad usa extractores propios, pero todas terminan en una representación comparable mediante histogramas o vectores de codewords. Sobre esas representaciones se realizan búsquedas por similitud y se comparan dos enfoques:

- una implementación propia en memoria;
- una implementación persistente usando PostgreSQL, GIN y pgvector/HNSW.

## 2. Arquitectura General

```mermaid
flowchart TB
    U["Usuario / Frontend Next.js"] --> API["Backend FastAPI"]
    API --> P["Pipeline multimodal"]
    P --> S["Preprocesamiento / Split"]
    S --> E["Extracción de características"]
    E --> C["Codebook / histograma"]
    C --> I["Índices propios"]
    C --> DB["PostgreSQL"]
    DB --> GIN["GIN / tsvector"]
    DB --> PGV["pgvector / HNSW"]
    I --> R["Ranking Top-k"]
    GIN --> R
    PGV --> R
    R --> API
    API --> U
```

El frontend permite seleccionar modalidad, enfoque de búsqueda y número de resultados. La API en FastAPI carga los pipelines, construye las representaciones de consulta y ejecuta la búsqueda en el índice propio o en PostgreSQL.

## 3. Flujo Por Modalidad

```mermaid
flowchart TB
    T["Texto / PDF"] --> ST["SplitText"]
    ST --> ET["TFIDFExtractor"]
    ET --> CT["CodebookText"]
    CT --> IT["InvertedIndex + SPIMI"]
    CT --> PT["PostgreSQL GIN"]

    IMG["Imagen"] --> EI["SIFTExtractor sobre imagen completa"]
    EI --> CI["K-Means: palabras visuales"]
    CI --> HI["Histograma SIFT + HSV"]
    HI --> II["VisualSearchIndex"]
    HI --> PI["pgvector HNSW"]

    A["Audio"] --> SA["SplitAudio"]
    SA --> EA["MFCCExtractor"]
    EA --> CA["MiniBatchKMeans: palabras acústicas"]
    CA --> IA["AudioSearchIndex"]
    CA --> PA["pgvector HNSW"]
```

## 4. Modalidad Textual

### 4.1. Preprocesamiento

El módulo `SplitText` divide documentos en fragmentos. También soporta archivos PDF mediante PyMuPDF, lo que permite buscar a partir del contenido extraído de un documento.

Archivo principal:

- `backend/src/split/split_text.py`

### 4.2. Extracción

La extracción textual se realiza con `TFIDFExtractor`. El flujo aplica:

- conversión a minúsculas;
- limpieza de caracteres;
- tokenización con NLTK;
- eliminación de stopwords;
- stemming con SnowballStemmer;
- cálculo de frecuencias de término.

Archivo principal:

- `backend/src/extractor/tfidf.py`

### 4.3. Codebook Textual

`CodebookText` selecciona los términos más representativos y conserva pesos IDF. Con esto cada fragmento textual se transforma en un histograma TF-IDF.

Archivo principal:

- `backend/src/codebook/codebook_text.py`

### 4.4. Índice Propio Y SPIMI

La búsqueda textual propia utiliza `InvertedIndex`. El índice almacena postings por codeword y usa similitud coseno para ordenar los resultados.

La construcción del índice se realiza mediante SPIMI. Los bloques parciales se escriben en disco como archivos JSON y luego se fusionan para construir el índice final. Esto permite separar la creación de bloques de la fase de merge y usar memoria secundaria durante el proceso.

Archivos principales:

- `backend/src/index/inverted_index.py`
- `backend/src/index/spimi.py`

### 4.5. PostgreSQL Para Texto

La comparación en PostgreSQL usa `tsvector`, índice GIN y ranking con `ts_rank`. Para las consultas se combina `websearch_to_tsquery` con una variante OR por términos, con el objetivo de permitir búsquedas más flexibles que una coincidencia estricta de todos los términos.

Archivos principales:

- `backend/api/postgres_indexer.py`
- `backend/api/routes/postgres.py`

## 5. Modalidad De Imágenes

### 5.1. Extracción Visual

La búsqueda visual usa SIFT sobre la imagen completa. Los descriptores se normalizan con RootSIFT para mejorar la comparación.

Archivo principal:

- `backend/src/extractor/sift.py`

### 5.2. Codebook Visual

Los descriptores SIFT se agrupan con K-Means para construir un vocabulario visual. En la versión final se usan:

- 100 palabras visuales SIFT;
- 16 bins HSV como descriptor adicional de color.

La representación final de imagen tiene 116 dimensiones.

Archivos principales:

- `backend/src/codebook/codebook_kmeans.py`
- `backend/api/image_pipeline.py`

### 5.3. Búsqueda Visual

La implementación propia usa `VisualSearchIndex`, que compara histogramas normalizados mediante distancia L2. La versión en PostgreSQL usa pgvector/HNSW sobre `vector(116)`.

Archivos principales:

- `backend/src/index/visual_search.py`
- `backend/api/postgres_indexer.py`

## 6. Modalidad De Audio

### 6.1. Preprocesamiento Y Extracción

El audio se divide en ventanas usando `SplitAudio`. Luego `MFCCExtractor` extrae coeficientes MFCC por frame mediante `librosa`.

Archivos principales:

- `backend/src/split/split_audio.py`
- `backend/src/extractor/mfcc.py`

### 6.2. Codebook Acústico

Los frames MFCC se agrupan mediante MiniBatchKMeans para construir un vocabulario acústico de 512 clusters. Cada archivo de audio queda representado como un histograma normalizado de palabras acústicas.

Archivo principal:

- `backend/api/mfcc_pipeline.py`

### 6.3. Búsqueda De Audio

La implementación propia usa `AudioSearchIndex`, con distancia L2 sobre histogramas acústicos. PostgreSQL usa pgvector/HNSW sobre `vector(512)`.

Archivos principales:

- `backend/src/index/audio_search.py`
- `backend/api/postgres_indexer.py`

## 7. Bases De Datos E Índices

El esquema de PostgreSQL se crea desde `backend/api/postgres_indexer.py`. Las tablas principales son:

| Tabla | Modalidad | Representación | Índice |
| --- | --- | --- | --- |
| `pg_text_docs` | Texto | `tsvector` generado desde `content` | GIN |
| `pg_image_docs` | Imagen | `vector(116)` | HNSW |
| `pg_audio_docs` | Audio | `vector(512)` | HNSW |

PostgreSQL aporta persistencia, consultas SQL y estructuras especializadas. Los índices propios permiten controlar y explicar directamente el proceso de recuperación.

## 8. Datasets

Los experimentos usan tres datasets principales:

| Dataset | Modalidad | Uso |
| --- | --- | --- |
| AG News | Texto | Evaluación textual en 1K, 10K y 100K documentos |
| Fashion200K | Imagen | Evaluación visual en 1K, 10K y 100K imágenes |
| FMA 100K WAV | Audio | Evaluación acústica en 1K, 10K y 100K archivos |

La unidad experimental es el documento o archivo fuente, no el número de chunks internos.

Los datos completos se preparan con:

```bash
docker compose --profile datasets run --rm --build datasets
docker compose exec backend python experiments/prepare_data.py
```

FMA requiere credenciales de Kaggle configuradas en `.env`.

## 9. Ejecución

Levantar el proyecto:

```bash
cp .env.example .env
docker compose up -d --build
```

Servicios:

| Servicio | URL |
| --- | --- |
| Frontend | `http://localhost:3000` |
| Backend | `http://localhost:8000` |
| PostgreSQL | `localhost:5432` |

Verificación:

```bash
docker compose ps
curl -s http://localhost:8000/health
curl -s http://localhost:8000/pipeline/status
```

## 10. Pruebas

La suite de pruebas se ejecuta con:

```bash
docker compose exec backend python -m pytest -q
```

La corrida registrada para el proyecto reporta:

```text
126 passed, 1 warning
```

## 11. Evaluación Experimental

Los benchmarks principales están en `experiments/`.

Ejecución rápida:

```bash
docker compose exec backend python experiments/prepare_data.py
docker compose exec backend python experiments/bench_text.py --scales 1k
docker compose exec backend python experiments/bench_image.py --scales 1k
docker compose exec backend python experiments/bench_audio.py --scales 1k
docker compose exec backend python experiments/plot_results.py
```

Ejecución completa:

```bash
docker compose exec backend python experiments/prepare_data.py
docker compose exec backend python experiments/bench_text.py --scales 1k 10k 100k
docker compose exec backend python experiments/bench_image.py --scales 1k 10k 100k
docker compose exec backend python experiments/bench_audio.py --scales 1k 10k 100k
docker compose exec backend python experiments/plot_results.py
```

Experimento de tamaños de diccionario textual:

```bash
docker compose exec backend python experiments/bench_text.py --scales 1k --codebook-sizes 250 500 1000 2000
```

Este experimento guarda resultados en:

```text
experiments/results/text_codebook_k_results.json
```

## 12. Métricas

La evaluación usa consultas tomadas de los propios datasets. La consulta se excluye del Top-k para evitar contar una coincidencia trivial.

Cuando una métrica aparece como `@10`, significa que se calcula sobre los primeros 10 resultados devueltos por el sistema, es decir, el Top 10. En este caso, `Precision@10` mide qué proporción de esos 10 resultados fue relevante, mientras que `Recall@10` mide qué proporción de todos los elementos relevantes disponibles en el dataset logró recuperarse dentro de esos 10 resultados.

| Métrica | Fórmula | Interpretación |
| --- | --- | --- |
| `precision_at_k` | `TP / (TP + FP)` | Proporción de resultados recuperados que son relevantes |
| `recall_at_k` | `TP / (TP + FN)` | Proporción de relevantes que fueron recuperados |
| `avg_tp_at_k` | promedio de verdaderos positivos | Aciertos relevantes por consulta |
| `avg_fp_at_k` | promedio de falsos positivos | Resultados no relevantes recuperados |
| `avg_fn_at_k` | promedio de falsos negativos | Relevantes no recuperados |

Ground truth usado:

- Texto: misma categoría de AG News.
- Imagen: misma categoría de Fashion200K.
- Audio: mismo género/carpeta de FMA.

## 13. Resultados

Los resultados guardados en `experiments/results/` son:

| Modalidad | Escala | Elementos indexados | Latencia propia (ms) | Latencia PostgreSQL (ms) | Precision@10 propia | Precision@10 PostgreSQL | Recall@10 propia | Recall@10 PostgreSQL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Texto | 1K | 1,001 | 1.171 | 10.951 | 0.584 | 0.550 | 0.0234 | 0.0221 |
| Texto | 10K | 10,008 | 13.322 | 35.115 | 0.690 | 0.730 | 0.0028 | 0.0029 |
| Texto | 100K | 100,054 | 187.956 | 113.094 | 0.766 | 0.795 | 0.0003 | 0.0003 |
| Imagen | 1K | 996 | 2.587 | 9.866 | 0.242 | 0.225 | 0.0123 | 0.0121 |
| Imagen | 10K | 9,966 | 22.953 | 11.108 | 0.284 | 0.310 | 0.0015 | 0.0016 |
| Imagen | 100K | 99,619 | 315.680 | 14.839 | 0.314 | 0.295 | 0.0002 | 0.0001 |
| Audio | 1K | 1,000 | 2.100 | 11.922 | 0.732 | 0.780 | 0.0115 | 0.0123 |
| Audio | 10K | 9,989 | 28.916 | 13.540 | 0.122 | 0.115 | 0.0018 | 0.0018 |
| Audio | 100K | 99,847 | 313.680 | 317.831 | 0.026 | 0.020 | 0.0004 | 0.0003 |

## 14. Análisis

### Texto

PostgreSQL GIN mantiene una efectividad ligeramente mayor en 10K y 100K documentos, con Precision@10 de 0.730 y 0.795 respectivamente. El índice propio queda cerca en calidad, con 0.690 y 0.766 en esas mismas escalas.

En latencia, el índice propio es más rápido en 1K y 10K, mientras que PostgreSQL GIN pasa a ser más rápido en 100K. Esto tiene sentido porque el índice propio debe rankear más candidatos conforme crece la colección, mientras que GIN aprovecha una estructura especializada y persistente. En la corrida final de 100K, SPIMI generó 101 bloques y el pico de memoria reportado fue de 1761.82 MB.

### Imagen

La latencia es directamente proporcional a la colección. Esto se observó ya que pasa de 2.587 ms en 1K a 22.953 ms en 10K y 315.680 ms en 100K. El crecimiento se explica porque el índice propio compara contra muchos más histogramas conforme aumenta la escala.

pgvector/HNSW, en cambio, mantiene una latencia mucho más estable: 9.866 ms en 1K, 11.108 ms en 10K y 14.839 ms en 100K. Esto muestra la ventaja de usar una estructura aproximada especializada cuando la colección ya es grande. En 100K, pgvector es claramente más rápido, aunque la precisión queda bastante cercana entre ambos enfoques: 0.314 para el índice propio y 0.295 para pgvector.

La precisión visual no es muy alta en términos absolutos porque Fashion200K tiene categorías amplias y visualmente ambiguas. Dos prendas pueden compartir categoría pero diferir mucho en color, forma o estilo; también puede ocurrir lo contrario, que dos imágenes sean visualmente parecidas aunque no tengan exactamente la misma etiqueta. Por eso Precision@10 es más útil para observar la calidad inmediata del ranking, mientras que Recall@10 queda bajo porque solo se recuperan 10 resultados de una cantidad grande de posibles relevantes por categoría.

### Audio

Audio es la modalidad más difícil del proyecto. La representación usa MFCC y un codebook acústico de 512 palabras, lo cual permite convertir cada archivo en un histograma comparable, pero la etiqueta usada como ground truth es el género/carpeta de FMA. Esa etiqueta es bastante amplia: dos canciones del mismo género pueden sonar muy distintas, y dos canciones de géneros distintos pueden compartir ritmo, timbre o instrumentación.

En 1K, la precisión es alta para ambos enfoques: 0.732 en el índice propio y 0.780 en pgvector. Sin embargo, al crecer la escala cae de forma marcada: en 10K baja a 0.122 y 0.115, y en 100K llega a 0.026 y 0.020. Esto no significa necesariamente que los MFCC no capturen información útil, sino que el problema se vuelve más difícil porque hay más distractores y porque el ground truth por género no describe similitud acústica fina.

En latencia, el comportamiento también cambia con la escala. En 1K y 10K pgvector es más estable, con 11.922 ms y 13.540 ms, mientras que el índice propio pasa de 2.100 ms a 28.916 ms. En 100K ambos enfoques quedan muy cerca: 313.680 ms para el índice propio y 317.831 ms para pgvector. Esto sugiere que, para audio, el costo no solo depende de la búsqueda vectorial, sino también de la forma de los histogramas, la distribución de los géneros y el tamaño efectivo de la colección indexada.

El Recall@10 es bajo en todas las escalas porque se compara un Top 10 contra grupos de relevantes muy grandes. Por ejemplo, si una categoría tiene cientos o miles de canciones, recuperar 10 resultados limita naturalmente el recall máximo observable. Por eso, para audio, Precision@10 describe mejor si los primeros resultados son razonables para el usuario, mientras que Recall@10 sirve más como referencia de cobertura global.

## 15. Trade-Offs

| Enfoque | Ventajas | Limitaciones |
| --- | --- | --- |
| Índices propios | Control del algoritmo, baja latencia en escalas pequeñas, fácil de explicar | Búsqueda lineal en imagen/audio, reconstrucción en memoria |
| PostgreSQL GIN | Muy fuerte para texto, persistente, integrado con SQL | Solo aplica directamente a texto |
| pgvector HNSW | Persistencia y búsqueda vectorial aproximada | Requiere ajustar parámetros y mantener la base |

## 16. Conclusiones

El proyecto integra texto, imagen y audio bajo una arquitectura común de recuperación. Aunque cada modalidad necesita extractores distintos, todas terminan en representaciones comparables que pueden indexarse y rankearse.

La comparación muestra que no hay una solución única para todos los casos. Los índices propios son útiles para entender y controlar el proceso de recuperación. Por otro lado, PostgreSQL y pgvector ofrecen una alternativa más preparada para persistencia y escalabilidad.

## 17. Trabajo Futuro

- Ejecutar más experimentos con distintos tamaños de diccionario textual.
- Guardar codebooks entrenados para reducir tiempos de arranque.
- Explorar embeddings modernos para imagen y audio.

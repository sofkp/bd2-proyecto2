# Fase 4 — Documentación de Gráficos

## 📊 Gráficos Generados

Ubicación: `experiments/grafica_analisis/`

### Audio
| Gráfico | Descripción | Uso en Informe |
|---------|------------|-----------------|
| `audio_latency.png` | Latencia (ms) por escala: 1K, 10K, 60K | Tabla comparativa Audio vs pgvector |
| `audio_throughput.png` | QPS por escala | Análisis escalabilidad |
| `audio_precision.png` | Precision@10 por escala | Validación de recuperación |

### Texto
| Gráfico | Descripción | Uso en Informe |
|---------|------------|-----------------|
| `text_latency.png` | Latencia (ms) InvertedIndex vs GIN | Comparativa modalidad Texto |
| `text_throughput.png` | QPS por escala | Análisis scalability |

### Imagen
| Gráfico | Descripción | Uso en Informe |
|---------|------------|-----------------|
| `image_latency.png` | Latencia VisualSearchIndex vs pgvector | Comparativa modalidad Imagen |

### Comparación Global
| Gráfico | Descripción | Uso en Informe |
|---------|------------|-----------------|
| `comparison_all_latency.png` | Latencia combinada: Audio (10K), Texto (10K), Imagen (100) | Resumen ejecutivo |

## 🔄 Estado de los Datos

### Completados
- ✅ Audio 1K: 0.04ms latencia, 26,879 QPS, P@10=1.000
- ✅ Audio 10K: 0.29ms latencia, 3,436 QPS, P@10=0.990
- ✅ Texto 1K: 2.83ms latencia, 353 QPS, P@10=1.000
- ✅ Texto 10K: 29.74ms latencia, 33.6 QPS, P@10=0.232
- ✅ Imagen 100: 0.219ms latencia, 4,564 QPS

### En Progreso
- 🔄 Audio 60K: Ejecutándose (MFCC + KMeans, ~20 min)
- ⏳ pgvector HNSW: Pendiente captura después de 60K

### Pendiente
- ❌ Integración pgvector en gráficos (datos se agregan cuando termine 60K)
- ❌ Actualizar analisis.tex con rutas de gráficos e insertar \includegraphics

## 🛠️ Reproducibilidad

Para regenerar gráficos después de completar benchmarks:
```bash
cd experiments
python plot_results.py
```

Los gráficos se actualizan automáticamente leyendo desde:
- `results/audio_results.json`
- `results/text_results.json`
- `results/image_results.json`

## 📝 Próximos Pasos

1. ⏳ Esperar finalización de `bench_audio.py` (esperado ~20-30 min)
2. 📊 Regenerar gráficos con datos pgvector
3. 📄 Insertar gráficos en `analisis.tex` usando `\includegraphics{grafica_analisis/...}`
4. 🎯 Completar tablas comparativas en secciones de Audio/Texto/Imagen
5. ✔️ Final: LaTeX → PDF

---
Generado: 2026-06-28 23:28

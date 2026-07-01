# Flujo de Trabajo — BD2 Proyecto 2

## 1. Primeros pasos (solo una vez)

### Clonar el repo
```bash
git clone https://github.com/LuixRom/bd2-proyecto2.git
cd bd2-proyecto2
```

### Inicializar el proyecto
```bash
bash init.sh
```
Esto instala dependencias de Python, del Frontend y levanta los contenedores cuando el `docker-compose.yml` esté listo.

---

## 2. Responsabilidades

| Integrante | Módulo | Detalle |
|---|---|---|
| LuixRom | Codebook + Frontend | Top-K palabras + K-Means visual + K-Means acústico + UI Next.js |
| hanksvi | Retrieval e Índices + Apps | SPIMI + índice invertido + histogramas visuales y acústicos + App 1 + App 2 |
| KarolayTamayoH | Feature Extraction + Evaluación | TF-IDF + SIFT + MFCC + Evaluación experimental + Comparativas GIN/pgvector |
| softkp | Split + Infra | División texto/imagen/audio (párrafos, patches, sliding windows) + Docker + PostgreSQL |

---

## 3. Ramas

Cada integrante trabaja en su rama asignada:

| Integrante | Rama |
|---|---|
| LuixRom | `feat/luix-codebook` |
| hanksvi | `feat/hanks-retrieval` |
| KarolayTamayoH | `feat/karolay-extractor` |
| softkp | `feat/soft-split` |

### Moverse a tu rama
```bash
git checkout feat/tu-rama
```

---

## 4. Flujo diario de trabajo

### Antes de empezar cada día
```bash
# Asegurarte de estar en tu rama
git checkout feat/tu-rama

# Traer los últimos cambios de dev
git pull origin dev
```

### Mientras trabajas
```bash
# Ver qué archivos has modificado
git status

# Agregar tus cambios
git add .

# Hacer commit con el formato correcto
git commit -m "[modulo] tipo: descripcion corta"
```

### Ejemplos de commits correctos
```
[codebook] feat: implement K-Means for visual words
[extractor] feat: add TF-IDF vectorizer
[split] fix: fix sliding window overlap
[retrieval] feat: implement SPIMI algorithm
[frontend] feat: add search UI component
```

### Subir tus cambios
```bash
git push origin feat/tu-rama
```

---

## 5. Flujo en GitHub Projects

El tablero Kanban refleja el estado de cada tarea:

```
Backlog → Ready → In Progress → In Review → Done
```

| Estado | Cuándo | Cómo |
|---|---|---|
| Backlog | Issue creado, sin empezar | Automático al crear el issue |
| Ready | Listo para empezar, entorno configurado | Manual, arrastrar en el tablero |
| In Progress | Trabajando activamente | Manual, arrastrar en el tablero |
| In Review | PR abierto esperando revisión | Automático al abrir el PR |
| Done | PR mergeado a dev | Automático al mergear |

---

## 6. Cómo abrir un Pull Request

Cuando termines una funcionalidad:

1. Sube tus cambios:
```bash
git push origin feat/tu-rama
```

2. Ve a GitHub → tu repo → verás el botón **"Compare & pull request"**

3. En la descripción del PR escribe:
```
Closes #numero-del-issue
```
Esto cierra el issue automáticamente cuando se mergee.

4. Asigna a otro integrante como reviewer

5. Espera la revisión antes de mergear

---

## 7. Tests

Cada integrante escribe sus tests dentro de su carpeta:

```
backend/tests/
├── split/        → softkp
├── extractor/    → KarolayTamayoH
├── codebook/     → LuixRom
└── index/        → hanksvi
```

### Correr todos los tests
```bash
bash run_tests.sh
```

### Correr solo tus tests
```bash
python -m pytest backend/tests/tu-modulo/ -v
```

⚠️ **Antes de abrir un PR siempre corre tus tests primero.**

---

## 8. Estructura del proyecto

```
bd2-proyecto2/
├── frontend/              # Next.js — LuixRom
├── backend/
│   ├── src/
│   │   ├── split/         # softkp
│   │   ├── extractor/     # KarolayTamayoH
│   │   ├── codebook/      # LuixRom
│   │   └── index/         # hanksvi
│   ├── api/               # FastAPI endpoints
│   └── tests/             # Tests por módulo
├── db/                    # Migraciones PostgreSQL
├── experiments/           # Benchmarks y comparativas
├── docker-compose.yml     # softkp
├── init.sh                # Inicializar proyecto
├── run_tests.sh           # Correr todos los tests
├── CONSTITUTION.md        # Reglas del equipo
└── FLUJO.md               # Este archivo
```

---

## 9. Reglas importantes

- ❌ No hacer push directo a `main` ni a `dev`
- ❌ No mergear tu propio PR
- ✅ Siempre correr `run_tests.sh` antes de abrir un PR
- ✅ Referenciar el issue en el PR con `Closes #numero`
- ✅ Máximo 100 líneas por archivo (excepto configuración)
- ✅ Un commit = una cosa concreta
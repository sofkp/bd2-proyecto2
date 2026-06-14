#!/bin/bash
set -e

echo " Inicializando BD2 Proyecto 2..."

# Cargar variables de entorno si existe .env
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# 1. Verificar dependencias
echo " Verificando dependencias..."
command -v docker  >/dev/null 2>&1 || { echo "❌ Docker no está instalado"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ Python3 no está instalado"; exit 1; }
command -v node    >/dev/null 2>&1 || { echo "❌ Node.js no está instalado"; exit 1; }

# Detectar Docker Compose (v2 'docker compose' o v1 'docker-compose')
if docker compose version >/dev/null 2>&1; then
    DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DC="docker-compose"
else
    echo " Docker Compose no está instalado"; exit 1
fi
echo "   Usando: $DC"

# 2. Levantar PostgreSQL + pgvector
echo " Levantando contenedores..."
if [ -s docker-compose.yml ]; then
    $DC up -d
else
    echo "  docker-compose.yml vacío, saltando este paso..."
fi

# 3. Esperar a PostgreSQL (portable: sirve en v1 y v2)
echo " Esperando a PostgreSQL..."
for i in $(seq 1 30); do
    if $DC exec -T db pg_isready -U "${POSTGRES_USER:-bd2}" -d "${POSTGRES_DB:-bd2_proyecto2}" >/dev/null 2>&1; then
        echo " PostgreSQL listo"
        break
    fi
    sleep 1
    if [ "$i" -eq 30 ]; then echo "  PostgreSQL tardó demasiado; revisa '$DC logs db'."; fi
done

# Asegurar extensión aunque el volumen ya existiera
$DC exec -T db psql -U "${POSTGRES_USER:-bd2}" -d "${POSTGRES_DB:-bd2_proyecto2}" \
    -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null 2>&1 || true

# 4. Entorno virtual + dependencias Python
echo " Configurando entorno Python..."
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install --quiet --upgrade pip
if [ -f backend/requirements.txt ]; then pip install --quiet -r backend/requirements.txt; fi

# 5. Dependencias Frontend
if [ -f frontend/package.json ]; then
    echo "  Instalando dependencias Frontend..."
    (cd frontend && npm install)
fi

# 6. Tests (no detiene el init si aún no hay tests)
if [ -f run_tests.sh ]; then
    echo " Corriendo tests..."
    bash run_tests.sh || echo "⚠️  Tests fallaron o aún no existen (normal al inicio)."
fi

echo ""
echo " Proyecto listo!"
echo "   Activa el entorno con: source .venv/bin/activate"
echo ""
echo " Datasets (opcional, NO se versionan — se descargan aparte):"
echo "   ./scripts/download_data.sh arxiv     # SciMMIR, liviano — empieza por aquí"
echo "   ./scripts/download_data.sh spotify   # ~44 MB"
echo "   ./scripts/download_data.sh fashion   # pesado (decenas de GB)"
echo "   (Kaggle pide credenciales una vez: ~/.kaggle/kaggle.json)"
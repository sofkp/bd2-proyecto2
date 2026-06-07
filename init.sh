#!/bin/bash

echo "🚀 Inicializando BD2 Proyecto 2..."

# 1. Verificar dependencias
echo "📦 Verificando dependencias..."
command -v docker >/dev/null 2>&1 || { echo "❌ Docker no está instalado"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose no está instalado"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ Python3 no está instalado"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "❌ Node.js no está instalado"; exit 1; }

# 2. Levantar PostgreSQL + pgvector 
echo "🐳 Levantando contenedores..."
if [ -s docker-compose.yml ]; then
    docker-compose up -d
else
    echo "⚠️  docker-compose.yml vacío, saltando este paso..."
fi

# 3. Esperar que PostgreSQL esté listo
echo " Esperando PostgreSQL..."
sleep 5

# 4. Instalar dependencias Python
echo " Instalando dependencias Python..."
pip install -r backend/requirements.txt

# 5. Instalar dependencias Frontend
echo "  Instalando dependencias Frontend..."
cd frontend && npm install && cd ..

# 6. Correr tests para verificar que todo está OK
echo " Corriendo tests..."
bash run_tests.sh

echo " Proyecto listo!"
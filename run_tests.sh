#!/bin/bash

echo " Corriendo tests BD2 Proyecto 2..."

# Tests Split 
echo " Tests Split..."
python -m pytest backend/tests/split/ -v

# Tests Extractor 
echo " Tests Extractor..."
python -m pytest backend/tests/extractor/ -v

# Tests Codebook 
echo " Tests Codebook..."
python -m pytest backend/tests/codebook/ -v

# Tests Retrieval e Índices 
echo " Tests Retrieval..."
python -m pytest backend/tests/index/ -v

echo " Todos los tests completados!"
-- pgvector: similitud vectorial (SIFT / MFCC / TF-IDF embeddings)
CREATE EXTENSION IF NOT EXISTS vector;
-- pg_trgm: búsqueda por trigramas (opcional, usa índices GIN)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- GIN ya viene integrado en PostgreSQL, no necesita extensión.
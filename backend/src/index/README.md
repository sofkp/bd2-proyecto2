# Retrieval and Index Module

This module receives codeword histograms from the Codebook stage and returns
ranked chunks to the FastAPI layer.

## Contracts

Input record:

```python
{
    "chunk_id": "text_doc_0",
    "modality": "text",
    "histogram": [2, 0, 1],
    "metadata": {"source": "document.txt"},
}
```

Output result:

```python
{
    "chunk_id": "text_doc_0",
    "score": 0.91,
    "metadata": {"source": "document.txt"},
}
```

## Components

- `models.py`: validates Codebook records and formats search results.
- `metrics.py`: cosine similarity, L2 distance, normalization, and top-K.
- `inverted_index.py`: text postings and cosine-ranked retrieval.
- `spimi.py`: configurable SPIMI block creation and block merging.
- `visual_search.py`: visual histogram retrieval using L2 distance.
- `audio_search.py`: acoustic histogram retrieval using L2 distance.

Text scores are cosine similarities, so a higher score is better. Visual and
audio scores are L2 distances, so a lower score is better.

## API

Available endpoints:

```text
GET  /health
POST /search/text
POST /search/music
POST /search/visual
```

Each search request contains the records to index, a query histogram, and the
number of requested results:

```json
{
  "records": [
    {
      "chunk_id": "image_0",
      "modality": "image",
      "histogram": [1, 0, 2],
      "metadata": {"product": "shoe"}
    }
  ],
  "query_histogram": [1, 0, 2],
  "k": 10
}
```

The endpoints are currently stateless and build an in-memory index per request.
PostgreSQL persistence can replace this storage without changing the Codebook
or API contracts.

## Run

Install dependencies:

```bash
python -m pip install -r backend/requirements.txt
```

Start the API:

```bash
python -m uvicorn backend.main:app --reload
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Tests

Run the complete backend suite:

```bash
python -m pytest backend/tests -v
```

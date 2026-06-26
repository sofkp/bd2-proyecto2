from collections.abc import Callable

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.api.schemas import SearchRequest, SearchResultResponse
from backend.src.index import AudioSearchIndex, InvertedIndex, VisualSearchIndex

router = APIRouter(prefix="/search", tags=["search"])


def _execute_search(
    request: SearchRequest,
    index_factory: Callable[[], object],
) -> list[SearchResultResponse]:
    index = index_factory()
    try:
        for record in request.records:
            index.add_record(record.model_dump())
        results = index.search(np.asarray(request.query_histogram), k=request.k)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return [
        SearchResultResponse(
            chunk_id=result.chunk_id,
            score=result.score,
            metadata=result.metadata,
        )
        for result in results
    ]


@router.post("/text", response_model=list[SearchResultResponse])
def search_text(request: SearchRequest) -> list[SearchResultResponse]:
    """Search text chunks using cosine similarity."""
    return _execute_search(request, InvertedIndex)


@router.post("/music", response_model=list[SearchResultResponse])
def search_music(request: SearchRequest) -> list[SearchResultResponse]:
    """Search music chunks using acoustic histogram distance."""
    return _execute_search(request, AudioSearchIndex)


@router.post("/visual", response_model=list[SearchResultResponse])
def search_visual(request: SearchRequest) -> list[SearchResultResponse]:
    """Search image chunks using visual histogram distance."""
    return _execute_search(request, VisualSearchIndex)

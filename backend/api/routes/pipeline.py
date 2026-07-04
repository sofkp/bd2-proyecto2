import time
from pydantic import BaseModel, Field
from fastapi import APIRouter, File, UploadFile

from backend.api.pipeline_state import text_pipeline
from backend.api.image_pipeline import image_pipeline
from backend.api.mfcc_pipeline import mfcc_pipeline

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)


def _wrap(results: list[dict], stats: dict, query_ms: float) -> dict:
    return {"results": results, "stats": {**stats, "query_ms": round(query_ms, 2)}}


@router.get("/status")
def pipeline_status() -> dict:
    return {
        "text": {
            "ready": text_pipeline.ready,
            "indexed_docs": text_pipeline.indexed_docs,
            "indexed_chunks": text_pipeline.indexed_chunks,
        },
        "image": {
            "ready": image_pipeline.ready,
            "indexed_images": image_pipeline.indexed_images,
        },
        "audio_mfcc": {
            "ready": mfcc_pipeline.ready,
            "indexed_files": mfcc_pipeline.indexed_files,
        },
    }


@router.post("/search/text")
def pipeline_search_text(request: TextSearchRequest) -> dict:
    t0 = time.perf_counter()
    results = text_pipeline.search(request.query, k=request.k)
    return _wrap(results, text_pipeline.index_stats(), (time.perf_counter() - t0) * 1000)


@router.post("/search/text/pdf")
async def pipeline_search_text_pdf(file: UploadFile = File(...), k: int = 10) -> dict:
    import fitz
    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    if not text.strip():
        return _wrap([], text_pipeline.index_stats(), 0)
    t0 = time.perf_counter()
    results = text_pipeline.search(text, k=k)
    return _wrap(results, text_pipeline.index_stats(), (time.perf_counter() - t0) * 1000)


@router.post("/search/image")
async def pipeline_search_image(file: UploadFile = File(...), k: int = 10) -> dict:
    image_bytes = await file.read()
    t0 = time.perf_counter()
    results = image_pipeline.search(image_bytes, k=k)
    return _wrap(results, image_pipeline.index_stats(), (time.perf_counter() - t0) * 1000)


@router.post("/search/audio-file")
async def pipeline_search_audio_file(file: UploadFile = File(...), k: int = 10) -> dict:
    audio_bytes = await file.read()
    t0 = time.perf_counter()
    results = mfcc_pipeline.search(audio_bytes, k=k)
    return _wrap(results, mfcc_pipeline.index_stats(), (time.perf_counter() - t0) * 1000)

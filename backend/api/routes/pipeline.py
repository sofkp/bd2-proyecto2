import time
from pydantic import BaseModel, Field
from fastapi import APIRouter, File, UploadFile

from backend.api.pipeline_state import text_pipeline
from backend.api.music_pipeline import music_pipeline
from backend.api.image_pipeline import image_pipeline
from backend.api.mfcc_pipeline import mfcc_pipeline

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)


class AudioFeaturesRequest(BaseModel):
    features: list[float] = Field(min_length=1)
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
        "music": {
            "ready": music_pipeline.ready,
            "indexed_songs": music_pipeline.indexed_songs,
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


@router.post("/search/music/lyrics")
def pipeline_search_music_lyrics(request: TextSearchRequest) -> dict:
    t0 = time.perf_counter()
    results = music_pipeline.search_by_lyrics(request.query, k=request.k)
    return _wrap(results, music_pipeline.index_stats(), (time.perf_counter() - t0) * 1000)


@router.post("/search/music/audio")
def pipeline_search_music_audio(request: AudioFeaturesRequest) -> dict:
    t0 = time.perf_counter()
    results = music_pipeline.search_by_audio_features(request.features, k=request.k)
    return _wrap(results, music_pipeline.index_stats(), (time.perf_counter() - t0) * 1000)


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


@router.get("/music/feature-names")
def music_feature_names() -> list[str]:
    return music_pipeline.audio_feature_names()

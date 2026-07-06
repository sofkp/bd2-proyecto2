from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router as search_router
from backend.api.routes.pipeline import router as pipeline_router
from backend.api.routes.postgres import router as postgres_router
from backend.api.pipeline_state import text_pipeline
from backend.api.image_pipeline import image_pipeline
from backend.api.mfcc_pipeline import mfcc_pipeline
from backend.api import postgres_indexer

IMAGES_DIR = Path(__file__).parent.parent.parent / "data" / "samples" / "images"
IMAGES_FULL_DIR = Path(__file__).parent.parent.parent / "data" / "full" / "fashion200k"
AUDIO_DIR = Path(__file__).parent.parent.parent / "data" / "full" / "audio"
AUDIO_FULL_DIR = Path(__file__).parent.parent.parent / "data" / "full"
AUDIO_SAMPLES_DIR = Path(__file__).parent.parent.parent / "data" / "samples" / "audio"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac"}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return max(0, int(value))
    except ValueError:
        return default


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = Path(__file__).parent.parent.parent
    index_full_data = _env_flag("APP_INDEX_FULL_DATA", default=False)
    max_text_docs = _env_int("APP_MAX_TEXT_DOCS", 10_000 if index_full_data else 0)
    max_images = _env_int("APP_MAX_IMAGES", 10_000 if index_full_data else 150)
    max_audio_files = _env_int("APP_MAX_AUDIO_FILES", 10_000 if index_full_data else 0)
    sample_dir = root / "data" / "samples" / "text"
    agnews_dir = root / "data" / "full" / "agnews"
    text_dirs = [sample_dir]
    if index_full_data:
        text_dirs.append(agnews_dir)
    text_pipeline.index_directories(text_dirs, max_files=max_text_docs or None)

    image_candidates = [IMAGES_DIR]
    if index_full_data:
        image_candidates.append(IMAGES_FULL_DIR)
    image_dirs = [d for d in image_candidates if d.exists()]
    if image_dirs:
        image_pipeline.index_directories(image_dirs, max_images=max_images)

    audio_sample_dir = root / "data" / "samples" / "audio"
    audio_full_dir = root / "data" / "full"
    audio_candidates = [audio_sample_dir]
    if index_full_data:
        audio_candidates.append(audio_full_dir)
    audio_dirs = [
        d for d in audio_candidates
        if d.exists() and any(p.suffix.lower() in AUDIO_EXTENSIONS for p in d.rglob("*"))
    ]
    if audio_dirs:
        mfcc_pipeline.index_directories(audio_dirs, max_files=max_audio_files or None)

    # PostgreSQL: crear schema y poblar con los mismos datos
    try:
        postgres_indexer.create_schema()
        postgres_indexer.index_text(text_dirs)
        if image_pipeline.ready:
            postgres_indexer.index_images(
                image_pipeline._index._histograms,
                image_pipeline._index._metadata,
            )
        if mfcc_pipeline.ready:
            postgres_indexer.index_audio(
                mfcc_pipeline._index._histograms,
                mfcc_pipeline._index._metadata,
            )
    except Exception as e:
        print(f"[postgres] Error al indexar: {e}")

    yield


app = FastAPI(title="BD2 Multimodal Retrieval API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

if IMAGES_FULL_DIR.exists():
    app.mount("/images-full", StaticFiles(directory=str(IMAGES_FULL_DIR)), name="images-full")

if AUDIO_DIR.exists():
    app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

if AUDIO_FULL_DIR.exists():
    app.mount("/audio-full", StaticFiles(directory=str(AUDIO_FULL_DIR)), name="audio-full")

if AUDIO_SAMPLES_DIR.exists():
    app.mount("/audio-samples", StaticFiles(directory=str(AUDIO_SAMPLES_DIR)), name="audio-samples")

app.include_router(search_router)
app.include_router(pipeline_router)
app.include_router(postgres_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

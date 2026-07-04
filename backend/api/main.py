from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router as search_router
from backend.api.routes.pipeline import router as pipeline_router
from backend.api.routes.postgres import router as postgres_router
from backend.api.pipeline_state import text_pipeline
from backend.api.music_pipeline import music_pipeline
from backend.api.image_pipeline import image_pipeline
from backend.api.mfcc_pipeline import mfcc_pipeline
from backend.api import postgres_indexer

IMAGES_DIR = Path(__file__).parent.parent.parent / "data" / "samples" / "images"
IMAGES_FULL_DIR = Path(__file__).parent.parent.parent / "data" / "full" / "tiny_imagenet"
AUDIO_DIR = Path(__file__).parent.parent.parent / "data" / "full" / "audio"
AUDIO_SAMPLES_DIR = Path(__file__).parent.parent.parent / "data" / "samples" / "audio"


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = Path(__file__).parent.parent.parent
    sample_dir = root / "data" / "samples" / "text"
    arxiv_dir = root / "data" / "full" / "arxiv"
    agnews_dir = root / "data" / "full" / "agnews"
    text_dirs = [sample_dir, arxiv_dir, agnews_dir]
    text_pipeline.index_directories(text_dirs)

    spotify_csv = Path(__file__).parent.parent.parent / "data" / "full" / "spotify_songs.csv"
    if spotify_csv.exists():
        music_pipeline.load_csv(spotify_csv)

    image_dirs = [d for d in (IMAGES_DIR, IMAGES_FULL_DIR) if d.exists()]
    if image_dirs:
        image_pipeline.index_directories(image_dirs)

    audio_sample_dir = root / "data" / "samples" / "audio"
    audio_full_dir = root / "data" / "full" / "audio"
    audio_dirs = [
        d for d in (audio_sample_dir, audio_full_dir)
        if d.exists() and any(d.rglob("*.wav"))
    ]
    if audio_dirs:
        mfcc_pipeline.index_directories(audio_dirs)

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

if AUDIO_SAMPLES_DIR.exists():
    app.mount("/audio-samples", StaticFiles(directory=str(AUDIO_SAMPLES_DIR)), name="audio-samples")

app.include_router(search_router)
app.include_router(pipeline_router)
app.include_router(postgres_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

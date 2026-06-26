from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router as search_router
from backend.api.routes.pipeline import router as pipeline_router
from backend.api.pipeline_state import text_pipeline
from backend.api.music_pipeline import music_pipeline
from backend.api.image_pipeline import image_pipeline
from backend.api.mfcc_pipeline import mfcc_pipeline

IMAGES_DIR = Path(__file__).parent.parent.parent / "data" / "samples" / "images"
AUDIO_DIR = Path(__file__).parent.parent.parent / "data" / "full" / "audio"


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = Path(__file__).parent.parent.parent
    sample_dir = root / "data" / "samples" / "text"
    arxiv_dir = root / "data" / "full" / "arxiv"
    text_pipeline.index_directories([sample_dir, arxiv_dir])

    spotify_csv = Path(__file__).parent.parent.parent / "data" / "full" / "spotify_songs.csv"
    if spotify_csv.exists():
        music_pipeline.load_csv(spotify_csv)

    if IMAGES_DIR.exists():
        image_pipeline.index_directory(IMAGES_DIR)

    audio_dir = root / "data" / "full" / "audio"
    if audio_dir.exists() and any(audio_dir.rglob("*.wav")):
        mfcc_pipeline.index_directory(audio_dir)

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

if AUDIO_DIR.exists():
    app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

app.include_router(search_router)
app.include_router(pipeline_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

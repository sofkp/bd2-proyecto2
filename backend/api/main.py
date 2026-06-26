from fastapi import FastAPI

from backend.api.routes import router as search_router

app = FastAPI(title="BD2 Multimodal Retrieval API")
app.include_router(search_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}

from fastapi import FastAPI
from backend.routers.ingest_router import router as ingest_router
from backend.routers.ask_router import router as ask_router

app = FastAPI()

app.include_router(ingest_router, prefix="/ingest")
app.include_router(ask_router, prefix="/ask")

@app.get("/health")
def health():
    return {"status": "ok"}

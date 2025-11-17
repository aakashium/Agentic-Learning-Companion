from fastapi import FastAPI
from backend.routers.ingest_router import router as ingest_router
from backend.routers.ask_router import router as ask_router
from backend.routers.quiz_router import router as quiz_router
from backend.routers.mentor_router import router as mentor_router
from backend.routers.eval_router import router as eval_router
from backend.routers.debug_router import router as debug_router
from backend.routers.health_router import router as health_router

app = FastAPI(title="Agentic Learning Companion - Backend")

app.include_router(ingest_router, prefix="/ingest")
app.include_router(ask_router, prefix="/ask")
app.include_router(quiz_router, prefix="/quiz")
app.include_router(mentor_router, prefix="/mentor")
app.include_router(eval_router, prefix="/eval")
app.include_router(debug_router, prefix="/debug")
app.include_router(health_router, prefix="/health")
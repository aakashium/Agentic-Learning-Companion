from fastapi import FastAPI
from routers.ingest_router import router as ingest_router
from routers.ask_router import router as ask_router
from routers.quiz_router import router as quiz_router
from routers.mentor_router import router as mentor_router

app = FastAPI(title="Agentic Learning Companion - Backend")

app.include_router(ingest_router, prefix="/ingest")
app.include_router(ask_router, prefix="/ask")
app.include_router(quiz_router, prefix="/quiz")
app.include_router(mentor_router, prefix="/mentor")
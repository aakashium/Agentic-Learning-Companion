from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def ingest_text():
    return {"message": "ingest endpoint working"}

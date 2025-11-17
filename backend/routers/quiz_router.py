from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def generate_quiz(topic: str):
    return {"topic": topic, "quiz": []}

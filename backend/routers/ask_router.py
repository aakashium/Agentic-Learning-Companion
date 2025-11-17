from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def ask_question(q: str):
    return {"question": q, "answer": "placeholder answer"}

from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def mentor_chat(message: str):
    return {"message": message, "reply": "mentor response placeholder"}

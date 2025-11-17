from fastapi import APIRouter, Query
from backend.vectorstore import VectorStore
from backend.llm_client import LLMClient

router = APIRouter()
vs = VectorStore()
llm = LLMClient()

@router.get("/")
def ask(q: str = Query(...)):
    docs = vs.search(q)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Use ONLY this context to answer:

{context}

Question: {q}
    """

    ans = llm.generate(prompt)

    return {"answer": ans, "sources": [d.page_content for d in docs]}

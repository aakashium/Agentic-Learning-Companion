from fastapi import APIRouter, Query
from backend.eval.compare_retrievers import evaluate_all_retrievers

router = APIRouter()

@router.get("/single")
def eval_single(query: str = Query(...), gold: str = Query(...), k: int = 5):
    """
    gold = comma-separated doc_ids
    """
    gold_ids = [g.strip() for g in gold.split(",")]
    return evaluate_all_retrievers(query, gold_ids, k)

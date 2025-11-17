import os
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from backend.config import settings

SCHEMA = Schema(
    doc_id=ID(stored=True, unique=True),
    content=TEXT(stored=True)
)


def _get_or_create_index():
    index_dir = settings.WHOOSH_INDEX_DIR
    os.makedirs(index_dir, exist_ok=True)

    if not index.exists_in(index_dir):
        return index.create_in(index_dir, SCHEMA)
    return index.open_dir(index_dir)


def index_documents(docs):
    """
    docs: list of dicts like {"doc_id": "...", "content": "..."}
    """
    ix = _get_or_create_index()
    writer = ix.writer()
    for d in docs:
        writer.update_document(
            doc_id=str(d["doc_id"]),
            content=d["content"]
        )
    writer.commit()


def bm25_search(query: str, top_k: int = 5):
    ix = _get_or_create_index()
    qp = QueryParser("content", schema=SCHEMA)
    q = qp.parse(query)

    hits_out = []
    with ix.searcher() as searcher:
        hits = searcher.search(q, limit=top_k)
        for h in hits:
            hits_out.append({
                "doc_id": h["doc_id"],
                "content": h["content"],
                "score": float(h.score),
            })
    return hits_out

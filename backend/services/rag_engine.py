import numpy as np
from backend.vectorstore.chroma_client import ChromaVectorStore
from backend.services.bm25_search import bm25_search
from backend.models.llm_client import LLMClient

class RAGEngine:
    def __init__(self, w_vec: float = 0.6, w_bm25: float = 0.4):
        self.vectorstore = ChromaVectorStore()
        self.llm = LLMClient()
        self.w_vec = w_vec
        self.w_bm25 = w_bm25

    def _normalize(self, scores):
        arr = np.array(scores, dtype=float)
        if arr.size == 0:
            return arr
        mn, mx = arr.min(), arr.max()
        if abs(mx - mn) < 1e-9:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn + 1e-9)

    def hybrid_retrieve(self, query: str, k: int = 5):
        # Vector search
        vec_results = self.vectorstore.retrieve_with_scores(query, k)
        vec_docs = []
        for doc, score in vec_results:
            doc_id = doc.metadata.get("doc_id", "vec_unknown")
            vec_docs.append({
                "doc_id": doc_id,
                "content": doc.page_content,
                "vec_score": float(score)
            })

        # BM25 search
        bm_results = bm25_search(query, top_k=k)

        # normalize scores
        vec_norm = self._normalize([d["vec_score"] for d in vec_docs]) if vec_docs else []
        bm_norm = self._normalize([d["score"] for d in bm_results]) if bm_results else []

        merged = {}

        # merge vector scores
        for i, d in enumerate(vec_docs):
            doc_id = d["doc_id"]
            merged.setdefault(doc_id, {"content": d["content"], "vec": 0.0, "bm": 0.0})
            merged[doc_id]["vec"] = float(vec_norm[i]) if len(vec_norm) > i else float(d["vec_score"])

        # merge bm25 scores
        for i, d in enumerate(bm_results):
            doc_id = d["doc_id"]
            merged.setdefault(doc_id, {"content": d["content"], "vec": 0.0, "bm": 0.0})
            merged[doc_id]["bm"] = float(bm_norm[i]) if len(bm_norm) > i else float(d["score"])

        final = []
        for doc_id, vals in merged.items():
            combined = self.w_vec * vals["vec"] + self.w_bm25 * vals["bm"]
            final.append({
                "doc_id": doc_id,
                "content": vals["content"],
                "vec_score": vals["vec"],
                "bm25_score": vals["bm"],
                "combined_score": combined,
            })

        final_sorted = sorted(final, key=lambda x: x["combined_score"], reverse=True)[:k]
        return final_sorted

    def answer(self, question: str, k: int = 5):
        retrieved = self.hybrid_retrieve(question, k=k)
        context = "\n\n".join(
            [f"[{i+1}] {item['content']}" for i, item in enumerate(retrieved)]
        )

        prompt = f"""
You are a helpful AI tutor. Use ONLY the context below to answer the question.
Cite sources like [1], [2] referring to the numbered context snippets.

Context:
{context}

Question: {question}

Answer:
"""
        answer = self.llm.generate(prompt)
        return {"answer": answer, "retrieved": retrieved}

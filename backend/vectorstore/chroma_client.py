import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from backend.config import settings

import os
os.environ["HF_HUB_USER_AGENT"] = settings.HF_HUB_USER_AGENT


class ChromaVectorStore:
    def __init__(self):
        os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)

        os.environ["HF_HUB_USER_AGENT"] = settings.HF_HUB_USER_AGENT

        # Open-source embeddings
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.db = Chroma(
            collection_name="alc_collection",
            embedding_function=self.embedder,
            persist_directory=settings.CHROMA_DB_PATH
        )

    def add_documents(self, docs):
        self.db.add_documents(docs)
        self.db.persist()

    def retrieve_with_scores(self, query: str, k: int = 5):
        return self.db.similarity_search_with_score(query, k=k)

    def retrieve(self, query: str, k: int = 5):
        return self.db.similarity_search(query, k=k)

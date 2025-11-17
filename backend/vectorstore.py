import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from backend.config import settings

class VectorStore:
    def __init__(self):
        os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)

        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.db = Chroma(
            collection_name="alc_collection",
            embedding_function=self.embedder,
            persist_directory=settings.CHROMA_DB_PATH,
        )

    def add(self, docs):
        self.db.add_documents(docs)

    def search(self, query, k=5):
        return self.db.similarity_search(query, k=k)

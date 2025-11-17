import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from backend.config import settings


class ChromaVectorStore:
    def __init__(self):
        os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)

        self.embedder = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"  
        )

        # Initialize persistent Chroma DB
        self.db = Chroma(
            collection_name="alc_collection",
            embedding_function=self.embedder,
            persist_directory=settings.CHROMA_DB_PATH
        )

    def add_documents(self, docs):
        """Add LangChain Document objects to Chroma"""
        self.db.add_documents(docs)
        self.db.persist()

    def retrieve_with_scores(self, query: str, k: int = 5):
        """
        Returns list of (Document, score)
        Score is similarity (higher = better)
        """
        return self.db.similarity_search_with_score(query, k=k)

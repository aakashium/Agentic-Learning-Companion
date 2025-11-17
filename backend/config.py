import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    EMBEDDING_MODEL = "gemini-embedding"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    CHROMA_DB_PATH = "vectorstore/db"
    BM25_INDEX_PATH = "vectorstore/bm25_index"

settings = Settings()


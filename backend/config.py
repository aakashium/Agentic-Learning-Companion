import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    CHROMA_DB_PATH = "vectorstore/db"
    WHOOSH_INDEX_DIR = "vectorstore/bm25_index"
    USER_AGENT = os.getenv("USER_AGENT")
    HF_HUB_USER_AGENT = os.getenv("HF_HUB_USER_AGENT", "Agentic-Learning-Companion/1.0")

settings = Settings()


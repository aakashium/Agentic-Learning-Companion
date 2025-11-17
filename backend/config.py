import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from backend directory explicitly
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)


load_dotenv()

class Settings:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    CHROMA_DB_PATH = "vectorstore/db"
    WHOOSH_INDEX_DIR = "vectorstore/bm25_index"
    USER_AGENT = os.getenv("USER_AGENT")
    HF_HUB_USER_AGENT = os.getenv("HF_HUB_USER_AGENT", "Agentic-Learning-Companion/1.0")

settings = Settings()


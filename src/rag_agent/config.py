from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if not value or value.lower() in {"replace_me", "your_api_key_here"}:
        return None
    return value


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_PATH = DATA_DIR / "knowledge.md"

OPENAI_API_KEY = _optional_env("OPENAI_API_KEY")
OPENAI_BASE_URL = _optional_env("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(DATA_DIR / "faiss_index")))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.6"))
LEXICAL_WEIGHT = float(os.getenv("LEXICAL_WEIGHT", "0.4"))
RETRIEVAL_K = int(os.getenv('RETRIEVAL_K', '4'))
MAX_QUERY_CHARS = int(os.getenv('MAX_QUERY_CHARS', '400'))
MAX_CONTEXT_CHARS = int(os.getenv('MAX_CONTEXT_CHARS', '12000'))
LLM_TIMEOUT_SECONDS = float(os.getenv('LLM_TIMEOUT_SECONDS', '60'))

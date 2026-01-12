import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Settings:
    api_host = os.getenv("API_HOST")
    api_port = int(os.getenv("API_PORT"))

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_model = os.getenv("OPENROUTER_MODEL")
    openrouter_site_url = os.getenv("OPENROUTER_SITE_URL")
    openrouter_app_name = os.getenv("OPENROUTER_APP_NAME")

    tavily_api_key = os.getenv("TAVILY_API_KEY")

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    chat_ttl_seconds = int(os.getenv("CHAT_TTL_SECONDS"))
    chat_max_turns = int(os.getenv("CHAT_MAX_TURNS"))

    postgres_url = os.getenv("POSTGRES_URL")

    kb_dir = Path(str(BASE_DIR / "kb"))
    kb_index_dir = Path(str(BASE_DIR / ".kb_index"))
    kb_emb_model = os.getenv("KB_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    kb_chunk_max_chars = int(os.getenv("KB_CHUNK_MAX_CHARS"))
    kb_chunk_overlap_chars = int(os.getenv("KB_CHUNK_OVERLAP_CHARS"))
    kb_hybrid_alpha = float(os.getenv("KB_HYBRID_ALPHA"))
    kb_candidates = int(os.getenv("KB_CANDIDATES"))
    kb_max_upload_bytes = int(os.getenv("KB_MAX_UPLOAD_BYTES"))
    kb_use_rerank = os.getenv("KB_USE_RERANK") == "1"
    kb_rerank_model = os.getenv("KB_RERANK_MODEL")
    kb_rerank_topn = int(os.getenv("KB_RERANK_TOPN"))


settings = Settings()
settings.kb_dir.mkdir(parents=True, exist_ok=True)
settings.kb_index_dir.mkdir(parents=True, exist_ok=True)

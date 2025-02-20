# Global configurations
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")


class Settings(BaseSettings):
    claude_api_key: str
    openai_api_key: str
    api_url: str
    model_name: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 1000

    class Config:
        env_file = ".env"


settings = Settings()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = os.getenv("CACHE_DIR", str(BASE_DIR / "cache"))

# Model configurations - For stella embeddings - currently using openAI
MODEL_CONFIG = {
    "name": "dunzhang/stella_en_400M_v5",
    "device": "cuda" if os.getenv("USE_GPU", "1") == "1" else "cpu",
    "cache_dir": CACHE_DIR,
}

# Milvus_configurations
MILVUS_CONFIG = {
    "uri": os.getenv("MILVUS_URI", "./database/handbooks.db"),
    "collection_name": {
        "openai": os.getenv("MILVUS_COLLECTION", "handbook_store"),
        "bge_m3_large_en_v1_5": os.getenv("MILVUS_COLLECTION_BGE", "handbook_store_bge"),
        "bge-m3": os.getenv("MILVUS_COLLECTION_BGE_M3", "handbook_store_bge_m3"),
    },
}

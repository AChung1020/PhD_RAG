# Global configurations
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

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
    "collection_name": os.getenv("MILVUS_COLLECTION", "handbook_store"),
}

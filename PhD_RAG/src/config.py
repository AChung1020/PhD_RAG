# Global configurations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = os.getenv('CACHE_DIR', str(BASE_DIR / 'cache'))

# Model configurations
MODEL_CONFIG = {
    'name': 'dunzhang/stella_en_400M_v5',
    'device': 'cuda' if os.getenv('USE_GPU', '1') == '1' else 'cpu',
    'batch_size': int(os.getenv('BATCH_SIZE', 32)),
    'cache_dir': CACHE_DIR
}

# Milvus_configurations
MILVUS_CONFIG = {
    'uri': os.getenv('MILVUS_URI', './handbook_data.db'),
    'collection_name': os.getenv('MILVUS_COLLECTION', 'document_store')
}

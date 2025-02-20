from typing import List

import torch
from FlagEmbedding import BGEM3FlagModel
from langchain.embeddings.base import Embeddings
from sklearn.preprocessing import normalize


class BGE_M3_Embeddings(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)  # .to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            embeddings = self.model.encode(
                texts, batch_size=12, max_length=8192  # Adjust batch size as needed
            )["dense_vecs"]

        return normalize(embeddings).tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_documents([query])[0]

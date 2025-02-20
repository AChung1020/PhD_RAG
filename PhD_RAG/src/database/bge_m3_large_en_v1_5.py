from typing import List

import torch
from langchain.embeddings.base import Embeddings
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer


class BGE_M3_Large_en_v1_5_Embeddings(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            outputs = self.model(**inputs)
            # sentence_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = (
                outputs.last_hidden_state[:, 0, :].cpu().numpy()
            )  # CLS token embedding
            # embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return normalize(embeddings).tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_documents([query])[0]

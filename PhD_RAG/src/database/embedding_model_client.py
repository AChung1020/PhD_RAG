from typing import List
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from PhD_RAG.src.config import MODEL_CONFIG
import torch


class StellaEmbeddings(Embeddings):
    def __init__(self, model_name: str = MODEL_CONFIG['name']):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=MODEL_CONFIG['cache_dir']
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=MODEL_CONFIG['cache_dir']
        )

        self.device = torch.device(MODEL_CONFIG['device'])
        self.model = self.model.to(self.device)
        self.model.eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings: List[List[float]] = []

        # Could batch this for better performance
        for text in texts:
            embedding: List[float] = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single piece of text"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get the mean of the last hidden state for a single vector
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

        return embedding.tolist()
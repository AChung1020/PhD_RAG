import os
from typing import List

import torch
from langchain.embeddings.base import Embeddings
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer

from PhD_RAG.src.config import MODEL_CONFIG


class StellaEmbeddings(Embeddings):
    def __init__(self, model_name: str = MODEL_CONFIG["name"], vector_dim: int = 1024):
        # Configure device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=MODEL_CONFIG["cache_dir"], trust_remote_code=True
        )

        self.model = (
            AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=MODEL_CONFIG["cache_dir"],
                use_memory_efficient_attention=False,
                unpad_inputs=False,
            )
            .to(self.device)
            .eval()
        )

        # Setup vector linear layer with specific dimension
        self.vector_linear = torch.nn.Linear(
            in_features=self.model.config.hidden_size,
            out_features=vector_dim,
        )

        # Load vector linear weights from the correct directory structure
        vector_linear_path = os.path.join(
            "PhD_RAG/cache/models--dunzhang--stella_en_400M_v5",  # Using model_name directly as the directory
            f"2_Dense_{vector_dim}/pytorch_model.bin",
        )

        if not os.path.exists(vector_linear_path):
            raise ValueError(f"Vector linear weights not found at {vector_linear_path}")

        vector_linear_dict = {
            k.replace("linear.", ""): v
            for k, v in torch.load(
                vector_linear_path, map_location=torch.device(self.device)
            ).items()
        }
        self.vector_linear.load_state_dict(vector_linear_dict)
        self.vector_linear.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding="longest",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            attention_mask = inputs["attention_mask"]
            last_hidden_state = self.model(**inputs)[0]

            last_hidden = last_hidden_state.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
            vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            vectors = normalize(self.vector_linear(vectors).cpu().numpy())

            return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a query with the instruction prompt"""
        query_prompt = f"Instruct: Retrieve semantically similar text.\nQuery: {query}"
        return self.embed_documents([query_prompt])[0]

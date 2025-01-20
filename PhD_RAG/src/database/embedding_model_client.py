from typing import List
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from PhD_RAG.src.config import MODEL_CONFIG
import torch
from sklearn.preprocessing import normalize
import os


class StellaEmbeddings(Embeddings):
    def __init__(self, model_name: str = MODEL_CONFIG['name']):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=MODEL_CONFIG['cache_dir'],
            trust_remote_code=True
        )

        # Configure device - try MPS first, then CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Disable memory efficient attention
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=MODEL_CONFIG['cache_dir'],
            use_memory_efficient_attention=False,
            unpad_inputs=False
        )

        self.model = self.model.to(self.device)

        # Setup vector linear layer
        vector_dim = 1024
        self.vector_linear = torch.nn.Linear(
            in_features=self.model.config.hidden_size,
            out_features=vector_dim
        ).to(self.device)

        # Load vector linear weights if they exist
        vector_linear_path = os.path.join(
            MODEL_CONFIG['cache_dir'],
            f"2_Dense_{vector_dim}/pytorch_model.bin"
        )
        if os.path.exists(vector_linear_path):
            vector_linear_dict = {
                k.replace("linear.", ""): v
                for k, v in torch.load(vector_linear_path, map_location=self.device).items()
            }
            self.vector_linear.load_state_dict(vector_linear_dict)

        self.model.eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using batching"""
        batch_size = 4  # Adjust based on your GPU memory
        all_embeddings = []

        # Set model to eval mode
        self.model.eval()

        # Set deterministic behavior
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch,
                    padding="longest",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                attention_mask = inputs["attention_mask"]
                last_hidden_state = self.model(**inputs)[0]

                # Mask and pool
                last_hidden = last_hidden_state.masked_fill(
                    ~attention_mask[..., None].bool(),
                    0.0
                )
                vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

                # Project and normalize
                vectors = self.vector_linear(vectors)
                vectors = normalize(vectors.cpu().numpy())

                all_embeddings.extend(vectors.tolist())

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query with the instruction prompt"""
        query_prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
        print(self.embed_documents([query_prompt + text])[0])
        return self.embed_documents([query_prompt + text])[0]

from langchain_milvus import Milvus
from embedding_model_client import StellaEmbeddings
from PhD_RAG.src.config import MODEL_CONFIG, MILVUS_CONFIG


def setup_vectorstore(texts):
    # Initialize Stella embeddings
    embeddings = StellaEmbeddings(MODEL_CONFIG['name'])

    # Create Milvus Lite vector store
    vector_store = Milvus.from_texts(
        texts=texts,
        embedding=embeddings,
        connection_args={"uri": MILVUS_CONFIG['uri']},
    )

    return vector_store

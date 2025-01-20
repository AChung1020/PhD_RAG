from langchain_milvus import Milvus
from PhD_RAG.src.database.embedding_model_client import StellaEmbeddings
from PhD_RAG.src.config import MODEL_CONFIG, MILVUS_CONFIG
from langchain_core.documents import Document


def setup_vectorstore(documents: list[Document]) -> Milvus:
    # Initialize Stella embeddings
    embeddings = StellaEmbeddings(MODEL_CONFIG['name'])

    # Create Milvus Lite vector store
    vector_store = Milvus.from_documents(
        documents=documents,
        embedding=embeddings,
        connection_args={"uri": MILVUS_CONFIG['uri']},
    )

    return vector_store

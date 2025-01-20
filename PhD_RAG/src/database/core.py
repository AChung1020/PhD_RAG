from langchain_milvus import Milvus
from PhD_RAG.src.database.embedding_model_client import StellaEmbeddings
from PhD_RAG.src.config import MODEL_CONFIG, MILVUS_CONFIG
from langchain_core.documents import Document


def setup_vectorstore(documents: list[Document], uuids: list[str]) -> Milvus:
    embeddings = StellaEmbeddings(MODEL_CONFIG['name'])

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": MILVUS_CONFIG['uri']},
        collection_name=MILVUS_CONFIG['collection_name'],
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 8}
        }
    )

    # Add error handling for document insertion
    try:
        vector_store.add_documents(documents, ids=uuids)
    except Exception as e:
        raise Exception(f"Failed to add documents to vector store: {str(e)}")

    return vector_store

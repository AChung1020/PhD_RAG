from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_milvus import Milvus
from PhD_RAG.src.database.embedding_model_client import StellaEmbeddings
from PhD_RAG.src.config import MODEL_CONFIG, MILVUS_CONFIG
from langchain_core.documents import Document


def setup_vectorstore(documents: list[Document], uuids: list[str]) -> Milvus:
    embeddings: StellaEmbeddings = StellaEmbeddings(MODEL_CONFIG['name'])

    vector_store: Milvus = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": MILVUS_CONFIG['uri']},
        collection_name=MILVUS_CONFIG['collection_name'],
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 8}
        }
    )

    try:
        vector_store.add_documents(documents, ids=uuids)
    except Exception as e:
        raise Exception(f"Failed to add documents to vector store: {str(e)}")

    return vector_store


def chunk_documents(doc_path: str) -> list[Document]:
    # read file
    with open(doc_path, encoding='utf-8') as f:
        markdown_file: str = f.read()

    # define split points
    headers_to_split_on: list[tuple[str, str]] = [("#", "Header_1")]

    # split text
    splitter: MarkdownHeaderTextSplitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    splits: list[Document] = splitter.split_text(markdown_file)

    return splits

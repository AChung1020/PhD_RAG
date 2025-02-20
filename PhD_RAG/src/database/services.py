from typing import Literal

from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

from PhD_RAG.src.config import MILVUS_CONFIG, OPENAI_API_KEY
from PhD_RAG.src.database.bge_m3 import BGE_M3_Embeddings
from PhD_RAG.src.database.bge_m3_large_en_v1_5 import \
    BGE_M3_Large_en_v1_5_Embeddings


def setup_vectorstore(
    documents: list[Document],
    uuids: list[str],
    model_type: Literal["openai", "bge-m3", "bge_m3_large_en_v1_5"] = "openai",
) -> Milvus:
    """
    Set up the vectorstore with given documents.

    Parameters
    ----------
    documents : list[Document]
        The documents to be added to the vectorstore.
    uuids : list[str]
        The unique identifiers corresponding to the documents.
    model_type : Literal["openai", "bge-m3", "bge_m3_large_en_v1_5"]
        Choose between "openai", "bge-m3", and "bge_m3_large_en_v1_5" embeddings

    Returns
    -------
    Milvus
        An initialized Milvus vectorstore instance containing the documents.
    """
    if model_type == "bge_m3_large_en_v1_5":
        embeddings: BGE_M3_Large_en_v1_5_Embeddings = BGE_M3_Large_en_v1_5_Embeddings()
        collection_name = MILVUS_CONFIG["collection_name"]["bge_m3_large_en_v1_5"]
    elif model_type == "bge-m3":
        embeddings: BGE_M3_Embeddings = BGE_M3_Embeddings()
        collection_name = MILVUS_CONFIG["collection_name"]["bge-m3"]
    elif model_type == "openai-large":
        embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
            model="text-embedding-3-large", api_key=OPENAI_API_KEY
        )
        collection_name = MILVUS_CONFIG["collection_name"]["openai-large"]
    elif model_type == "openai-small":
        embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key=OPENAI_API_KEY
        )
        collection_name = MILVUS_CONFIG["collection_name"]["openai-small"]

    vector_store: Milvus = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": MILVUS_CONFIG["uri"]},
        collection_name=collection_name,
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 8},
        },
    )

    try:
        vector_store.add_documents(documents, ids=uuids)
    except Exception as e:
        raise Exception(f"Failed to add documents to vector store: {str(e)}")

    return vector_store


def chunk_documents(doc_path: str) -> list[Document]:
    """
    Chunk the documents into smaller, more manageable pieces.

    Parameters
    ----------
    doc_path : str
        The path to the document in your local directory.

    Returns
    -------
    chunked_docs : list[Document]
        A list of chunked documents ready for further processing.

    Notes
    -----
    This function breaks down large documents into smaller chunks
    to improve processing and retrieval efficiency.
    """
    # read file
    with open(doc_path, encoding="utf-8") as f:
        markdown_file: str = f.read()

    # define split points
    headers_to_split_on: list[tuple[str, str]] = [("#", "Header_1")]

    # split text
    splitter: MarkdownHeaderTextSplitter = MarkdownHeaderTextSplitter(
        headers_to_split_on, strip_headers=False
    )
    splits: list[Document] = splitter.split_text(markdown_file)

    return splits

from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

from PhD_RAG.src.config import MILVUS_CONFIG, OPENAI_API_KEY


def setup_vectorstore(documents: list[Document], uuids: list[str]) -> Milvus:
    """
    Set up the vectorstore with given documents.

    Parameters
    ----------
    documents : list[Document]
        The documents to be added to the vectorstore.
    uuids : list[str]
        The unique identifiers corresponding to the documents.

    Returns
    -------
    Milvus
        An initialized Milvus vectorstore instance containing the documents.
    """
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
        model="text-embedding-3-large", api_key=OPENAI_API_KEY
    )

    vector_store: Milvus = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": MILVUS_CONFIG["uri"]},
        collection_name=MILVUS_CONFIG["collection_name"],
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

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_milvus import Milvus
from PhD_RAG.src.config import MODEL_CONFIG, MILVUS_CONFIG, OPENAI_API_KEY
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def setup_vectorstore(documents: list[Document], uuids: list[str]) -> Milvus:
    """
    Set up the vectorstore.
    :param documents: list[Document]: The documents to add to the vectorstore.
    :param uuids: list[str]: The uuids of the documents.
    :return: Milvus instance: The vectorstore instance.
    """
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

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
    """
    Chunk the documents.
    :param doc_path: str: The path to the document in your local directory
    :return: chunked_docs: list[Document]: The chunked documents.
    """
    # read file
    with open(doc_path, encoding='utf-8') as f:
        markdown_file: str = f.read()

    # define split points
    headers_to_split_on: list[tuple[str, str]] = [("#", "Header_1")]

    # split text
    splitter: MarkdownHeaderTextSplitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    splits: list[Document] = splitter.split_text(markdown_file)

    return splits

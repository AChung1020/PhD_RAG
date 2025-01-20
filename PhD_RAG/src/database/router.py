from fastapi import APIRouter, FastAPI
from logging import getLogger

from langchain_milvus import Milvus
from collections import defaultdict
import tiktoken
from uuid import uuid4

from PhD_RAG.src.database.utils import chunk_documents
from PhD_RAG.src.database.core import setup_vectorstore
from langchain_core.documents import Document
from PhD_RAG.src.config import MILVUS_CONFIG, MODEL_CONFIG
from PhD_RAG.src.database.embedding_model_client import StellaEmbeddings


router: APIRouter = APIRouter()
logger = getLogger(__name__)


@router.post("/create_vectorstore")
async def create_vectorstore():
    logger.info("Creating vectorstore...")

    document_paths = ['Data/MD_handbooks/laney-graduate-studies-handbook-cleaned.md', 'Data/MD_handbooks/csi-phd-handbook-2024.pdf.md']

    chunked_docs: list[Document] = []
    # Process each document path
    for path in document_paths:
        logger.info(f"Processing document: {path}")
        res: list[Document] = chunk_documents(path)
        chunked_docs.extend(res)

    uuids: list[str] = [str(uuid4()) for _ in range(len(chunked_docs))]

    logger.info(f"Number of chunks: {len(chunked_docs)}")

    setup_vectorstore(chunked_docs, uuids)
    return {"message": "Vectorstore created"}


@router.post("/query_results")
async def query_results(query: str):
    embeddings = StellaEmbeddings(MODEL_CONFIG['name'])
    vector_store = Milvus(
        connection_args={"uri": MILVUS_CONFIG['uri']},
        embedding_function=embeddings,
        collection_name=MILVUS_CONFIG['collection_name'],
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    print(len(docs))

    return {"docs": docs}


@router.get("/chunk_size")
async def chunk_size():
    document_paths = ['Data/MD_handbooks/laney-graduate-studies-handbook-cleaned.md',
                      'Data/MD_handbooks/csi-phd-handbook-2024.pdf.md']

    chunked_docs: list[Document] = []
    token_counts = defaultdict(int)

    # Initialize tiktoken encoder, openai token estimation
    enc = tiktoken.get_encoding("cl100k_base")

    for path in document_paths:
        logger.info(f"Processing document: {path}")
        res: list[Document] = chunk_documents(path)
        chunked_docs.extend(res)

    # Count tokens for each chunk and group them into intervals
    for doc in chunked_docs:
        token_count: int = len(enc.encode(doc.page_content))
        interval = (token_count // 100) * 100
        if interval <= 1000:
            token_counts[interval] += 1

    # Create the histogram data
    histogram_data = []
    for i in range(0, 1100, 100):
        range_start = i
        range_end = i + 100
        count = token_counts[i]
        histogram_data.append({
            "total_docs": len(chunked_docs),
            "range": f"{range_start}-{range_end}",
            "count": count,
            "start": range_start,
            "end": range_end
        })

    return {"histogram_data": histogram_data}


def init_app(app: FastAPI) -> None:
    app.include_router(
        router, prefix="/database", tags=["database"],)

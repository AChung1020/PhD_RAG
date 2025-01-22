from fastapi import APIRouter, FastAPI, HTTPException
from logging import getLogger

from langchain_milvus import Milvus
from collections import defaultdict
import tiktoken
from tiktoken import Encoding
from uuid import uuid4

from PhD_RAG.src.database.services import chunk_documents
from PhD_RAG.src.database.services import setup_vectorstore
from langchain_core.documents import Document
from PhD_RAG.src.config import MILVUS_CONFIG, MODEL_CONFIG, OPENAI_API_KEY
from pymilvus import connections, utility
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

router: APIRouter = APIRouter()
logger = getLogger(__name__)


@router.put("/create_vectorstore")
async def create_vectorstore():
    logger.info("Creating vectorstore...")

    document_paths: list[str] = ['Data/MD_handbooks/laney-graduate-studies-handbook-cleaned.md',
                      'Data/MD_handbooks/csi-phd-handbook-2024.pdf.md']

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


@router.delete("/delete_vectorstore")
async def delete_vectorstore():
    try:
        # Ensure connection to Milvus
        connections.connect(uri=MILVUS_CONFIG['uri'])

        # Check if collection exists before trying to delete
        if utility.has_collection(MILVUS_CONFIG['collection_name']):
            utility.drop_collection(MILVUS_CONFIG['collection_name'])
            return {"message": f"Collection {MILVUS_CONFIG['collection_name']} successfully deleted"}
        else:
            return {"message": f"Collection {MILVUS_CONFIG['collection_name']} does not exist"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete collection: {str(e)}"
        )
    finally:
        # Clean up connection
        connections.disconnect("default")


@router.post("/query_results")
async def query_results(query: str):
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
    vector_store: Milvus = Milvus(
        connection_args={"uri": MILVUS_CONFIG['uri']},
        embedding_function=embeddings,
        collection_name=MILVUS_CONFIG['collection_name'],
    )

    results = vector_store.similarity_search_with_score(
        query, k=1
    )
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

    retriever: VectorStoreRetriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs: list[Document] = retriever.invoke(query)

    return {"docs": docs}


@router.get("/chunk_size")
async def chunk_size():
    document_paths: list['str'] = ['Data/MD_handbooks/laney-graduate-studies-handbook-cleaned.md',
                                   'Data/MD_handbooks/csi-phd-handbook-2024.pdf.md']

    chunked_docs: list[Document] = []
    token_counts: defaultdict = defaultdict(int)

    # Initialize tiktoken encoder, openai token estimation
    enc: Encoding = tiktoken.get_encoding("cl100k_base")

    for path in document_paths:
        logger.info(f"Processing document: {path}")
        res: list[Document] = chunk_documents(path)
        chunked_docs.extend(res)

    # Count tokens for each chunk and group them into intervals
    for doc in chunked_docs:
        token_count: int = len(enc.encode(doc.page_content))
        interval: int = (token_count // 100) * 100
        if interval <= 1000:
            token_counts[interval] += 1

    # Create the histogram data
    histogram_data: list = []
    for i in range(0, 1100, 100):
        range_start: int = i
        range_end: int = i + 100
        count: int = token_counts[i]
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
        router, prefix="/database", tags=["database"], )

from fastapi import APIRouter, FastAPI
from logging import getLogger
from PhD_RAG.src.database.utils import chunk_documents
from PhD_RAG.src.database.core import setup_vectorstore
from langchain_core.documents import Document

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

    logger.info(f"Number of chunks: {len(chunked_docs)}")

    setup_vectorstore(chunked_docs)

    return {"message": "Vectorstore created"}


def init_app(app: FastAPI) -> None:
    app.include_router(
        router, prefix="/database", tags=["database"],)

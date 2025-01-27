import logging

import anthropic
from fastapi import APIRouter, FastAPI, HTTPException
from langchain_core.documents import Document

from PhD_RAG.src.api.services import process_query
from PhD_RAG.src.config import settings
from PhD_RAG.src.database.router import query_results
from PhD_RAG.src.models import ChatRequest, ChatResponse

router = APIRouter()
logger = logging.getLogger(__name__)


client = anthropic.Anthropic(api_key=settings.claude_api_key)


@router.post("/chat", response_model=ChatResponse)
async def get_response(request: ChatRequest) -> ChatResponse:
    """
    Handle chatbot interactions by receiving a user query and generating a response.

    Parameters
    ----------
    request : ChatRequest
        A request object containing the user's query string.

    Returns
    -------
    ChatResponse
        A response object containing:
        - 'answer': A string representing the chatbot's response to the user's query,
          based on the retrieved documents.

    Raises
    ------
    HTTPException
        - 400: If the query is empty or invalid.
        - 500: If an error occurs during document retrieval or response generation.
    """
    query: str = request.query
    if not query:
        raise HTTPException(detail="Empty query", status_code=400)

    try:
        retrieved_context: dict[str, list[Document]] = await query_results(query)
        response: str = await process_query(query, retrieved_context)
    except Exception as e:
        raise HTTPException(detail=str(e), status_code=500)
    return ChatResponse(answer=response)


def init_app(app: FastAPI) -> None:
    app.include_router(
        router,
        prefix="/api/v1",
        tags=["chatbot"],
    )

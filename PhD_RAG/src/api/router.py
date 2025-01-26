import logging

import anthropic
from fastapi import APIRouter, FastAPI, HTTPException

from PhD_RAG.src.config import settings
from PhD_RAG.src.database.router import query_results
from PhD_RAG.src.models import ChatRequest, ChatResponse

router = APIRouter()
logger = logging.getLogger(__name__)


client = anthropic.Anthropic(api_key=settings.claude_api_key)


@router.post("/chat", response_model=ChatResponse)
async def get_response(request: ChatRequest):
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
    query = request.query
    if not query:
        raise HTTPException(detail="Empty query", status_code=400)

    try:
        retrieved_context = await query_results(query)
        response = await process_query(query, retrieved_context)
    except Exception as e:
        raise HTTPException(detail=str(e), status_code=500)
    return ChatResponse(answer=response)


async def process_query(query: str, retrieved_context: dict) -> str:
    context = "\n".join(doc.page_content for doc in retrieved_context["docs"])
    max_context_length = 4000
    if len(context) > max_context_length:
        context = context[:max_context_length]
        logger.warning("Context truncated due to length restrictions.")

    prompt = f"""
    Context: {context}
--------------------------------------
    User Query: {query}
--------------------------------------
    Based on the provided context, answer the user's query accurately. If the context lacks sufficient information, state that clearly.
    """

    print(f"Created Prompt: {prompt}")
    try:
        response = client.messages.create(
            model=settings.model_name,
            max_tokens=settings.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        raise ValueError(f"Failed to generate response: {e}")


def init_app(app: FastAPI) -> None:
    app.include_router(
        router,
        prefix="/api/v1",
        tags=["chatbot"],
    )

import os
import aiohttp
from fastapi import APIRouter, HTTPException, FastAPI
from PhD_RAG.src.models import ChatRequest, ChatResponse

router = APIRouter()

CLAUDE_API_KEY = os.getenv("CLAUD_API_KEY")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

@router.post("/chat", response_model=ChatResponse)
async def get_response(request: ChatRequest):
    """
    Handle chatbot interactions by receiving a query and returning an answer.
    """
    query = request.query
    if not query:
        raise HTTPException(detail="Empty query", status_code=400)

    response = process_query(query)
    return ChatResponse(answer=response)

async def process_query(query: str):
    headers = {
        'x-api-key': CLAUDE_API_KEY,
        'Content-Type': 'application/json',
        'anthropic-version': '2023-06-01',
    }

    payload = {
        'model': 'claude-3-5-sonnet-20241022',
        'max_tokens':300,
        'messages': [
            {'role':'user', 'content':query}
        ],
    }


    return "This is a placeholder response."


def init_app(app: FastAPI) -> None:
    app.include_router(
        router, prefix="/api/v1", tags=["chatbot"],
    )
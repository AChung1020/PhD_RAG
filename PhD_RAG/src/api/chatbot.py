
import anthropic
from fastapi import APIRouter, FastAPI, HTTPException

from PhD_RAG.src.config import settings
from PhD_RAG.src.models import ChatRequest, ChatResponse

router = APIRouter()

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


@router.post("/chat", response_model=ChatResponse)
async def get_response(request: ChatRequest):
    """
    Handle chatbot interactions by receiving a query and returning an answer.
    """
    query = request.query
    if not query:
        raise HTTPException(detail="Empty query", status_code=400)

    try:
        response = await process_query(query)
    except Exception as e:
        raise HTTPException(detail=str(e), status_code=500)
    return ChatResponse(answer=response)


async def process_query(query: str) -> str:
    context = ""
    prompt = f"""
        Context: {context}
    
        User Query: {query}
    
        Based on the provided context, answer the user's query accurately. If the context lacks sufficient information, state that clearly.
    """

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

from fastapi import APIRouter, HTTPException
from PhD_RAG.src.models import ChatRequest, ChatResponse

router = APIRouter()

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

def process_query(query):
    return "This is a placeholder response."
# global models
from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str


#     session id, user id..


class ChatResponse(BaseModel):
    answer: str
    # source documents, session id..

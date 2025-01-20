from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PhD_RAG.src.api import chatbot

app = FastAPI(title="PhD RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chatbot.router, prefix="/api/v1", tags=["chatbot"])

@app.get("/")
async def root():
    return {"message": "Hello World"}
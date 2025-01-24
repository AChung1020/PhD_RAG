
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PhD_RAG.src.database.router import init_app as init_database
from PhD_RAG.src.api import chatbot

app = FastAPI(title="PhD RAG Chatbot")


app = FastAPI()


def init_routers(app: FastAPI) -> None:
    """Initialize all routers"""
    init_database(app)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_routers(app)


app.include_router(chatbot.router, prefix="/api/v1", tags=["chatbot"])


@app.get("/")
async def root():
    return {"message": "Hello World"}

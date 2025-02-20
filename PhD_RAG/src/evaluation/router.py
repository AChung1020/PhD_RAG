import json

from fastapi import APIRouter, FastAPI
from langchain_core.documents import Document
import tqdm

from PhD_RAG.src.api.services import generate_qa_pairs
from PhD_RAG.src.database.router import query_results
from PhD_RAG.src.database.services import chunk_documents
from PhD_RAG.src.evaluation.services import (compute_map_at_k,
                                             compute_success_at_k)

router: APIRouter = APIRouter()


@router.post("/generate_qa")
async def generate_qa_dataset():
    """
    Generate QA pairs from the documents in vectorstore

    retrieves chunks from stored documents and create Q&A pairs

    Returns:
    -------
    """
    document_paths: list[str] = [
        "Data/MD_handbooks/laney-graduate-studies-handbook-cleaned.md",
        "Data/MD_handbooks/csi-phd-handbook-2024.pdf.md",
    ]
    chunked_docs: list[Document] = []

    # Process each document path
    for path in document_paths:
        chunks: list[Document] = chunk_documents(path)
        chunked_docs.extend(chunks)

    # Generate QA pairs with Claude
    qa_pairs: list[dict] = await generate_qa_pairs(chunked_docs)

    with open("Data/qa/new_claude_qa_pairs.json", "w") as f:
        json.dump(qa_pairs, f, indent=4)
    print("Q&A dataset saved!")


@router.post("/evaluate_retrieval")
async def evaluate_retrieval(k: int, model: str):
    """
    Evaluate the retrieval performance of the model at k
    :param k: number of documents to retrieve
    :param model: bge-m3, openai-small, openai-large

    :return:
    """
    eval_dataset = []
    with open("Data/qa/new_claude_qa_pairs.json", "r") as f:
        qa_pairs = json.load(f)

    # qa_pairs = qa_pairs[:5]
    from tqdm import tqdm
    for qa in tqdm(qa_pairs, desc="Evaluating retrieval"):
        query = qa["query"]
        pos_chunk = qa["pos"]

        response = await query_results(query, k, model)
        retrieved_docs = [doc.page_content for doc in response["docs"]]

        eval_dataset.append(
            {
                "query": query,
                "retrieved_docs": retrieved_docs,
                "pos_chunk": pos_chunk,
            }
        )

    map_k = compute_map_at_k(eval_dataset, k=k)
    success_k = compute_success_at_k(eval_dataset, k=k)

    print(f"MAP@{k}: {map_k:.4f}")
    print(f"Success@{k}: {success_k:.4f}")

    with open(f"{model}_retrieved_dataset_Kat{k}.json", "w") as f:
        json.dump(eval_dataset, f, indent=4)


def init_app(app: FastAPI) -> None:
    app.include_router(
        router,
        prefix="/evaluation",
        tags=["evaluation"],
    )

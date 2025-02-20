import json
import logging

import anthropic
from langchain_core.documents import Document

from PhD_RAG.src.config import settings
from PhD_RAG.src.evaluation.prompts import create_qa_prompt

logger = logging.getLogger(__name__)


client = anthropic.Anthropic(api_key=settings.claude_api_key)


async def generate_qa_pairs(doc_chunks: list[Document]) -> list[dict]:
    """
    Generate Q&A pairs from document chunks using LLM.

    Parameters
    ----------
    doc_chunks : list[Document]
        A list of document chunks from the handbook.

    Returns
    -------
    list[dict]
        A dataset containing Q&A pairs along with their reference chunks.
        [
            {"query": "...", "answer": "...", "pos": "..."},
            ...
        ]
    """
    qa_pairs = []
    # doc_chunks = doc_chunks[:5]

    print(len(doc_chunks))
    for chunk in doc_chunks:
        prompt: str = create_qa_prompt(chunk)
        try:
            response: anthropic.types.Message = client.messages.create(
                model=settings.model_name,
                max_tokens=settings.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            json_text = response.content[0].text.strip()
            try:
                print(json_text)

                qa_pair = json.loads(json_text)
                qa_pair["pos"] = chunk.page_content
                qa_pairs.append(qa_pair)

            except json.JSONDecodeError as e:
                print(f"JSON Error: {e} | Response: {json_text}")  # Debugging output
                continue  # Skip and proceed
            except Exception as e:
                print(f"Failed to generate response: {e}")
                continue  # Skip this chunk and move to the next

        except Exception as e:
            raise ValueError(f"Failed to generate response: {e}")
    return qa_pairs


async def process_query(
    query: str, retrieved_context: dict[str, list[Document]]
) -> str:
    """
    Processes a user query with given retrieved context
    and generating a response using Claude.

    Parameters
    ----------
    query : str
        The user's query to be processed.
    retrieved_context : Dict[str, List[Document]]
        The list of retrieved documents containing relevant content.

    Returns
    -------
    str
        The generated chatbot response.
    """
    context: str = "\n".join(doc.page_content for doc in retrieved_context["docs"])
    max_context_length: int = 4000
    if len(context) > max_context_length:
        context = context[:max_context_length]
        logger.warning("Context truncated due to length restrictions.")

    prompt: str = f"""
    Context: {context}
--------------------------------------
    User Query: {query}
--------------------------------------
    Based on the provided context, answer the user's query accurately. If the context lacks sufficient information, state that clearly.
    """

    print(f"Created Prompt: {prompt}")
    try:
        response: anthropic.types.Message = client.messages.create(
            model=settings.model_name,
            max_tokens=settings.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        raise ValueError(f"Failed to generate response: {e}")

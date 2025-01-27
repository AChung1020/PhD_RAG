import logging

import anthropic
from langchain_core.documents import Document

from PhD_RAG.src.config import settings

logger = logging.getLogger(__name__)


client = anthropic.Anthropic(api_key=settings.claude_api_key)


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

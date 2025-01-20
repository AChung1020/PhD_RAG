from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
import torch


def chunk_documents(doc_path: str) -> list[Document]:
    # read file
    with open(doc_path, encoding='utf-8') as f:
        markdown_file: str = f.read()

    # define split points
    headers_to_split_on: list[tuple[str, str]] = [("#", "Header_1")]

    # split text
    splitter: MarkdownHeaderTextSplitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    splits: list[Document] = splitter.split_text(markdown_file)

    return splits

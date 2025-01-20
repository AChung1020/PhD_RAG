from langchain_text_splitters import MarkdownHeaderTextSplitter


def chunk_documents(doc_path: str):
    headers_to_split_on = [("#", "Header 1")]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

    splits = splitter.split_text(doc_path)

"""
Heading-aware chunking for the Nexora handbooks.

Replace your existing chunk_document() in app/services/ingestion.py with this function
(and add the imports at the top of that file). It has the same signature:
chunk_document(list_of_documents) -> list_of_chunk_documents.

Why: a plain character splitter can cut a section in half or bury one short fact
(e.g. "at least 14 characters") inside a long chunk. Splitting on the ## headings keeps
each policy together, and prefixing every chunk with its section title gives the
embedding model the words a user is likely to search for ("Password Reset Policy").
"""
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def chunk_document(docs, chunk_size=700, chunk_overlap=100):
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = []
    for doc in docs:
        for section in header_splitter.split_text(doc.page_content):
            section.metadata = {**doc.metadata, **section.metadata}
            for piece in size_splitter.split_documents([section]):
                path = " > ".join(
                    piece.metadata[k] for k in ("h1", "h2", "h3") if piece.metadata.get(k)
                )
                if path:
                    piece.page_content = f"[Section: {path}]\n{piece.page_content}"
                chunks.append(piece)
    return chunks

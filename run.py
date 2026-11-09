from app.services.ingestion import load_file , chunk_document
from pathlib import Path
from app.rag.vectorstore import add_documents

docs = load_file(Path("data/sample_kb/company_it_handbook.md"))
docs2 = chunk_document(docs)

add_documents(docs2)
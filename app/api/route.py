from pathlib import Path
from fastapi import APIRouter, UploadFile, File,  HTTPException , Header
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents
from app.services.ingestion import load_file, chunk_document, SUPPORTED_FILE_TYPES
from app.services.audit import write_audit

router = APIRouter(prefix="/api", tags=["RAG"])
settings = get_settings()

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000, description="The question to ask the RAG system")

@router.get("/health", summary="Health Check", description="Check the health of the RAG service")
def health_check():
    return {"status": "ok" , "service": settings.app_name}

@router.post("/chat", summary="Chat with the RAG system", description="Ask a question to the RAG system")
def chat(payload: ChatRequest):
    try:
        result = ask(payload.question)
        write_audit(payload.question, result["source_used"] , result.get("trace", []))
        return {"answer": result["answer"],
                "source_used": result["source_used"],
                "trace": result.get("trace", []),
                "citation": result.get("citation", []),
                "rewritten_query": result.get("rewritten_query", payload.question),
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_FILE_TYPES:
        raise HTTPException(status_code=400, detail=f"Supported: {', '.join(sorted(SUPPORTED_FILE_TYPES))}")
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename).name
    dest.write_bytes(await file.read())
    docs = load_file(dest)
    chunks = chunk_document(docs)
    ids = add_documents(chunks)
    return {"message": "Document indexed", "file": dest.name, "chunks": len(chunks), "ids_created": len(ids)}
import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag_pipeline import ingest_pdf, build_qa_chain

app = FastAPI(title="RAG Chatbot Workshop")

# --- Storage Setup ---
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "conversations.json"
DATA_DIR.mkdir(exist_ok=True)

# Configuration CORS (Cross-Origin Resource Sharing)
# Allows the frontend (if we build one) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models (Schemas) ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    doc_id: str
    question: str
    history: Optional[List[ChatMessage]] = None

class Source(BaseModel):
    page_number: Optional[int]
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[Source]] = None

# --- Helper Functions ---

def save_interaction(doc_id: str, question: str, answer: str):
    """Save the chat interaction to a JSON file."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "doc_id": doc_id,
        "question": question,
        "answer": answer,
    }
    
    conversations = []
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                conversations = json.load(f)
        except json.JSONDecodeError:
            conversations = []
            
    conversations.append(record)
    
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)

# --- Endpoints ---

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Endpoint to upload and ingest a PDF."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF.")
        
    # Save file temporarily
    temp_dir = "tmp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Call our Azure/Pinecone pipeline
        doc_id = ingest_pdf(file_path)
    except Exception as e:
        # Note: This will fail if AZURE_EMBEDDING_DEPLOYMENT is empty
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
            
    return {"doc_id": doc_id, "message": "PDF processed successfully"}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_doc(request: ChatRequest):
    """Endpoint to ask a question about a specific document."""
    try:
        qa_chain = build_qa_chain(request.doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building QA chain: {e}")
        
    # Format history for the prompt
    history_text = ""
    if request.history:
        for msg in request.history:
            prefix = "User" if msg.role == "user" else "Assistant"
            history_text += f"{prefix}: {msg.content}\n"
            
    question_with_history = history_text + f"User: {request.question}"
    
    # Run the chain
    result = qa_chain.invoke(question_with_history)
    
    answer = result["answer"]
    
    # Extract sources
    sources = []
    for doc in result.get("source_documents", []):
        sources.append(
            Source(
                page_number=doc.metadata.get("page_number"),
                snippet=doc.page_content[:200] + "...",
            )
        )
        
    # Save interaction
    try:
        save_interaction(request.doc_id, request.question, answer)
    except Exception as e:
        print(f"Error saving interaction: {e}")

    return ChatResponse(answer=answer, sources=sources)
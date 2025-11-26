"""RAG pipeline implemented with LangChain components using Azure OpenAI."""
from __future__ import annotations
import logging
import time
import uuid
from typing import Optional

# LangChain & Community Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

# --- CRITICAL CHANGE: Importing Azure classes ---
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# Import variables from our config.py
from app.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_CHAT_DEPLOYMENT,
    AZURE_EMBEDDING_DEPLOYMENT,
    PINECONE_API_KEY,
    PINECONE_DIMENSION,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

print(f"DEBUG: AZURE_EMBEDDING_DEPLOYMENT = {AZURE_EMBEDDING_DEPLOYMENT}")
print(f"DEBUG: PINECONE_INDEX_NAME = {PINECONE_INDEX_NAME}")

# 1. Verification of Keys
if not AZURE_OPENAI_API_KEY:
    raise RuntimeError("AZURE_OPENAI_API_KEY is not configured.")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not configured.")

# 2. Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

def _ensure_pinecone_index() -> None:
    """Create Pinecone index if it doesn't exist."""
    existing_indexes = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME in existing_indexes:
        return
    
    logger.info("Creating Pinecone index '%s'...", PINECONE_INDEX_NAME)
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=PINECONE_DIMENSION,
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
    )
    
    # Wait for index to be ready
    while True:
        idx = pc.describe_index(PINECONE_INDEX_NAME)
        if idx.status["ready"]:
            break
        time.sleep(1)

_ensure_pinecone_index()

# 3. Initialize Azure Models
# Note: This will fail if AZURE_EMBEDDING_DEPLOYMENT is empty in .env
embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
)

llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_CHAT_DEPLOYMENT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    add_start_index=True,
)

# 4. Ingestion Function (PDF -> Pinecone)
def ingest_pdf(file_path: str, doc_id: Optional[str] = None) -> str:
    """Load PDF, split, and store vectors in Pinecone."""
    print(f"--- STARTING PDF LOAD: {file_path} ---")
    try:
        document_id = doc_id or str(uuid.uuid4())
        
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        print(f"Loaded {len(pages)} pages from PDF.")
        
        # Add metadata for filtering later
        for i, page in enumerate(pages, start=1):
            page.metadata["doc_id"] = document_id
            page.metadata["page_number"] = page.metadata.get("page", i)
            
        chunks = text_splitter.split_documents(pages)
        print(f"Created {len(chunks)} text chunks.")
        
        # Ensure every chunk has the doc_id
        for chunk in chunks:
            chunk.metadata.setdefault("doc_id", document_id)
            
        print("--- CONNECTING TO AZURE EMBEDDINGS ---")
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
            namespace=None,
        )
        print("--- UPLOAD COMPLETE ---")
        
        return document_id
    except Exception as e:
        logger.error(f"Error ingesting PDF: {e}")
        raise e

# 5. Retrieval Helper
def get_retriever_for_doc(doc_id: str):
    """Create a retriever specific to one document ID."""
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )
    
    return vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {"doc_id": {"$eq": doc_id}},
        }
    )

def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

DEFAULT_SYSTEM_PROMPT = (
    "You are a domain expert assistant that answers questions using the "
    "provided context. If the answer is not contained in the context, "
    "respond with 'I could not find that in the document.'"
)

# 6. Build the Chat Chain
def build_qa_chain(doc_id: str) -> RunnableParallel:
    """Build the RAG chain."""
    retriever = get_retriever_for_doc(doc_id)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    
    answer_chain = (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
            "system_prompt": lambda _: DEFAULT_SYSTEM_PROMPT,
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return RunnableParallel(
        answer=answer_chain,
        source_documents=retriever,
    )
# src/api/main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.vector.langchain_faiss import FAISSVectorStore
from src.rag.rag_system import RAGSystem

@asynccontextmanager
async def lifespan(app: FastAPI):
    index_path = os.getenv("PERSIST_PATH", "data/processed/faiss_index")
    embed_provider = os.getenv("EMBED_PROVIDER", "huggingface")
   
    vector_store = FAISSVectorStore(embed_provider)
    vector_store.load_index(index_path)
    rag = RAGSystem(vector_store)

    app.state.vector_store = vector_store
    app.state.rag = rag
    print("✅ Pipeline RAG prêt")

    try:
        yield
    finally:
        app.state.vector_store = None
        app.state.rag = None
        print("👋 Arrêt propre")

app = FastAPI(
    title="RAG Events API",
    description="API d'interrogation RAG (événements culturels)",
    version="1.0.0",
    lifespan=lifespan,   # ⬅️ remplace on_event
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# importe les routes après la création de app
from src.api.controllers.ask_controller import router as ask_router
from src.api.controllers.rebuild_controller import router as rebuild_router
app.include_router(ask_router)
app.include_router(rebuild_router)

@app.get("/health")
def health():
    return {"status": "ok"}

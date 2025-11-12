from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.rag.rag_system import RAGSystem
from src.api.shemas.rebuild_out import RebuildOut
from src.api.shemas.doc_item import DocItem
from src.vector.langchain_faiss import FAISSVectorStore

router = APIRouter(tags=["RAG"])



class RebuildIn(BaseModel):
    docs: List[DocItem] = Field(
        ...,
        description="Liste de documents/segments à indexer (obligatoire)."
    )
    persist_path: str = Field(
        default="data/processed/faiss_index",
        description="Chemin de persistance de l'index FAISS."
    )
    embedding_provider: str = Field(
        default="mistral",
        description="Provider d'embeddings : 'mistral' ou 'huggingface'."
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "docs": [
                {
                    "text": "Concert Jazz à Paris, 15/11/2025, Salle Pleyel.",
                    "metadata": {"category": "jazz", "date_start": "2025-11-15", "url": "https://exemple/jazz"}
                },
                {
                    "text": "Expo photo à Lyon, 20/11/2025.",
                    "metadata": {"category": "exposition", "date_start": "2025-11-20"}
                }
            ],
            "persist_path": "data/processed/faiss_index",
        }]
    }}

@router.post(
    "/rebuild",
    response_model=RebuildOut,
    status_code=status.HTTP_200_OK,
    summary="(Re)construire la base vectorielle FAISS",
    description=(
        "Reconstruit l'index FAISS à partir d'une liste de documents "
    ),
    responses={
        200: {"description": "Index reconstruit et sauvegardé."},
        400: {"description": "Mauvaise requête (docs vides)."},
        500: {"description": "Erreur serveur lors de la (re)construction de l'index."},
    },
)
def rebuild_index(payload: RebuildIn, request: Request) -> RebuildOut:
    if not payload.docs:
        raise HTTPException(status_code=400, detail="`docs` ne peut pas être vide.")

    try:
        vs = FAISSVectorStore(
            embedding_provider=payload.embedding_provider,
        )

        chunks = [
            {"text": d.text, **(d.metadata or {})}
            for d in payload.docs
        ]

        vs.create_index(chunks)
        vs.save_index(payload.persist_path)

        rag_new = RAGSystem(vs)
        request.app.state.vector_store = vs
        request.app.state.rag = rag_new

        return RebuildOut(
            ok=True,
            count=len(chunks),
            index_path=payload.persist_path,
            created_at=datetime.utcnow(),
            provider=payload.embedding_provider,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

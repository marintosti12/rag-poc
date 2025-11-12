from datetime import datetime
from typing import List, Optional, Any

from fastapi import APIRouter, HTTPException, Request, status

from src.api.shemas.ask_in import AskIn
from src.api.shemas.ask_out import AskOut
from src.rag.rag_system import RAGSystem
from src.vector.langchain_faiss import FAISSVectorStore

router = APIRouter(tags=["RAG"])


@router.post(
    "/ask",
    response_model=AskOut,
    status_code=status.HTTP_200_OK,
    summary="Poser une question au système",
    description=(
        "Retourne une réponse **augmentée** (RAG) à partir de la question fournie.\n\n"
        "**Remarques**\n"
        "- `k` contrôle le nombre de passages récupérés.\n"
    ),
    responses={
        200: {
            "description": "Réponse générée avec les contextes utilisés.",
            "content": {
                "application/json": {
                    "example": {
                        "answer": "Réponse du système",
                    }
                }
            },
        },
        400: {"description": "Mauvaise requête (question vide, index non initialisé, etc.)."},
        500: {"description": "Erreur serveur lors de la génération."},
    },
)
def ask_rag(payload: AskIn, request: Request) -> AskOut:
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    try:
        rag = getattr(request.app.state, "rag", None)
        if rag is None:
            raise HTTPException(503, "Pipeline non chargé")

        result = rag.query(question=question, k=payload.k)
        
        return AskOut(
            answer=result.get("answer", ""),
        )

    except HTTPException as j:
        raise
    except Exception as e:
        print(e)
        msg = str(e)
        if "non initialisé" in msg.lower() or "not initialized" in msg.lower():
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=500, detail=msg)

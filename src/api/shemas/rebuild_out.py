from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class RebuildOut(BaseModel):
    ok: bool = Field(..., description="Statut de l'opération.")
    count: int = Field(..., description="Nombre d'items indexés.")
    index_path: str = Field(..., description="Chemin de l'index persistant.")
    created_at: datetime = Field(..., description="Horodatage UTC de fin d'opération.")
    provider: str = Field(..., description="Provider d'embeddings utilisé.")

    model_config = {"json_schema_extra": {
        "examples": [{
            "ok": True,
            "count": 2,
            "index_path": "data/processed/faiss_index",
            "created_at": "2025-11-03T10:22:45.123456+00:00",
            "provider": "mistral",
        }]
    }}


from typing import Optional
from pydantic import BaseModel, Field


class AskIn(BaseModel):
    question: str = Field(..., description="Question utilisateur.")
    k: int = Field(3, ge=1, le=20, description="Nombre de résultats.")
    model_config = {"json_schema_extra": {
        "examples": [{
            "question": "Trouve moi des évenements de jazz ?",
            "k": 3,
        }]
    }}
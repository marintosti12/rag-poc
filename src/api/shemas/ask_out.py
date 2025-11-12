from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class AskOut(BaseModel):
    answer: str = Field(..., description="Réponse générée par le système.")

    model_config = {"json_schema_extra": {
        "examples": [{
            "answer": "Réponse du système.",
        }]
    }}
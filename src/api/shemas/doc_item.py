from typing import Optional
from pydantic import BaseModel, Field


class DocItem(BaseModel):
    text: str = Field(..., description="Contenu textuel du document (chunk).")
    metadata: Optional[dict] = Field(
        default=None,
        description="Métadonnées associées (titre, date, url, etc.)."
    )

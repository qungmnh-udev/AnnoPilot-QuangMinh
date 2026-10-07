from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ScoringSettings(BaseModel):
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)

from pydantic import BaseModel
from typing import Optional, Dict
import uuid

class Chunk(BaseModel):
    id: str
    text: str
    source: str
    metadata: Optional[Dict] = None

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())

class Section(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict] = None 
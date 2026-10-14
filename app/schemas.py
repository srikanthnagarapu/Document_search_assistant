from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentCreate(BaseModel):
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full text of the document")
    department: Optional[str] = Field(None, description="Department name, e.g., Engineering")
    role: Optional[str] = Field(None, description="Role required to view the doc, e.g., manager")
    tags: Optional[List[str]] = Field(default_factory=list, description="List of tags for filtering")
    version: Optional[int] = Field(1, description="Document version, auto‑incremented on updates")

class DocumentOut(BaseModel):
    id: int
    title: str
    department: Optional[str]
    role: Optional[str]
    tags: List[str]
    version: int
    created_at: str
    updated_at: Optional[str]

class SearchResult(BaseModel):
    id: int
    title: str
    snippet: str
    score: float
    department: Optional[str]
    role: Optional[str]
    tags: List[str]
class AnswerResult(BaseModel):
    answer: str
    sources: List[SearchResult] = []


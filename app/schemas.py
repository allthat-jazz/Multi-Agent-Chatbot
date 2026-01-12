from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=20)

class AskResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[dict] = []
    tools_used: list[str] = []
    error: str | None = None

class SessionInfo(BaseModel):
    session_id: str
    title: str

class CreateSessionResponse(BaseModel):
    session_id: str
    title: str

class ReindexResponse(BaseModel):
    ok: bool
    docs: int
    chunks: int

class UploadResponse(BaseModel):
    ok: bool
    saved: list[dict]
    reindex: ReindexResponse

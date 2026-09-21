from fastapi import FastAPI
from pydantic import BaseModel, Field

from .structured_graph import build_graph


app = FastAPI(
    title="Zepto Support Assistant",
    description="A local policy-grounded GenAI support assistant.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    query: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


graph = build_graph()


@app.get("/")
def root():
    return {
        "service": "Zepto Support Assistant",
        "status": "running",
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = graph.invoke(
        {
            "query": request.query,
        }
    )

    response = result["response"]

    return AskResponse(
        answer=response.answer,
        sources=response.sources,
        confidence=response.confidence,
    )
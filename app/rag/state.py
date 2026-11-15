from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class RouteDecision(BaseModel):
    route: Literal["kb", "external", "direct"] = Field(
        description=(
            "kb: questions about the company, its employees, policies, IT or HR. "
            "external: third-party products, vendors or public information. "
            "direct: greetings and small talk only."
        )
    )


class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(
        description="good: the evidence contains what is needed to answer. weak: key facts are missing."
    )


class AgentState(TypedDict):
    question: str
    current_query: str
    kb_docs: List[Document]
    web_results: str
    web_grade: str
    kb_grade: str
    answer: str
    source_used: str
    retry_count: int
    trace: List[str]
    citations: List[dict]
from typing import TypedDict, List
from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    question: str
    chat_history: list

    need_retrieval: bool
    clarification_needed: bool
    classification_reason: str

    rewritten_query: str
    retry_count: int

    retrieved_docs: List[Document]

    context_relevant: bool
    context_score: float
    context_reason: str

    version_conflict_found: bool
    latest_docs: List[Document]
    conflict_summary: str

    answer: str

    is_answer_supported: bool
    faithfulness_score: float
    unsupported_claims: List[str]

    final_answer: str

import json
import re
from typing import List
from langchain_core.documents import Document


def extract_json(text: str) -> dict:
    """Safely extract JSON from an LLM response."""
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return {}

    return {}


def format_docs_with_sources(docs: List[Document]) -> str:
    """Format retrieved docs with metadata so the LLM can cite sources."""
    formatted_docs = []

    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata

        source_text = f"""
[Source {i}]
File: {metadata.get("file_name", "Unknown")}
Page: {metadata.get("page_number", "Unknown")}
Version: {metadata.get("version", "Unknown")}
Effective Date: {metadata.get("effective_date", "Unknown")}
Latest: {metadata.get("is_latest", "Unknown")}

Content:
{doc.page_content}
"""
        formatted_docs.append(source_text)

    return "\n\n".join(formatted_docs)

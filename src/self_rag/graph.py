import json
from typing import Any, Dict

from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from src.config import Settings, load_settings
from src.self_rag.state import GraphState
from src.self_rag.utils import extract_json, format_docs_with_sources


def build_self_rag_app(settings: Settings | None = None):
    """
    Build and compile the Self-RAG LangGraph app.

    Flow:
    classify -> rewrite -> retrieve -> grade context
      -> retry if poor -> version/conflict check -> generate answer
      -> faithfulness check -> final answer
    """
    settings = settings or load_settings()

    embedding = CohereEmbeddings(
        model=settings.embedding_model,
        cohere_api_key=settings.cohere_api_key
    )

    vectorstore = FAISS.load_local(
        settings.faiss_db_path,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )

    # MMR improves document diversity, useful when expected docs span multiple PDFs.
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.top_k, "fetch_k": settings.fetch_k}
    )

    llm = ChatCohere(
        model=settings.chat_model,
        cohere_api_key=settings.cohere_api_key,
        temperature=settings.temperature
    )

    def llm_classify_query(question: str, has_chat_history: bool) -> dict:
        prompt = f"""
            You are a query router for a corporate policy/SOP Self-RAG system.

            Decide how to handle the user question.

            Return ONLY valid JSON:
            {{"need_retrieval": true/false, "clarification_needed": true/false, "reason": "short reason"}}

            Rules:
            - retrieval true: policy/SOP rules, reimbursement, claims, invoice, travel, leave, WFH, approval, documents, deployment, IT access, procurement, vendor, security, allowed/not allowed.
            - clarification true: vague follow-up like "is this allowed", "what about this", "can I do that" when there is no chat history.
            - retrieval false: greeting, general definition, small talk, assistant/project question, or non-policy question.
            - If chat history exists, vague follow-up questions can use retrieval.

            Examples:
            "Hello" -> {{"need_retrieval": false, "clarification_needed": false, "reason": "greeting"}}
            "What is SOP?" -> {{"need_retrieval": false, "clarification_needed": false, "reason": "general definition"}}
            "Can I claim hotel bill?" -> {{"need_retrieval": true, "clarification_needed": false, "reason": "reimbursement policy"}}
            "What documents are needed for this?" with chat history -> {{"need_retrieval": true, "clarification_needed": false, "reason": "follow-up policy context"}}
            "Is this allowed?" without chat history -> {{"need_retrieval": false, "clarification_needed": true, "reason": "ambiguous follow-up without context"}}

            Chat history exists: {has_chat_history}

            Question:
            {question}
            """

        response = llm.invoke(prompt)
        parsed = extract_json(response.content)

        return {
            "need_retrieval": parsed.get("need_retrieval", False),
            "clarification_needed": parsed.get("clarification_needed", False),
            "reason": parsed.get("reason", "llm fallback")
        }

    def classify_query(state: GraphState) -> Dict[str, Any]:
        question = state["question"].strip()
        q = question.lower()
        chat_history = state.get("chat_history", [])
        has_chat_history = len(chat_history) > 0

        if not q:
            return {
                "need_retrieval": False,
                "clarification_needed": True,
                "classification_reason": "empty query",
                "retry_count": state.get("retry_count", 0)
            }

        no_retrieval_phrases = [
            "hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
            "what is sop", "what is policy", "what is rag", "what is self rag",
            "explain sop", "meaning of sop", "who are you", "what can you do"
        ]

        if q in no_retrieval_phrases:
            return {
                "need_retrieval": False,
                "clarification_needed": False,
                "classification_reason": "rule: general/greeting query",
                "retry_count": state.get("retry_count", 0)
            }

        ambiguous_patterns = [
            "is this allowed", "is that allowed", "is it allowed",
            "can i do this", "can i do that", "what about this",
            "what about that", "what about it", "is this possible",
            "is that possible", "what documents are needed for this",
            "what approval is needed for this", "is this reimbursable",
            "can i claim this"
        ]

        if any(pattern in q for pattern in ambiguous_patterns) and not has_chat_history:
            return {
                "need_retrieval": False,
                "clarification_needed": True,
                "classification_reason": "rule: ambiguous follow-up without chat history",
                "retry_count": state.get("retry_count", 0)
            }

        retrieval_keywords = [
            "claim", "reimbursement", "reimbursable", "expense", "invoice", "gst",
            "bill", "hotel", "cab", "travel", "food", "leave", "notice period",
            "wfh", "work from home", "approval", "documents", "required document",
            "needed", "deploy", "deployment", "production", "friday", "access",
            "vpn", "jira", "git", "purchase", "procurement", "vendor", "security",
            "allowed", "not allowed", "can i", "am i allowed", "eligible",
            "eligibility", "limit", "policy", "sop"
        ]

        if any(keyword in q for keyword in retrieval_keywords):
            return {
                "need_retrieval": True,
                "clarification_needed": False,
                "classification_reason": "rule: policy/SOP keyword matched",
                "retry_count": state.get("retry_count", 0)
            }

        result = llm_classify_query(question, has_chat_history)

        return {
            "need_retrieval": result["need_retrieval"],
            "clarification_needed": result["clarification_needed"],
            "classification_reason": f"llm: {result['reason']}",
            "retry_count": state.get("retry_count", 0)
        }

    def clarification_response(state: GraphState) -> Dict[str, Any]:
        return {
                        "final_answer": """
            Please clarify what you are referring to.

            Example:
            - Is cab after 10 PM allowed?
            - Is Friday production deployment allowed?
            - Is leave during notice period allowed?
            - Is this expense reimbursable?

            Confidence: Low
            """.strip()
                    }

    def rewrite_query(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        chat_history = state.get("chat_history", [])

        if not chat_history:
            return {"rewritten_query": question}

        history_text = ""
        for msg in chat_history[-6:]:
            if isinstance(msg, HumanMessage):
                history_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"Assistant: {msg.content}\n"

        prompt = f"""
            Rewrite the latest user question into ONE standalone search query for corporate policy/SOP PDF retrieval.

            Rules:
            - Preserve exact amount, city, item, action, and policy topic from the user question.
            - Do not generalize the query.
            - Do not remove numbers like 4200, 50000, 500000.
            - If question is about hotel/cab/food/travel claim, include:
            travel policy, expense policy, reimbursement limit, approval, required documents, invoice, GST.
            - If question is about purchase/software/vendor, include:
            procurement SOP, approval matrix, purchase value, PO, IT security review, required documents.
            - If question is about deployment/production, include:
            deployment SOP, approval, checklist, production release, exception.
            - If question is about leave/WFH, include:
            leave policy, WFH policy, eligibility, approval.
            - Use chat history only to resolve this/that/it/same.
            - Do NOT answer.
            - Return only the search query.

            Chat history:
            {history_text}

            Latest question:
            {question}
            """
        response = llm.invoke(prompt)
        return {"rewritten_query": response.content.strip()}

    def rewrite_query_for_retry(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        previous_query = state.get("rewritten_query", question)
        context_reason = state.get("context_reason", "")
        retry_count = state.get("retry_count", 0)

        prompt = f"""
            You are improving a failed search query for a corporate policy/SOP RAG system.

            The previous retrieval was weak.

            Original user question:
            {question}

            Previous search query:
            {previous_query}

            Reason context was weak:
            {context_reason}

            Rewrite the query to retrieve better policy/SOP documents.
            Add useful keywords if relevant:
            reimbursement, claim, approval, invoice, GST, leave, deployment, procurement, IT access, security, vendor, expense, travel.

            Return only the improved search query.
            """
        response = llm.invoke(prompt)
        return {
            "rewritten_query": response.content.strip(),
            "retry_count": retry_count + 1
        }

    def retrieve_documents(state: GraphState) -> Dict[str, Any]:
        query = state.get("rewritten_query") or state["question"]
        docs = retriever.invoke(query)
        return {"retrieved_docs": docs}

    def grade_context(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        rewritten_query = state.get("rewritten_query", question)
        docs = state.get("retrieved_docs", [])
        context = format_docs_with_sources(docs)

        prompt = f"""
            You are a context relevance grader for a corporate policy RAG system.

            Check whether the retrieved context is relevant enough to answer the user question.

            User question:
            {question}

            Search query:
            {rewritten_query}

            Retrieved context:
            {context}

            Return ONLY valid JSON:
            {{"context_relevant": true/false, "score": 0.0, "reason": "short reason"}}

            Scoring:
            0.8-1.0 = clearly answers
            0.5-0.79 = partially useful
            below 0.5 = weak/irrelevant
            """
        response = llm.invoke(prompt)
        parsed = extract_json(response.content)

        return {
            "context_relevant": parsed.get("context_relevant", False),
            "context_score": float(parsed.get("score", 0.0)),
            "context_reason": parsed.get("reason", "")
        }

    def check_version_conflict(state: GraphState) -> Dict[str, Any]:
        docs = state.get("retrieved_docs", [])
        doc_info = []

        for doc in docs:
            doc_info.append({
                "file_name": doc.metadata.get("file_name"),
                "version": doc.metadata.get("version"),
                "effective_date": doc.metadata.get("effective_date"),
                "is_latest": doc.metadata.get("is_latest"),
                "page_number": doc.metadata.get("page_number"),
                "content": doc.page_content[:1000]
            })

        prompt = f"""
                You are a corporate policy version and conflict checker.

                Check:
                1. Are there multiple versions of the same policy?
                2. Do old/new versions conflict?
                3. Which latest documents should be used?

                Retrieved documents:
                {json.dumps(doc_info, indent=2)}

                Return ONLY valid JSON:
                {{"version_conflict_found": true/false, "latest_doc_files": ["file.pdf"], "conflict_summary": "short summary"}}
                """
        response = llm.invoke(prompt)
        parsed = extract_json(response.content)

        latest_doc_files = set(parsed.get("latest_doc_files", []))
        latest_docs = []

        for doc in docs:
            file_name = doc.metadata.get("file_name")
            is_latest = doc.metadata.get("is_latest")
            if file_name in latest_doc_files or is_latest is True:
                latest_docs.append(doc)

        if not latest_docs:
            latest_docs = docs

        return {
            "version_conflict_found": parsed.get("version_conflict_found", False),
            "latest_docs": latest_docs,
            "conflict_summary": parsed.get("conflict_summary", "")
        }

    def generate_answer(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        docs = state.get("latest_docs") or state.get("retrieved_docs", [])
        conflict_summary = state.get("conflict_summary", "")
        context = format_docs_with_sources(docs)

        prompt = f"""
            You are PolicyPilot AI, a careful corporate policy and SOP assistant.

            Your task:
            Answer the user question using ONLY the retrieved context.

            Strict rules:
            1. Do not invent rules, limits, dates, approvals, exceptions, or documents.
            2. Use only facts clearly present in the retrieved context.
            3. Prefer latest policy/SOP version when a version conflict exists.
            4. If old and latest versions conflict, mention the latest version used.
            5. If the question asks about an amount range like "above 50000" and the context has multiple slabs/ranges, show all applicable ranges. Do not collapse them into one approval.
            6. If exact amount, location, document type, or condition is needed but missing, say what is missing and give conditional answer from the context.
            7. If the context contains required documents, approvals, exceptions, or special checks, include them.
            8. If the retrieved context is not enough, say exactly:
            "I could not find a clear rule for this in the available policy documents."
            9. Do not use outside knowledge.
            10. Keep the answer concise but complete.

            Reasoning instruction:
            Before answering, internally check:
            - Which retrieved source is most relevant?
            - Is there any amount slab, limit, date, version, or exception?
            - Does the answer exactly follow the retrieved rule?
            - Are all claims supported by source text?

            Conflict/version note:
            {conflict_summary}

            Retrieved context:
            {context}

            User question:
            {question}

            Return the answer in this format:

            RAG AI:
            <clear answer. If conditional, say it clearly.>

            Reason:
            <brief explanation using only retrieved context.>

            Required Documents / Approval:
            <list documents, approvals, reviews, or say "Not specified in retrieved context".>

            Sources:
            - <file name>, Page <page>, Version <version>

            Confidence:
            High / Medium / Low (based on how directly the context supports the answer)
            Suggested follow-up questions:
            1. <A useful follow-up question based on the answer>
            2. <Another useful follow-up question>
            3. <Another useful follow-up question>
            """
        response = llm.invoke(prompt)
        return {"answer": response.content.strip()}

    def verify_faithfulness(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        answer = state.get("answer", "")
        docs = state.get("latest_docs") or state.get("retrieved_docs", [])
        context = format_docs_with_sources(docs)

        prompt = f"""
            You are a strict but fair faithfulness evaluator for a RAG system.

            Check whether the answer is supported by the retrieved context.

            Rules:
            - Mark supported if the answer is directly stated or clearly implied by context.
            - Do not require exact wording.
            - Ignore formatting differences like INR 50,000 vs 50000 vs ₹50,000.
            - Reject only claims not present, not implied, or contradictory.

            Question:
            {question}

            Retrieved context:
            {context}

            Answer:
            {answer}

            Return ONLY valid JSON:
            {{"is_answer_supported": true/false, "faithfulness_score": 0.0, "unsupported_claims": [], "reason": "short reason"}}

            Scoring:
            1.0 = every claim clearly supported
            0.8 = mostly supported, minor wording difference
            0.6 = partially supported
            below 0.6 = important claims unsupported
            """
        response = llm.invoke(prompt)
        parsed = extract_json(response.content)

        return {
            "is_answer_supported": parsed.get("is_answer_supported", False),
            "faithfulness_score": float(parsed.get("faithfulness_score", 0.0)),
            "unsupported_claims": parsed.get("unsupported_claims", []),
        }

    def final_response(state: GraphState) -> Dict[str, Any]:
        if state.get("is_answer_supported", False) or state.get("faithfulness_score", 0.0) >= 0.75:
            final_answer = state.get("answer", "")
        else:
            unsupported_claims = state.get("unsupported_claims", [])
            final_answer = f"""
                I could not generate a fully supported answer from the available policy documents.

                Reason:
                The faithfulness checker found that the answer may not be fully supported.

                Unsupported claims:
                {unsupported_claims}

                Please confirm with HR/Finance or upload the correct/latest policy document.

                Confidence: Low
                """
        return {"final_answer": final_answer.strip()}

    def direct_answer(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        prompt = f"""
        Answer this general question briefly.
        Do not pretend to use policy documents.

        Question:
        {question}
        """
        response = llm.invoke(prompt)
        return {"final_answer": response.content.strip()}

    def poor_context_response(state: GraphState) -> Dict[str, Any]:
        return {
            "final_answer": """
            I could not find relevant information in the available policy/SOP documents even after retrying retrieval.

            Please try rephrasing the question or upload the correct policy document.

            Confidence: Low
            """.strip()
                    }

    def route_after_classification(state: GraphState) -> str:
        if state.get("clarification_needed", False):
            return "clarification_response"
        if state.get("need_retrieval", True):
            return "rewrite_query"
        return "direct_answer"

    def route_after_context_grading(state: GraphState) -> str:
        score = state.get("context_score", 0.0)
        retry_count = state.get("retry_count", 0)

        if score >= 0.5:
            return "check_version_conflict"
        if retry_count < 1:
            return "rewrite_query_for_retry"
        return "poor_context_response"

    workflow = StateGraph(GraphState)

    workflow.add_node("classify_query", classify_query)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("clarification_response", clarification_response)
    workflow.add_node("rewrite_query_for_retry", rewrite_query_for_retry)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("grade_context", grade_context)
    workflow.add_node("check_version_conflict", check_version_conflict)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("verify_faithfulness", verify_faithfulness)
    workflow.add_node("final_response", final_response)
    workflow.add_node("direct_answer", direct_answer)
    workflow.add_node("poor_context_response", poor_context_response)

    workflow.add_edge(START, "classify_query")

    workflow.add_conditional_edges(
        "classify_query",
        route_after_classification,
        {
            "rewrite_query": "rewrite_query",
            "direct_answer": "direct_answer",
            "clarification_response": "clarification_response"
        }
    )

    workflow.add_edge("rewrite_query", "retrieve_documents")
    workflow.add_edge("rewrite_query_for_retry", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "grade_context")

    workflow.add_conditional_edges(
        "grade_context",
        route_after_context_grading,
        {
            "check_version_conflict": "check_version_conflict",
            "rewrite_query_for_retry": "rewrite_query_for_retry",
            "poor_context_response": "poor_context_response"
        }
    )

    workflow.add_edge("check_version_conflict", "generate_answer")
    workflow.add_edge("generate_answer", "verify_faithfulness")
    workflow.add_edge("verify_faithfulness", "final_response")

    workflow.add_edge("direct_answer", END)
    workflow.add_edge("clarification_response", END)
    workflow.add_edge("poor_context_response", END)
    workflow.add_edge("final_response", END)

    return workflow.compile()

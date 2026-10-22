import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.self_rag.graph import build_self_rag_app


st.set_page_config(
    page_title="PolicyPilot AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --card-bg: rgba(255, 255, 255, 0.78);
        --card-border: rgba(30, 41, 59, 0.10);
        --muted: #64748b;
        --text: #0f172a;
        --accent: #7c3aed;
        --accent-2: #06b6d4;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(124,58,237,0.13), transparent 30%),
            radial-gradient(circle at top right, rgba(6,182,212,0.13), transparent 32%),
            linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        border-right: 1px solid rgba(15, 23, 42, 0.08);
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.2rem;
        padding-bottom: 6rem;
    }

    .hero-card {
        padding: 1.5rem 1.7rem;
        border: 1px solid var(--card-border);
        background: var(--card-bg);
        border-radius: 1.4rem;
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(14px);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2.25rem;
        line-height: 1.05;
        font-weight: 900;
        color: var(--text);
        margin-bottom: 0.4rem;
        letter-spacing: -0.035em;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 1rem;
        margin-bottom: 0.85rem;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.55rem;
    }

    .pill {
        font-size: 0.8rem;
        font-weight: 650;
        color: #334155;
        padding: 0.35rem 0.6rem;
        border-radius: 999px;
        background: rgba(124, 58, 237, 0.08);
        border: 1px solid rgba(124, 58, 237, 0.13);
    }

    .section-label {
        color: #475569;
        font-size: 0.86rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.065em;
        margin: 1.2rem 0 0.6rem 0;
    }

    .small-muted {
        color: #64748b;
        font-size: 0.88rem;
        margin-bottom: 0.8rem;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 1.15rem;
        border: 1px solid rgba(15,23,42,0.07);
        background: rgba(255,255,255,0.78);
        box-shadow: 0 10px 26px rgba(15,23,42,0.045);
        margin-bottom: 0.85rem;
        padding: 0.25rem;
    }
    div[data-testid="stChatMessage"] * {
    color: #111827 !important;
}

    .stButton > button {
        border-radius: 0.85rem;
        border: 1px solid rgba(15,23,42,0.10);
        background: rgba(255,255,255,0.78);
        transition: 0.15s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(124,58,237,0.35);
        box-shadow: 0 10px 20px rgba(124,58,237,0.10);
    }

    textarea {
        border-radius: 0.8rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Self-RAG graph...")
def load_self_rag_app():
    return build_self_rag_app()


self_rag_app = load_self_rag_app()


# -----------------------------
# Session state
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "turns" not in st.session_state:
    st.session_state.turns = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# -----------------------------
# Helper functions
# -----------------------------
def clear_chat():
    st.session_state.chat_history = []
    st.session_state.turns = []
    st.session_state.last_result = None
    st.session_state.pending_question = None


def submit_question(question: str):
    if question and question.strip():
        st.session_state.pending_question = question.strip()
        st.rerun()


def render_sources(result: dict, turn_index: int):
    docs = result.get("retrieved_docs", [])

    if not docs:
        st.caption("No retrieved sources for this response.")
        return

    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata

        st.markdown(
            f"""
**Source {i}**  
File: `{metadata.get("file_name", "Unknown")}`  
Page: `{metadata.get("page_number", "Unknown")}`  
Version: `{metadata.get("version", "Unknown")}`  
Latest: `{metadata.get("is_latest", "Unknown")}`
"""
        )

        st.text_area(
            label=f"Content preview {turn_index}-{i}",
            value=doc.page_content[:1200],
            height=150,
            key=f"source_preview_{turn_index}_{i}",
            disabled=True,
        )


def render_debug(result: dict):
    debug_data = {
        "need_retrieval": result.get("need_retrieval"),
        "clarification_needed": result.get("clarification_needed"),
        "classification_reason": result.get("classification_reason"),
        "rewritten_query": result.get("rewritten_query"),
        "retry_count": result.get("retry_count"),
        "context_relevant": result.get("context_relevant"),
        "context_score": result.get("context_score"),
        "context_reason": result.get("context_reason"),
        "version_conflict_found": result.get("version_conflict_found"),
        "conflict_summary": result.get("conflict_summary"),
        "faithfulness_score": result.get("faithfulness_score"),
        "is_answer_supported": result.get("is_answer_supported"),
        "unsupported_claims": result.get("unsupported_claims"),
    }
    st.json(debug_data)


def run_question(question: str):
    with st.spinner("Running Self-RAG pipeline..."):
        result = self_rag_app.invoke(
            {
                "question": question,
                "chat_history": st.session_state.chat_history,
                "retry_count": 0,
            }
        )

    answer = result.get("final_answer", "")

    st.session_state.chat_history.extend(
        [
            HumanMessage(content=question),
            AIMessage(content=answer),
        ]
    )

    st.session_state.turns.append(
        {
            "question": question,
            "answer": answer,
            "result": result,
        }
    )

    st.session_state.last_result = result


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### 🧠 PolicyPilot")
    st.caption("Self-RAG assistant for corporate policy and SOP PDFs.")

    st.divider()
    st.markdown("#### Controls")

    show_sources = st.toggle("Show retrieved sources", value=True)
    show_debug = st.toggle("Show pipeline debug", value=False)

    st.button("🧹 New chat", use_container_width=True, on_click=clear_chat)

    st.divider()
    st.markdown("#### Example questions")

    examples = [
        "Can I claim 4200 hotel bill in Bangalore?",
        "What documents are needed for this?",
        "Can I deploy to production on Friday?",
        "What approval is needed for software purchase above 50000?",
        "Is this allowed?",
    ]

    for ex in examples:
        if st.button(ex, use_container_width=True):
            submit_question(ex)

    st.divider()
    st.markdown("#### Last run")

    last = st.session_state.last_result
    if last:
        c1, c2 = st.columns(2)
        c1.metric("Context", f"{last.get('context_score', 0) or 0:.2f}")
        c2.metric("Faithful", f"{last.get('faithfulness_score', 0) or 0:.2f}")
        st.caption(f"Retry count: {last.get('retry_count', 0)}")
        st.caption(f"Classifier: {last.get('classification_reason', 'N/A')}")
    else:
        st.caption("Ask a question to see pipeline metrics.")


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🧠 PolicyPilot AI</div>
        <div class="hero-subtitle">
            Chat with corporate policy PDFs. The assistant retrieves evidence, checks context quality,
            handles follow-ups, verifies faithfulness, and cites sources.
        </div>
        <div class="pill-row">
            <span class="pill">Self-RAG</span>
            <span class="pill">Memory</span>
            <span class="pill">FAISS + Cohere</span>
            <span class="pill">Source grounded</span>
            <span class="pill">Faithfulness checked</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="small-muted">Ask a main question, then continue with follow-ups like “What documents are needed for this?” or “Is this allowed?”</div>',
    unsafe_allow_html=True,
)


# -----------------------------
# Process pending question first
# -----------------------------
if st.session_state.pending_question:
    question_to_run = st.session_state.pending_question
    st.session_state.pending_question = None
    run_question(question_to_run)
    st.rerun()


# -----------------------------
# Conversation
# -----------------------------
st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)

if not st.session_state.turns:
    with st.chat_message("assistant", avatar="🧠"):
        st.write(
            "Hi! Ask a corporate policy/SOP question. I can answer with sources and remember follow-up questions in this session."
        )

for idx, turn in enumerate(st.session_state.turns):
    with st.chat_message("user", avatar="👤"):
        st.write(turn["question"])

    with st.chat_message("assistant", avatar="🧠"):
        st.write(turn["answer"])

        result = turn.get("result", {})

        if show_sources:
            with st.expander("📚 Retrieved sources", expanded=False):
                render_sources(result, turn_index=idx)

        if show_debug:
            with st.expander("🛠 Pipeline debug", expanded=False):
                render_debug(result)


# -----------------------------
# Bottom ChatGPT-style input only
# -----------------------------
bottom_question = st.chat_input("Ask a policy/SOP question...")

if bottom_question and bottom_question.strip():
    run_question(bottom_question.strip())
    st.rerun()
# 🧠 PolicyPilot AI

## Self-RAG Corporate Policy & SOP Decision Assistant

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge\&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-purple?style=for-the-badge)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-orange?style=for-the-badge)
![Cohere](https://img.shields.io/badge/LLM-Cohere-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=for-the-badge\&logo=streamlit)
![Self-RAG](https://img.shields.io/badge/Architecture-Self--RAG-black?style=for-the-badge)

---

## 🚀 Project Overview

**PolicyPilot AI** is an industry-style **Self-RAG based corporate policy and SOP assistant** that answers employee policy questions using company PDF documents.

Unlike a normal RAG system that simply retrieves documents and generates an answer, PolicyPilot AI performs multiple self-checking steps before giving the final response.

It can:

* Retrieve relevant policy/SOP PDF chunks
* Understand follow-up questions using memory
* Rewrite vague questions into better search queries
* Grade retrieved context quality
* Retry retrieval when context is weak
* Detect policy version conflicts
* Prefer latest policy versions
* Generate source-grounded answers
* Verify faithfulness before responding
* Show retrieved sources and pipeline debug information
* Provide a premium ChatGPT-style Streamlit UI

This project is designed as a **resume-level and industry-level GenAI application** using modern agentic RAG patterns.

---

## 🎯 Problem Statement

In large organizations, employees often struggle to find correct answers from multiple policy and SOP documents.

Examples:

* Can I claim a hotel bill of ₹4200 in Bangalore?
* What approval is required for software purchase above ₹50,000?
* Can I deploy to production on Friday?
* What documents are needed for this claim?
* Is cab fare after 10 PM reimbursable?

Traditional keyword search is weak because:

* Policies are spread across multiple PDFs
* Some documents have old and new versions
* Questions may be incomplete or follow-up based
* Users need exact approval rules, limits, and documents
* LLMs may hallucinate if not properly grounded

**PolicyPilot AI solves this by combining RAG + Self-RAG + LangGraph + evaluation.**

---

## 🧠 What is Self-RAG in this Project?

Normal RAG flow:

```text
User Question
    ↓
Retrieve Documents
    ↓
Generate Answer
```

PolicyPilot AI Self-RAG flow:

```text
User Question
    ↓
Hybrid Query Classifier
    ↓
Memory-Based Query Rewrite
    ↓
FAISS Retrieval
    ↓
Context Relevance Grader
    ↓
Retry Retrieval if Context is Poor
    ↓
Version / Conflict Checker
    ↓
Grounded Answer Generator
    ↓
Faithfulness Verifier
    ↓
Final Response
```

The system does not blindly answer.
It checks whether:

* Retrieval is needed
* The question is ambiguous
* Retrieved context is useful
* Policy versions conflict
* The answer is supported by source documents
* The final response should be shown or safely refused

---

## ✨ Key Features

### ✅ 1. PDF-Based Corporate Policy Assistant

The system ingests corporate policy/SOP PDF documents and builds a searchable FAISS vector database.

Supported example document types:

* Travel Policy
* Expense Reimbursement SOP
* Procurement SOP
* Leave Policy
* Work From Home Policy
* IT Access SOP
* Information Security Policy
* Production Deployment SOP
* Vendor Invoice Validation Policy

---

### ✅ 2. FAISS Vector Search

The project uses FAISS for local vector database storage.

```text
PDF → Text → Chunks → Embeddings → FAISS Vector DB
```

FAISS enables fast semantic retrieval from policy documents.

---

### ✅ 3. Cohere Embeddings

The project uses Cohere embeddings for converting text chunks into vectors.

Example:

```python
CohereEmbeddings(model="embed-v4.0")
```

---

### ✅ 4. Cohere LLM

Cohere chat model is used for:

* Query classification
* Query rewriting
* Context grading
* Version conflict detection
* Answer generation
* Faithfulness verification

---

### ✅ 5. Hybrid Query Classifier

The classifier uses a hybrid approach:

```text
Rule-based check first
    ↓
If unclear, LLM classifier
```

It decides whether a user question needs retrieval from PDFs.

Examples:

| User Query                       | Decision               |
| -------------------------------- | ---------------------- |
| Hello                            | No retrieval           |
| What is SOP?                     | No retrieval           |
| Can I claim hotel bill?          | Retrieval required     |
| Is this allowed? with no history | Clarification required |
| Is this allowed? with history    | Retrieval required     |

---

### ✅ 6. Memory for Follow-Up Questions

The Streamlit app stores conversation history using:

```python
HumanMessage
AIMessage
```

This allows follow-up questions like:

```text
User: Can I claim 4200 hotel bill in Bangalore?
User: What documents are needed for this?
User: Is this allowed?
```

The assistant understands what “this” refers to.

---

### ✅ 7. Query Rewriting

Follow-up questions are rewritten into standalone search queries.

Example:

```text
User follow-up:
What documents are needed for this?

Rewritten query:
Required documents for claiming INR 4200 hotel bill in Bangalore under travel and expense policy
```

This improves retrieval quality.

---

### ✅ 8. Context Relevance Grading

After retrieval, the system checks whether retrieved chunks are useful enough.

Output example:

```json
{
  "context_relevant": true,
  "context_score": 0.86,
  "context_reason": "Retrieved context contains hotel reimbursement limit and required documents."
}
```

If context score is low, the system retries retrieval with a better query.

---

### ✅ 9. Poor Context Retry

If the first retrieval is weak:

```text
Poor context
    ↓
Rewrite query better
    ↓
Retrieve again
    ↓
Grade context again
```

If still poor, the system gives a safe response instead of hallucinating.

---

### ✅ 10. Policy Version Conflict Checker

The system checks whether multiple versions of the same policy are retrieved.

Example:

```text
Travel Policy v1 → older
Travel Policy v2 → latest
```

The assistant prefers the latest version and can mention conflict handling.

---

### ✅ 11. Faithfulness Verification

Before final response, the system checks whether the generated answer is supported by retrieved context.

Output example:

```json
{
  "is_answer_supported": true,
  "faithfulness_score": 0.92,
  "unsupported_claims": []
}
```

If unsupported, the system blocks or safely refuses the answer.

---

### ✅ 12. Premium Streamlit UI

The project includes a clean ChatGPT-like interface with:

* Sidebar controls
* New chat button
* Example questions
* Chat memory
* Retrieved source viewer
* Pipeline debug panel
* Context score
* Faithfulness score
* Retry count
* Classification reason

---

## 🏗️ Architecture

```text
                    ┌──────────────────────────┐
                    │      User Question       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Hybrid Query Classifier  │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
   ┌──────────────────────┐              ┌──────────────────────┐
   │ Clarification Needed │              │   Retrieval Needed   │
   └──────────┬───────────┘              └──────────┬───────────┘
              │                                     │
              ▼                                     ▼
   ┌──────────────────────┐              ┌──────────────────────┐
   │ Ask Clarification    │              │ Query Rewrite +      │
   │                      │              │ Memory Resolution    │
   └──────────────────────┘              └──────────┬───────────┘
                                                     │
                                                     ▼
                                        ┌──────────────────────────┐
                                        │ FAISS Vector Retrieval   │
                                        └──────────┬───────────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────────────┐
                                        │ Context Relevance Grader │
                                        └──────────┬───────────────┘
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         │                                                   │
                         ▼                                                   ▼
              ┌──────────────────────┐                         ┌──────────────────────┐
              │ Poor Context Retry   │                         │ Version Conflict     │
              │ Query Rewrite Again  │                         │ Checker              │
              └──────────┬───────────┘                         └──────────┬───────────┘
                         │                                                  │
                         └───────────────► Retrieval Again                  ▼
                                                            ┌──────────────────────────┐
                                                            │ Grounded Answer Generator│
                                                            └──────────┬───────────────┘
                                                                       │
                                                                       ▼
                                                            ┌──────────────────────────┐
                                                            │ Faithfulness Verifier    │
                                                            └──────────┬───────────────┘
                                                                       │
                                                                       ▼
                                                            ┌──────────────────────────┐
                                                            │ Final Answer + Sources   │
                                                            └──────────────────────────┘
```

---

## 📁 Project Structure

```text
policy_pilot_ai/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── src/
│   ├── config.py
│   │
│   ├── self_rag/
│   │   ├── graph.py
│   │   ├── state.py
│   │   └── utils.py
│   │
│   └── evaluation/
│       ├── custom_metrics.py
│       └── golden_tests.py
│
├── notebooks/
│   └── self_rag.ipynb
│
├── data/
│   └── self_rag_policy_dataset/
│
└── vector_db/
    └── faiss_policy_index/
```

---

## 🛠️ Tech Stack

| Layer           | Technology                   |
| --------------- | ---------------------------- |
| LLM             | Cohere Chat                  |
| Embeddings      | Cohere Embed v4.0            |
| Vector Database | FAISS                        |
| Agent Workflow  | LangGraph                    |
| RAG Framework   | LangChain                    |
| UI              | Streamlit                    |
| Evaluation      | Custom metrics + RAGAS-ready |
| Language        | Python                       |
| Data Source     | Corporate Policy/SOP PDFs    |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/policy-pilot-ai.git
cd policy-pilot-ai
```

---

### 2. Create virtual environment

```bash
python -m venv .venv
```

Activate it:

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Add environment variables

Create a `.env` file:

```env
COHERE_API_KEY=your_cohere_api_key_here
```

Do not push `.env` to GitHub.

Use `.env.example` for reference:

```env
COHERE_API_KEY=your_cohere_api_key_here
```

---

### 5. Add or generate FAISS vector database

If `vector_db/faiss_policy_index/` is not available, run the ingestion notebook or ingestion script first.

Expected vector DB path:

```text
vector_db/faiss_policy_index/
```

---

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

Open browser:

```text
http://localhost:8501
```

---

## 💬 Example Questions

Try these questions in the UI:

```text
Can I claim 4200 hotel bill in Bangalore?
```

```text
What documents are needed for this?
```

```text
Can I deploy to production on Friday?
```

```text
What approval is needed for software purchase above 50000?
```

```text
Is this allowed?
```

```text
Can I claim cab fare after 10 PM?
```

```text
Can I take leave during notice period?
```

---

## 🧪 Evaluation

The project supports custom evaluation metrics.

### Custom Metrics

| Metric               | Meaning                                      |
| -------------------- | -------------------------------------------- |
| Hit Rate@5           | At least one expected document was retrieved |
| All Expected Found@5 | All expected documents were retrieved        |
| Context Score        | LLM-graded relevance of retrieved context    |
| Faithfulness Score   | LLM-graded answer support                    |
| Retry Count          | Whether Self-RAG had to retry retrieval      |

Example:

```text
Hit Rate@5: 1.0
All Expected Found@5: 0.8
Average Context Score: 0.86
Average Faithfulness Score: 0.91
```

---

## 📊 Why Custom Evaluation?

RAGAS is useful, but for corporate policy systems, business-level checks are also important.

For example:

```text
Question:
What approval is needed for software purchase above 50000?

Expected document:
AceCorp_Procurement_SOP.pdf

Expected logic:
- INR 50,001 to INR 100,000 → Department Head
- INR 100,001 to INR 500,000 → Department Head + Procurement Head
- Above INR 500,000 → Business Unit Head + Finance Head
```

This kind of approval matrix accuracy is better checked using custom rules and golden questions.

---

## 🔍 Example Self-RAG Debug Output

```json
{
  "need_retrieval": true,
  "clarification_needed": false,
  "classification_reason": "rule: policy/SOP keyword matched",
  "rewritten_query": "hotel reimbursement claim INR 4200 Bangalore travel policy expense policy reimbursement limit approval required documents GST invoice",
  "retry_count": 0,
  "context_relevant": true,
  "context_score": 0.89,
  "version_conflict_found": false,
  "faithfulness_score": 0.94,
  "is_answer_supported": true,
  "unsupported_claims": []
}
```

---

## 🧩 Main Self-RAG Nodes

### 1. `classify_query`

Decides whether the question needs PDF retrieval.

### 2. `rewrite_query`

Converts follow-up questions into standalone search queries.

### 3. `retrieve_documents`

Retrieves relevant chunks from FAISS.

### 4. `grade_context`

Checks whether retrieved chunks are useful.

### 5. `rewrite_query_for_retry`

Improves the query if context is poor.

### 6. `check_version_conflict`

Checks old vs latest policy versions.

### 7. `generate_answer`

Generates a grounded answer from retrieved context.

### 8. `verify_faithfulness`

Checks whether the generated answer is supported.

### 9. `final_response`

Returns either the answer or a safe refusal.

---

## 🖥️ UI Preview

The Streamlit UI includes:

* Premium chat interface
* Example question buttons
* New chat button
* Retrieved source expander
* Debug panel
* Context and faithfulness metrics
* Follow-up memory

---

## 📌 Sample Output

### User

```text
What approval is needed for software purchase above 50000?
```

### PolicyPilot AI

```text
RAG AI:
Approval depends on the exact purchase value.

Reason:
The Procurement SOP approval matrix defines different approval levels based on purchase value.

Required Documents / Approval:
- INR 50,001 to INR 100,000: Department Head approval
- INR 100,001 to INR 500,000: Department Head and Procurement Head approval
- Above INR 500,000: Business Unit Head and Finance Head approval
- Software purchase also requires IT security review before PO creation

Sources:
- AceCorp_Procurement_SOP.pdf, Page 2, Version v1.0
- AceCorp_Procurement_SOP.pdf, Page 1, Version v1.0

Confidence:
High

Suggested follow-up questions:
1. What documents are required for this software purchase?
2. Is IT security review mandatory for software purchase?
3. How many vendor quotes are needed for this amount?
```

---

## 🔐 Security Notes

Do not commit:

```text
.env
API keys
vector DB with confidential company data
real internal policies
employee data
production secrets
```

This demo project uses fictional corporate documents.

---

## 🧠 Resume Highlight

You can describe this project on resume like this:

```text
Built PolicyPilot AI, a Self-RAG based corporate policy/SOP assistant using LangGraph, FAISS, Cohere embeddings, Cohere LLM, and Streamlit. Implemented hybrid query routing, memory-based query rewriting, context relevance grading, poor-context retry, policy version conflict detection, grounded answer generation, faithfulness verification, source citation, and custom evaluation metrics such as Hit Rate@5 and expected-document recall.
```

---

## 🎥 Demo Flow

Recommended demo video flow:

1. Show project architecture
2. Ask main question:

   ```text
   Can I claim 4200 hotel bill in Bangalore?
   ```
3. Ask follow-up:

   ```text
   What documents are needed for this?
   ```
4. Show retrieved sources
5. Show debug panel
6. Ask procurement approval question
7. Explain context score and faithfulness score
8. Show code structure
9. Show evaluation metrics

---

## 🚧 Future Improvements

Possible future upgrades:

* Add RAGAS evaluation dashboard
* Add user authentication
* Add upload-your-own-policy feature
* Add admin panel for document ingestion
* Add PDF preview with highlighted citations
* Add multi-tenant company support
* Add role-based policy access
* Add Azure OpenAI or OpenAI evaluator
* Add LangSmith tracing
* Add Datadog logs and monitoring
* Add Docker deployment
* Deploy on Hugging Face Spaces / Streamlit Cloud / AWS

---

## 🤝 Contribution

This is a demo project for learning and portfolio use.

Contributions, suggestions, and improvements are welcome.

---

## 📄 License

This project is for educational and demonstration purposes.

---

## ⭐ Final Note

PolicyPilot AI demonstrates how a production-style GenAI system should not only generate answers, but also **retrieve, verify, reason, cite, and self-correct** before responding.

This makes it much stronger than a basic RAG chatbot.

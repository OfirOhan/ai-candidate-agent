<div align="center">

# 🤖 AI Candidate Agent

**An AI-powered digital avatar designed to represent you to recruiters 24/7.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Ollama-black?style=for-the-badge&logo=ollama&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-FFA500?style=for-the-badge&logo=database&logoColor=white" />
</p>

</div>

---

## 📖 Overview

The **AI Candidate Agent** is a conversational AI representative that answers technical, behavioral, and logistical questions about a candidate's profile on their behalf.

Moving beyond standard "Chat with PDF" retrieval, it combines an **Agentic ReAct architecture** with an **Advanced RAG Pipeline**. The agent intelligently routes between querying structured verified facts and executing hybrid search over unstructured documents, delivering accurate, grounded, and context-aware responses — all running on **local open-source models** with zero external API costs.

### 🛠️ Tech Stack
| Layer | Technologies |
|-------|-------------|
| **LLM & Embeddings** | Ollama (Qwen3), Nomic embeddings, `sentence-transformers` |
| **Retrieval** | ChromaDB (dense), BM25 Okapi (sparse), Reciprocal Rank Fusion, Qwen3-Reranker |
| **Agent & Orchestration** | LangChain, ReAct-style tool calling |
| **Ingestion** | `unstructured`, PyMuPDF, section-aware chunking |
| **Evaluation** | RAGAS, DeepEval (GEval), custom retrieval-gate analysis |
| **Frontend** | Streamlit |

---

## ✨ Key Features

### 🕵️‍♂️ Agentic Tool Calling
The system doesn't blindly query a vector database. It utilizes an LLM agent with multiple tools at its disposal:
* `get_structured_data`: Retrieves verified, hard facts (salary expectations, availability, specific degree names) from a structured JSON store.
* `search_documents`: Executes the RAG pipeline over unstructured data (CVs, cover letters, certificates).
* **Multi-Step Fallbacks:** The agent is capable of chaining tools — if the structured data returns a short summary, the agent will dynamically follow up with a semantic document search to gather richer detail.

### 🧠 Advanced RAG Pipeline
The document engine (`rag/ingest.py` and `rag/retriever.py`) implements advanced ingestion and search techniques:
* **Intelligent Ingestion & Chunking:** Uses `unstructured` to parse diverse formats (PDF, DOCX) while intelligently grouping text by logical document sections. It detects complex headers and prepends the section title to each chunk to guarantee semantic richness.
* **Dual-Index Generation:** During ingestion, an LLM automatically generates a factual 5-6 sentence summary of every document. This is stored in a separate summary index to handle broad conversational queries.
* **Query Routing:** Dynamically classifies queries as `BROAD` (fetching the pre-computed candidate summaries) or `SPECIFIC` (triggering deep search).
* **Query Expansion:** Uses an LLM to generate multiple semantic variations of the user's query to maximize recall.
* **Hybrid Search & RRF:** Combines dense semantic vector search (ChromaDB + SentenceTransformers) with sparse keyword search (BM25 Okapi) using **Reciprocal Rank Fusion**.
* **Cross-Encoder Re-ranking:** Re-scores the fused candidate pool using `Qwen3-Reranker-0.6B`, returning the top-8 most relevant chunks to maximize context quality.

### 📊 Automated Evaluation Suite
Built with **RAGAS** and **DeepEval (GEval)**, the `evaluation/` module rigorously benchmarks the agent across 7 components:

| Component | What it measures |
|-----------|-----------------|
| **Tool Selection** | Whether the agent picks the right tool for each question |
| **RAG Quality (RAGAS)** | Faithfulness, answer relevancy, context precision & recall |
| **Retrieval Gate Localization** | Where retrieval fails: ingestion vs. recall vs. re-rank |
| **Answer Correctness (GEval)** | LLM-as-judge scoring of factual correctness vs. ground truth |
| **Refusal Accuracy** | Correct handling of out-of-scope and sensitive questions |
| **Ingestion Quality** | Chunk coverage, section detection, summary quality |
| **Router Accuracy** | Broad vs. specific query classification |

**Current Results** *(6 candidates · 426 questions · 11.1s avg latency)*:

| Metric | Score |
|--------|-------|
| Tool Selection Accuracy | **92.5%** |
| Refusal Accuracy | **97.7%** |
| Router Accuracy | **81.6%** |
| RAG — Faithfulness | **91.2%** |
| RAG — Answer Relevancy | **85.9%** |
| RAG — Context Recall | **76.0%** |
| RAG — Context Precision | **72.3%** |
| Answer Correctness (GEval) | **79.2%** |

A standout feature of the suite is **retrieval gate localization**, which traces each failed query to the exact stage it broke down — ingestion, recall, or re-ranking. This pinpointed the re-ranker as the primary bottleneck and informed a targeted upgrade to `Qwen3-Reranker-0.6B`, rather than guessing from an aggregate recall score.

### 💻 Candidate Setup Dashboard
A sleek Streamlit interface where the candidate can easily input verified structured facts and upload unstructured PDFs/Docs for automatic chunking and ingestion.

<p align="center">
  <img src="images/candidate_side.png" alt="Candidate Setup Dashboard" width="500" />
</p>

### 💬 Recruiter Chat Interface
Recruiters interact with the AI agent through a clean conversational UI, asking questions about the candidate's background, skills, and availability — all answered in real-time by the agentic pipeline.

<p align="center">
  <img src="images/recruiter_side.png" alt="Recruiter Chat Interface" width="500" />
</p>

---

## 🏗️ Architecture Overview

```mermaid
graph TD;
    A[Recruiter] -->|Questions| B(Streamlit Chat Interface);
    B --> C{LangChain Agent};
    C -->|Fetch Exact Facts| D[(Structured Store)];
    C -->|Search Context| E[Advanced RAG Pipeline];
    
    E --> F{Query Router};
    F -->|Broad| G[Fetch Summary Collection];
    F -->|Specific| H[Query Expansion];
    
    H --> I[Vector Search ChromaDB];
    H --> J[BM25 Keyword Search];
    
    I --> K((Reciprocal Rank Fusion));
    J --> K;
    
    K --> L[Qwen3-Reranker-0.6B];
    L --> C;
    
    C -->|Generate Response| B;
```

---

## 🚀 Getting Started

### Prerequisites
1. Ensure you have **Python 3.10+** installed.
2. Install and run [Ollama](https://ollama.ai/).
3. Pull the required models:
   ```bash
   ollama pull qwen3
   ```

### Installation
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <your-repo-url>
   cd ai-candidate-agent
   ```
2. Set up your virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Running the App
Start the Streamlit application:
```bash
streamlit run main.py
```
1. **Candidate Setup (`/setup`):** Fill in your verified details and upload your CV/documents.
2. **Recruiter Chat (`/recruiter`):** Share the link with recruiters so they can chat with your personalized AI agent!

---

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

The **AI Candidate Agent** is an AI-powered conversational representative built to answer technical, behavioral, and logistical questions about a candidate's profile. 

Moving beyond standard "Chat with PDF" retrieval, it utilizes an **Agentic ReAct architecture** and an **Advanced RAG Pipeline**. The agent intelligently navigates between querying structured verified facts and executing hybrid searches over unstructured documentation, delivering accurate and context-aware responses.

---

## ✨ Key Features

### 🕵️‍♂️ Agentic Tool Calling
The system doesn't blindly query a vector database. It utilizes an LLM agent with multiple tools at its disposal:
* `get_structured_data`: Retrieves verified, hard facts (salary expectations, availability, specific degree names) from a structured JSON store.
* `search_documents`: Executes the RAG pipeline over unstructured data (CVs, cover letters, certificates).
* **Multi-Step Fallbacks:** The agent is capable of chaining tools—if the structured data returns a short summary, the agent will dynamically follow up with a semantic document search to gather richer detail.

### 🧠 Advanced RAG Pipeline
The document engine (`rag/ingest.py` and `rag/retriever.py`) implements advanced ingestion and search techniques:
* **Intelligent Ingestion & Chunking:** Uses `unstructured` to parse diverse formats (PDF, DOCX) while intelligently grouping text by logical document sections. It detects complex headers and prepends the section title to each chunk to guarantee semantic richness.
* **Dual-Index Generation:** During ingestion, an LLM automatically generates a factual 5-6 sentence summary of every document. This is stored in a separate summary index to handle broad conversational queries.
* **Query Routing:** Dynamically classifies queries as `BROAD` (fetching the pre-computed candidate summaries) or `SPECIFIC` (triggering deep search).
* **Query Expansion:** Uses an LLM to generate multiple semantic variations of the user's query to maximize recall.
* **Hybrid Search & RRF:** Combines dense semantic vector search (ChromaDB + SentenceTransformers) with sparse keyword search (BM25 Okapi) using **Reciprocal Rank Fusion**.
* **Cross-Encoder Re-ranking:** Re-scores the retrieved chunks using `ms-marco-MiniLM` to ensure the final context injected into the prompt has maximum relevance.

### 📊 Automated Evaluation Suite
Built with **Ragas** and **DeepEval**, the `evaluation/` module rigorously benchmarks the agent against test datasets to measure context precision, hallucination rates, and tool selection accuracy.

**Baseline Evaluation Results:**
| Metric | Score | Description |
|--------|-------|-------------|
| **Tool Selection Accuracy** | **93.7%** | The agent's ability to correctly choose between structured data and document search. |
| **Faithfulness (Ragas)** | **88.8%** | Measures how factually accurate the generated answer is based on the retrieved context. |
| **Router Accuracy** | **86.1%** | The system's accuracy in routing queries to `BROAD` (summaries) vs `SPECIFIC` (hybrid search). |
| **Answer Relevancy (Ragas)** | **77.1%** | How relevant the generated answer is to the original user query. |
| **Context Recall (Ragas)** | **63.5%** | Evaluates if the retrieved chunks contain all the necessary information to answer the query. |
| **Hallucination Rate (DeepEval)** | **25.6%** | The percentage of responses containing fabricated information (lower is better). |

> **🚀 Note on Baselines & Future Work:** These metrics represent the system's baseline performance running on local models via Ollama. While tool-routing and faithfulness are highly accurate, there is room to improve Context Recall and reduce the Hallucination Rate. My immediate next steps include refining the chunking strategy, experimenting with larger parameter models, and fine-tuning the cross-encoder to drive recall up and hallucinations down.

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
    
    K --> L[Cross-Encoder Re-ranker];
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
   ollama pull qwen3  # Or the model configured in your environment
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



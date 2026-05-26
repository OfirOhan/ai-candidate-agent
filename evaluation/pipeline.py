"""
Headless pipeline wrapper for evaluation.

Provides functions to ingest text, run retrieval, and run the full agent
without requiring Streamlit or a running server.
"""

import agent.tools as tools_module
from rag.ingest import text_splitter, embedder, get_collection
from rag.retriever import retrieve
from agent.agent import run as agent_run


def ingest_text_for_eval(text: str, candidate_id: str):
    """Ingest raw text directly into ChromaDB (bypasses PDF extraction)."""
    chunks = text_splitter.split_text(text)
    embeddings = embedder.encode(chunks).tolist()
    collection = get_collection(candidate_id)

    ids = [f"eval_chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"[Eval Pipeline] Ingested {len(chunks)} chunks for '{candidate_id}'")


def run_retrieval(question: str, candidate_id: str, top_k: int = 3) -> list[str]:
    """Run retrieval only, return list of context chunks."""
    return retrieve(question, candidate_id, top_k=top_k)


def run_agent_answer(question: str) -> str:
    """Run the full agent with a fresh conversation, return answer string."""
    answer, _ = agent_run([], question)
    return answer


def run_full_pipeline(
    question: str,
    candidate_id: str,
    top_k: int = 3,
) -> dict:
    """
    Run retrieval + agent for a single question.

    Returns:
        {
            "question": str,
            "contexts": list[str],   # from retriever
            "answer": str,           # from agent
        }
    """
    contexts = run_retrieval(question, candidate_id, top_k=top_k)
    answer = run_agent_answer(question)
    return {
        "question": question,
        "contexts": contexts,
        "answer": answer,
    }


def set_candidate_id(candidate_id: str):
    """Monkey-patch the agent's CANDIDATE_ID for evaluation."""
    tools_module.CANDIDATE_ID = candidate_id


def restore_candidate_id(original_id: str = "candidate_001"):
    """Restore the original CANDIDATE_ID after evaluation."""
    tools_module.CANDIDATE_ID = original_id

"""
Headless pipeline wrapper for evaluation.

Provides functions to ingest text, run retrieval, and run the full agent
without requiring Streamlit or a running server.
"""

import agent.tools as tools_module
from rag.ingest import ingest_document, text_splitter, embedder, get_collection
from rag.retriever import retrieve
from agent.agent import run as agent_run


def ingest_file_for_eval(file_path: str, candidate_id: str, doc_type: str = "cv"):
    """Ingest a file using the full production pipeline (section-aware + summary).

    This replaces the old ingest_text_for_eval which used a raw text splitter
    without section extraction or summary generation.
    """
    ingest_document(file_path, candidate_id, doc_type=doc_type)
    print(f"[Eval Pipeline] Ingested '{file_path}' for '{candidate_id}' via production path")


def ingest_text_for_eval(text: str, candidate_id: str):
    """Fallback: Ingest raw text directly into ChromaDB (bypasses PDF extraction).

    NOTE: This is kept for backward compatibility but ingest_file_for_eval
    is preferred as it creates section metadata and summaries.
    """
    chunks = text_splitter.split_text(text)
    embeddings = embedder.encode(chunks).tolist()
    collection = get_collection(candidate_id)

    ids = [f"eval_chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"[Eval Pipeline] Ingested {len(chunks)} chunks for '{candidate_id}' (raw text)")


def run_retrieval(question: str, candidate_id: str, top_k: int = 3) -> dict:
    """Run retrieval only, return full result dict with chunks, route, expanded_queries."""
    return retrieve(question, candidate_id, top_k=top_k)


def run_agent_answer(question: str) -> tuple[str, list]:
    """Run the full agent with a fresh conversation, return (answer, tool_trajectory)."""
    answer, _, trajectory = agent_run([], question)
    return answer, trajectory


def run_full_pipeline(
    question: str,
    candidate_id: str,
    top_k: int = 3,
) -> dict:
    """
    Run the full agent for a single question and capture all metadata.

    Returns:
        {
            "question": str,
            "contexts": list[str],        # from retriever (only if RAG was used)
            "answer": str,                # from agent
            "tool_trajectory": list[dict], # sequence of tool calls
            "final_tool": str | None,     # last tool used (or None if no tools)
            "route": str | None,          # "broad"/"specific" if RAG was used
        }
    """
    # Run the agent — it will internally call tools as needed
    answer, trajectory = run_agent_answer(question)

    # Determine the final tool used
    final_tool = trajectory[-1]["tool"] if trajectory else None

    # Get retrieval metadata if search_documents was used
    route = None
    contexts = []
    if any(t["tool"] == "search_documents" for t in trajectory):
        # Retrieve the route metadata captured during the agent run
        retrieval_meta = tools_module.get_last_retrieval_meta()
        route = retrieval_meta.get("route")
        # Run retrieval again to get the actual chunks for evaluation
        # (the agent consumed them as text, we need the list form)
        retrieval_result = run_retrieval(question, candidate_id, top_k=top_k)
        contexts = retrieval_result["chunks"]

    return {
        "question": question,
        "contexts": contexts,
        "answer": answer,
        "tool_trajectory": trajectory,
        "final_tool": final_tool,
        "route": route,
    }


def set_candidate_id(candidate_id: str):
    """Monkey-patch the agent's CANDIDATE_ID for evaluation."""
    tools_module.CANDIDATE_ID = candidate_id


def restore_candidate_id(original_id: str = "candidate_001"):
    """Restore the original CANDIDATE_ID after evaluation."""
    tools_module.CANDIDATE_ID = original_id

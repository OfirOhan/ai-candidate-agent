"""
Evaluation harness — orchestrates data seeding, pipeline execution,
component evaluations, and report generation.

Components:
  1. Tool Selection — expected vs actual tool
  2. RAG Quality — RAGAS + DeepEval Hallucination (RAG-only)
  3. Answer Correctness — GEval (all questions except negative)
  4. Refusal Accuracy — negative questions
  5. Ingestion Quality — chunk stats, coverage, summary quality
  6. Router Accuracy — broad/specific classification
"""

import json
import os
import shutil
import time
from pathlib import Path

import chromadb
import pandas as pd

from evaluation.pipeline import (
    run_full_pipeline,
    set_candidate_id,
    restore_candidate_id,
)
from rag.ingest import ingest_document
from store.structured import DATA_PATH as STRUCTURED_DATA_PATH

# Paths
EVAL_DIR = Path(__file__).parent
DATA_DIR = EVAL_DIR / "data"
REPORTS_DIR = EVAL_DIR / "reports"

GOLDEN_DATASET_PATH = DATA_DIR / "golden_dataset.json"
CANDIDATE_SEED_PATH = DATA_DIR / "candidate_seed.json"
RESUME_PATH = DATA_DIR / "synthetic_resume.md"

EVAL_CANDIDATE_ID = "eval_candidate"
CHROMA_PATH = "./chroma_db"


def _load_golden_dataset(category_filter: str | None = None) -> list[dict]:
    """Load the golden Q&A dataset, optionally filtering by category."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if category_filter:
        dataset = [q for q in dataset if q["category"] == category_filter]
    return dataset


def _seed_structured_data() -> str | None:
    """
    Copy candidate_seed.json → store/data/candidate.json.
    Returns the path to the backup file if one existed, else None.
    """
    backup_path = None
    if os.path.exists(STRUCTURED_DATA_PATH):
        backup_path = STRUCTURED_DATA_PATH + ".eval_backup"
        shutil.copy2(STRUCTURED_DATA_PATH, backup_path)

    os.makedirs(os.path.dirname(STRUCTURED_DATA_PATH), exist_ok=True)
    shutil.copy2(CANDIDATE_SEED_PATH, STRUCTURED_DATA_PATH)
    print(f"[Harness] Seeded structured data from {CANDIDATE_SEED_PATH}")
    return backup_path


def _restore_structured_data(backup_path: str | None):
    """Restore the original candidate.json from backup."""
    if backup_path and os.path.exists(backup_path):
        shutil.move(backup_path, STRUCTURED_DATA_PATH)
        print("[Harness] Restored original structured data.")
    elif not backup_path:
        if os.path.exists(STRUCTURED_DATA_PATH):
            os.remove(STRUCTURED_DATA_PATH)


def _seed_documents():
    """Ingest the synthetic resume using the real production pipeline.

    Calls ingest_document() directly so the eval path is identical to
    production — section extraction, contextualised chunks, metadata,
    and summary generation are all included.
    """
    ingest_document(str(RESUME_PATH), EVAL_CANDIDATE_ID, doc_type="cv")
    print(f"[Harness] Ingested synthetic resume into '{EVAL_CANDIDATE_ID}'")


def _cleanup_eval_collections():
    """Delete the evaluation ChromaDB collections (chunks + summaries)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    for name in [EVAL_CANDIDATE_ID, f"{EVAL_CANDIDATE_ID}_summaries"]:
        try:
            client.delete_collection(name=name)
            print(f"[Harness] Cleaned up collection '{name}'")
        except Exception:
            pass


def _run_pipeline_on_dataset(
    dataset: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    Run the RAG pipeline on every question in the golden dataset.

    Returns list of dicts with keys:
        id, question, answer, contexts, ground_truth, category, expected_source,
        difficulty, expected_route, tool_trajectory, final_tool, route, latency_s
    """
    results = []
    total = len(dataset)

    for i, item in enumerate(dataset):
        question = item["question"]
        print(f"\n[Harness] ({i+1}/{total}) Processing: {question[:80]}...")

        start = time.time()
        try:
            pipeline_result = run_full_pipeline(
                question=question,
                candidate_id=EVAL_CANDIDATE_ID,
                top_k=top_k,
            )
            elapsed = time.time() - start
            print(f"  → Answer: {pipeline_result['answer'][:100]}... ({elapsed:.1f}s)")
            print(f"  → Tool: {pipeline_result['final_tool']} | Route: {pipeline_result['route']}")
        except Exception as e:
            print(f"  → ERROR: {e}")
            pipeline_result = {
                "answer": f"[ERROR] {e}",
                "contexts": [],
                "tool_trajectory": [],
                "final_tool": None,
                "route": None,
            }
            elapsed = time.time() - start

        results.append({
            "id": item["id"],
            "question": question,
            "answer": pipeline_result["answer"],
            "contexts": pipeline_result["contexts"],
            "ground_truth": item["ground_truth"],
            "category": item["category"],
            "expected_source": item["expected_source"],
            "difficulty": item["difficulty"],
            "expected_route": item.get("expected_route"),
            "tool_trajectory": pipeline_result["tool_trajectory"],
            "final_tool": pipeline_result["final_tool"],
            "route": pipeline_result["route"],
            "latency_s": round(elapsed, 2),
        })

    return results


# ── Component runners ───────────────────────────────────────────────────────

def _run_tool_selection(pipeline_results: list[dict]) -> pd.DataFrame | None:
    """Evaluate whether the agent picked the expected tool."""
    from evaluation.evaluators.tool_evaluator import run_tool_evaluation
    df = run_tool_evaluation(pipeline_results)
    df.to_csv(REPORTS_DIR / "tool_selection_scores.csv", index=False)
    return df


def _run_rag_quality(pipeline_results: list[dict], judge_model: str) -> tuple:
    """Evaluate RAG quality (RAGAS + Hallucination) on RAG-routed questions only."""
    rag_data = [
        r for r in pipeline_results
        if r.get("final_tool") == "search_documents" and r.get("contexts")
    ]
    print(f"[Harness] {len(rag_data)} questions routed through RAG")

    if not rag_data:
        print("[Harness] No RAG-routed questions — skipping RAGAS and Hallucination")
        return None, None

    from evaluation.evaluators.ragas_evaluator import run_ragas_evaluation
    ragas_df = run_ragas_evaluation(rag_data, judge_model=judge_model)
    ragas_df.to_csv(REPORTS_DIR / "ragas_scores.csv", index=False)

    from evaluation.evaluators.deepeval_evaluator import run_deepeval_hallucination
    hallucination_df = run_deepeval_hallucination(rag_data, judge_model=judge_model)
    hallucination_df.to_csv(REPORTS_DIR / "hallucination_scores.csv", index=False)

    return ragas_df, hallucination_df


def _run_geval(pipeline_results: list[dict], judge_model: str) -> pd.DataFrame | None:
    """Evaluate answer correctness via GEval on all non-negative questions."""
    geval_data = [r for r in pipeline_results if r["category"] != "negative"]
    print(f"[Harness] {len(geval_data)} questions for GEval")

    if not geval_data:
        return None

    from evaluation.evaluators.deepeval_evaluator import run_deepeval_geval
    df = run_deepeval_geval(geval_data, judge_model=judge_model)
    df.to_csv(REPORTS_DIR / "geval_scores.csv", index=False)
    return df


def _run_refusal(pipeline_results: list[dict]) -> pd.DataFrame | None:
    """Evaluate whether the agent correctly refuses negative questions."""
    negative_data = [r for r in pipeline_results if r["category"] == "negative"]
    print(f"[Harness] {len(negative_data)} negative questions")

    if not negative_data:
        return None

    from evaluation.evaluators.refusal_evaluator import run_refusal_evaluation
    df = run_refusal_evaluation(negative_data)
    df.to_csv(REPORTS_DIR / "refusal_scores.csv", index=False)
    return df


def _run_ingestion(judge_model: str) -> dict | None:
    """Evaluate ingestion quality (chunk stats, coverage, summary quality)."""
    from evaluation.evaluators.ingestion_evaluator import run_ingestion_evaluation
    report = run_ingestion_evaluation(EVAL_CANDIDATE_ID, judge_model=judge_model)
    with open(REPORTS_DIR / "ingestion_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def _run_router(pipeline_results: list[dict]) -> pd.DataFrame | None:
    """Evaluate broad/specific routing accuracy."""
    router_data = [r for r in pipeline_results if r.get("expected_route") is not None]
    print(f"[Harness] {len(router_data)} questions with expected route")

    if not router_data:
        return None

    from evaluation.evaluators.router_evaluator import run_router_evaluation
    df = run_router_evaluation(router_data)
    df.to_csv(REPORTS_DIR / "router_scores.csv", index=False)
    return df


# ── Component dispatch table ────────────────────────────────────────────────

ALL_COMPONENTS = ["tool_selection", "rag", "geval", "refusal", "ingestion", "router"]


# ── Main entry point ────────────────────────────────────────────────────────

def run_evaluation(
    components: list[str] | None = None,
    category_filter: str | None = None,
    top_k: int = 3,
    judge_model: str = "qwen3",
    dry_run: bool = False,
    report_format: str = "html",
    reuse_results: bool = False,
) -> dict:
    """
    Full component-based evaluation pipeline:
      1. Seed data (structured + documents)
      2. Run agent pipeline on golden dataset (captures tool trajectory)
      3. Evaluate each component
      4. Generate report

    Args:
        components: List of components to run. Options:
            "tool_selection", "rag", "geval", "refusal", "ingestion", "router"
            None means all components.
        category_filter: Optional category name to filter the dataset
        top_k: Number of chunks to retrieve
        judge_model: Ollama model for LLM-as-judge
        dry_run: If True, skip metric computation (just run pipeline)
        report_format: "html", "json", or "csv"
        reuse_results: If True, load existing pipeline_results.json

    Returns:
        Dict with all evaluation results.
    """
    if components is None:
        components = ALL_COMPONENTS

    print("=" * 70)
    print("  COMPONENT-BASED EVALUATION HARNESS")
    print("=" * 70)
    print(f"  Components : {components}")
    print(f"  Judge model: {judge_model}")
    print(f"  Category   : {category_filter or 'all'}")
    print(f"  Top-K      : {top_k}")
    print(f"  Dry run    : {dry_run}")
    print(f"  Reuse      : {reuse_results}")
    print("=" * 70)

    # ── Step 1: Load dataset ────────────────────────────────────────
    dataset = _load_golden_dataset(category_filter)
    print(f"\n[Harness] Loaded {len(dataset)} questions from golden dataset.")

    backup_path = None

    if reuse_results:
        raw_path = REPORTS_DIR / "pipeline_results.json"
        with open(raw_path, "r", encoding="utf-8") as f:
            pipeline_results = json.load(f)
        if category_filter:
            pipeline_results = [r for r in pipeline_results if r["category"] == category_filter]
        print(f"[Harness] Reusing {len(pipeline_results)} results from {raw_path}")
    else:
        # ── Step 2: Seed data ───────────────────────────────────────
        backup_path = _seed_structured_data()
        _cleanup_eval_collections()
        _seed_documents()
        set_candidate_id(EVAL_CANDIDATE_ID)

    try:
        if not reuse_results:
            # ── Step 3: Run pipeline ────────────────────────────────
            start_time = time.time()
            pipeline_results = _run_pipeline_on_dataset(dataset, top_k=top_k)
            pipeline_elapsed = time.time() - start_time
            print(f"\n[Harness] Pipeline completed in {pipeline_elapsed:.1f}s")

            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            raw_path = REPORTS_DIR / "pipeline_results.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(pipeline_results, f, indent=2, ensure_ascii=False)
            print(f"[Harness] Raw results saved to {raw_path}")

        # ── Step 4: Component Evaluations ───────────────────────────
        eval_results = {
            "pipeline_results": pipeline_results,
            "tool_eval_df": None,
            "ragas_df": None,
            "hallucination_df": None,
            "geval_df": None,
            "refusal_df": None,
            "ingestion_report": None,
            "router_df": None,
            "report_path": None,
        }

        if dry_run:
            print("\n[Harness] Dry run — skipping all evaluations")
        else:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)

            for component in components:
                print(f"\n{'─' * 50}")
                print(f"  Component: {component}")
                print("─" * 50)

                if component == "tool_selection":
                    eval_results["tool_eval_df"] = _run_tool_selection(pipeline_results)

                elif component == "rag":
                    ragas_df, hall_df = _run_rag_quality(pipeline_results, judge_model)
                    eval_results["ragas_df"] = ragas_df
                    eval_results["hallucination_df"] = hall_df

                elif component == "geval":
                    eval_results["geval_df"] = _run_geval(pipeline_results, judge_model)

                elif component == "refusal":
                    eval_results["refusal_df"] = _run_refusal(pipeline_results)

                elif component == "ingestion":
                    eval_results["ingestion_report"] = _run_ingestion(judge_model)

                elif component == "router":
                    eval_results["router_df"] = _run_router(pipeline_results)

        # ── Step 5: Generate report ─────────────────────────────────
        if not dry_run:
            from evaluation.report import generate_report
            report_path = generate_report(
                pipeline_results=pipeline_results,
                eval_results=eval_results,
                output_format=report_format,
            )
            eval_results["report_path"] = report_path
            print(f"\n[Harness] Report generated: {report_path}")

    finally:
        if not reuse_results:
            restore_candidate_id()
            _restore_structured_data(backup_path)
            _cleanup_eval_collections()

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)

    return eval_results

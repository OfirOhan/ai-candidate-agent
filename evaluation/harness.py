"""
Evaluation harness — orchestrates data seeding, pipeline execution,
metric evaluation, and report generation.
"""

import json
import os
import shutil
import time
from pathlib import Path

import chromadb
import pandas as pd

from evaluation.pipeline import (
    ingest_text_for_eval,
    run_full_pipeline,
    set_candidate_id,
    restore_candidate_id,
)
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
        # No original file existed — remove the seeded one
        if os.path.exists(STRUCTURED_DATA_PATH):
            os.remove(STRUCTURED_DATA_PATH)


def _seed_documents():
    """Ingest the synthetic resume into ChromaDB under the eval collection."""
    with open(RESUME_PATH, "r", encoding="utf-8") as f:
        resume_text = f.read()
    ingest_text_for_eval(resume_text, EVAL_CANDIDATE_ID)
    print(f"[Harness] Ingested synthetic resume into collection '{EVAL_CANDIDATE_ID}'")


def _cleanup_eval_collection():
    """Delete the evaluation ChromaDB collection."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        client.delete_collection(name=EVAL_CANDIDATE_ID)
        print(f"[Harness] Cleaned up ChromaDB collection '{EVAL_CANDIDATE_ID}'")
    except Exception:
        pass  # Collection may not exist


def _run_pipeline_on_dataset(
    dataset: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    Run the RAG pipeline on every question in the golden dataset.

    Returns list of dicts with keys:
        question, answer, contexts, ground_truth, category, expected_source, difficulty
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
        except Exception as e:
            print(f"  → ERROR: {e}")
            pipeline_result = {"answer": f"[ERROR] {e}", "contexts": []}
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
            "latency_s": round(elapsed, 2),
        })

    return results


def run_evaluation(
    frameworks: list[str],
    category_filter: str | None = None,
    top_k: int = 3,
    judge_model: str = "qwen3",
    dry_run: bool = False,
    report_format: str = "html",
    reuse_results: bool = False,
) -> dict:
    """
    Full evaluation pipeline:
      1. Seed data (structured + documents)
      2. Run RAG pipeline on golden dataset
      3. Evaluate with RAGAS and/or DeepEval
      4. Generate report

    Args:
        frameworks: List of frameworks to use, e.g. ["ragas", "deepeval"]
        category_filter: Optional category name to filter the dataset
        top_k: Number of chunks to retrieve
        judge_model: Ollama model for LLM-as-judge
        dry_run: If True, skip metric computation (just run pipeline)
        report_format: "html", "json", or "csv"

    Returns:
        Dict with pipeline_results, ragas_df, deepeval_df, and report_path.
    """
    print("=" * 70)
    print("  RAG EVALUATION HARNESS")
    print("=" * 70)
    print(f"  Frameworks : {frameworks}")
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
        # ── Load existing pipeline results ──────────────────────────
        raw_path = REPORTS_DIR / "pipeline_results.json"
        with open(raw_path, "r", encoding="utf-8") as f:
            pipeline_results = json.load(f)
        if category_filter:
            pipeline_results = [r for r in pipeline_results if r["category"] == category_filter]
        print(f"[Harness] Reusing {len(pipeline_results)} results from {raw_path}")
    else:
        # ── Step 2: Seed data ───────────────────────────────────────
        backup_path = _seed_structured_data()
        _cleanup_eval_collection()
        _seed_documents()
        set_candidate_id(EVAL_CANDIDATE_ID)

    try:
        if not reuse_results:
            # ── Step 3: Run pipeline ────────────────────────────────
            start_time = time.time()
            pipeline_results = _run_pipeline_on_dataset(dataset, top_k=top_k)
            pipeline_elapsed = time.time() - start_time
            print(f"\n[Harness] Pipeline completed in {pipeline_elapsed:.1f}s")

            # Save raw results
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            raw_path = REPORTS_DIR / "pipeline_results.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(pipeline_results, f, indent=2, ensure_ascii=False)
            print(f"[Harness] Raw results saved to {raw_path}")

        # ── Step 4: Evaluate ────────────────────────────────────────
        ragas_df = None
        deepeval_df = None

        if not dry_run:
            # Filter to questions with contexts for retrieval metrics
            eval_data = [
                r for r in pipeline_results
                if r["expected_source"] not in ("none",)
            ]

            if "ragas" in frameworks:
                from evaluation.evaluators.ragas_evaluator import run_ragas_evaluation
                ragas_df = run_ragas_evaluation(eval_data, judge_model=judge_model)
                ragas_csv = REPORTS_DIR / "ragas_scores.csv"
                ragas_df.to_csv(ragas_csv, index=False)
                print(f"[Harness] RAGAS scores saved to {ragas_csv}")

            if "deepeval" in frameworks:
                from evaluation.evaluators.deepeval_evaluator import run_deepeval_evaluation
                deepeval_df = run_deepeval_evaluation(eval_data, judge_model=judge_model)
                deepeval_csv = REPORTS_DIR / "deepeval_scores.csv"
                deepeval_df.to_csv(deepeval_csv, index=False)
                print(f"[Harness] DeepEval scores saved to {deepeval_csv}")

        # ── Step 5: Generate report ─────────────────────────────────
        report_path = None
        if not dry_run:
            from evaluation.report import generate_report
            report_path = generate_report(
                pipeline_results=pipeline_results,
                ragas_df=ragas_df,
                deepeval_df=deepeval_df,
                output_format=report_format,
            )
            print(f"\n[Harness] Report generated: {report_path}")

    finally:
        # ── Cleanup ─────────────────────────────────────────────────
        if not reuse_results:
            restore_candidate_id()
            _restore_structured_data(backup_path)
            _cleanup_eval_collection()

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)

    return {
        "pipeline_results": pipeline_results,
        "ragas_df": ragas_df,
        "deepeval_df": deepeval_df,
        "report_path": report_path,
    }

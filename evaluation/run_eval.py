"""
RAG Evaluation Runner

Configure evaluation settings below, then run:
    python -m evaluation.run_eval
"""

from evaluation.harness import run_evaluation

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these values instead of using CLI arguments
# ═══════════════════════════════════════════════════════════════════════════

# Which evaluation frameworks to run: "ragas", "deepeval", or both
FRAMEWORKS = ["ragas", "deepeval"]

# Filter golden dataset to a specific category, or None for all
# Options: "personal", "education", "experience", "skills", "projects",
#          "certifications", "preferences", "scheduling", "negative", "complex"
CATEGORY_FILTER = None

# Report output format: "html", "json", or "csv"
REPORT_FORMAT = "html"

# If True, only run the RAG pipeline (collect answers) without scoring
DRY_RUN = False

# Number of chunks the retriever returns
TOP_K = 3

# Ollama model used as LLM judge for RAGAS and DeepEval metrics
JUDGE_MODEL = "qwen3"

# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    results = run_evaluation(
        frameworks=FRAMEWORKS,
        category_filter=CATEGORY_FILTER,
        top_k=TOP_K,
        judge_model=JUDGE_MODEL,
        dry_run=DRY_RUN,
        report_format=REPORT_FORMAT,
    )

    # Print summary
    if results["ragas_df"] is not None:
        print("\n── RAGAS Summary ──")
        df = results["ragas_df"]
        metric_cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
        for col in metric_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f"  {col:30s}  mean={vals.mean():.4f}  min={vals.min():.4f}  max={vals.max():.4f}")

    if results["deepeval_df"] is not None:
        print("\n── DeepEval Summary ──")
        df = results["deepeval_df"]
        metric_cols = [c for c in df.columns if c.startswith("deepeval_")]
        for col in metric_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                label = col.replace("deepeval_", "")
                print(f"  {label:30s}  mean={vals.mean():.4f}  min={vals.min():.4f}  max={vals.max():.4f}")

    if results["report_path"]:
        print(f"\n✅ Report saved to: {results['report_path']}")

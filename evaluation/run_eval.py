"""
Component-Based Evaluation Runner

Configure evaluation settings below, then run:
    python -m evaluation.run_eval
"""

from evaluation.harness import run_evaluation

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these values instead of using CLI arguments
# ═══════════════════════════════════════════════════════════════════════════

# Which evaluation components to run (None = all):
# Options: "tool_selection", "rag", "geval", "refusal", "ingestion", "router"
COMPONENTS = None  # None means all

# Filter golden dataset to a specific category, or None for all
# Options: "personal", "education", "experience", "skills", "projects",
#          "certifications", "preferences", "negative", "complex"
CATEGORY_FILTER = None

# Report output format: "html", "json", or "csv"
REPORT_FORMAT = "html"

# If True, only run the RAG pipeline (collect answers) without scoring
DRY_RUN = False

# If True, skip the RAG pipeline and reuse existing pipeline_results.json
REUSE_PIPELINE_RESULTS = False

# Number of chunks the retriever returns
TOP_K = 3

# Ollama model used as LLM judge for RAGAS and DeepEval metrics
JUDGE_MODEL = "qwen3"

# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    results = run_evaluation(
        components=COMPONENTS,
        category_filter=CATEGORY_FILTER,
        top_k=TOP_K,
        judge_model=JUDGE_MODEL,
        dry_run=DRY_RUN,
        report_format=REPORT_FORMAT,
        reuse_results=REUSE_PIPELINE_RESULTS,
    )

    # ── Tool Selection Summary ──
    if results["tool_eval_df"] is not None:
        print("\n── Tool Selection Summary ──")
        df = results["tool_eval_df"]
        total = len(df)
        correct = df["tool_correct"].sum()
        print(f"  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
        print(f"  Fallbacks: {df['used_fallback'].sum()}")
        print(f"  Missing fallbacks: {df['missing_fallback'].sum()}")

    # ── RAGAS Summary ──
    if results["ragas_df"] is not None:
        print("\n── RAGAS Summary (RAG-only) ──")
        df = results["ragas_df"]
        metric_cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
        for col in metric_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f"  {col:30s}  mean={vals.mean():.4f}  min={vals.min():.4f}  max={vals.max():.4f}")

    # ── Hallucination Summary ──
    if results["hallucination_df"] is not None:
        print("\n── Hallucination Summary (RAG-only) ──")
        df = results["hallucination_df"]
        vals = df["deepeval_hallucination"].dropna()
        if len(vals) > 0:
            print(f"  {'Hallucination':30s}  mean={vals.mean():.4f}  min={vals.min():.4f}  max={vals.max():.4f}")

    # ── GEval Summary ──
    if results["geval_df"] is not None:
        print("\n── GEval Correctness Summary (All Questions) ──")
        df = results["geval_df"]
        vals = df["deepeval_correctness"].dropna()
        if len(vals) > 0:
            print(f"  {'Correctness':30s}  mean={vals.mean():.4f}  min={vals.min():.4f}  max={vals.max():.4f}")

    # ── Refusal Summary ──
    if results["refusal_df"] is not None:
        print("\n── Refusal Accuracy Summary ──")
        df = results["refusal_df"]
        total = len(df)
        print(f"  Refusal accuracy: {df['refused_correctly'].sum()}/{total}")
        print(f"  Hallucinations: {df['hallucinated'].sum()}/{total}")
        print(f"  Professional redirects: {df['professional_redirect'].sum()}/{total}")

    # ── Router Summary ──
    if results["router_df"] is not None:
        print("\n── Router Accuracy Summary ──")
        df = results["router_df"]
        total = len(df)
        correct = df["route_correct"].sum()
        print(f"  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

    # ── Ingestion Summary ──
    if results["ingestion_report"] is not None:
        print("\n── Ingestion Quality Summary ──")
        report = results["ingestion_report"]
        cs = report.get("chunk_stats", {})
        print(f"  Chunks: {cs.get('total_chunks', 0)} (avg size: {cs.get('avg_chunk_size', 0)})")
        sc = report.get("section_coverage", {})
        print(f"  Section coverage: {sc.get('coverage_pct', 0)}%")
        sq = report.get("summary_quality", {})
        print(f"  Summary quality: {sq.get('llm_score', 'N/A')}")
        ep = report.get("embedding_probes", {})
        print(f"  Embedding hit rate: {ep.get('hit_rate', 0)*100:.0f}%")

    if results["report_path"]:
        print(f"\n✅ Report saved to: {results['report_path']}")

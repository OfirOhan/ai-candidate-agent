"""
Component-Based Evaluation Runner (Multi-Candidate)

Configure evaluation settings below, then run:
    python -m evaluation.run_eval
"""

from evaluation.harness import run_evaluation

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Which candidates to evaluate (None = all, or list of indices e.g. [1, 4])
CANDIDATES = None

# Which evaluation components to run (None = all):
# Options: "tool_selection", "rag", "geval", "refusal", "ingestion", "router"
COMPONENTS = None

# Filter golden dataset to a specific category, or None for all
CATEGORY_FILTER = None

# Report output format: "html", "json", or "csv"
REPORT_FORMAT = "html"

# If True, only run the RAG pipeline (collect answers) without scoring
DRY_RUN = False

# If True, skip the RAG pipeline and reuse existing pipeline_results.json
REUSE_PIPELINE_RESULTS = False

# Number of chunks the retriever returns
TOP_K = 3

# Ollama model used as LLM judge
JUDGE_MODEL = "qwen3"

# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    results = run_evaluation(
        candidates=CANDIDATES,
        components=COMPONENTS,
        category_filter=CATEGORY_FILTER,
        top_k=TOP_K,
        judge_model=JUDGE_MODEL,
        dry_run=DRY_RUN,
        report_format=REPORT_FORMAT,
        reuse_results=REUSE_PIPELINE_RESULTS,
    )

    # ── Candidates Summary ──
    if results.get("candidates"):
        print("\n── Candidates ──")
        for c in results["candidates"]:
            print(f"  {c['name']} ({c['eval_id']}): {c['question_count']} questions, {c['doc_count']} docs")

    # ── Tool Selection Summary ──
    if results["tool_eval_df"] is not None:
        print("\n── Tool Selection Summary ──")
        df = results["tool_eval_df"]
        total = len(df)
        correct = df["tool_correct"].sum()
        print(f"  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
        # Per-candidate breakdown
        if "candidate_name" in df.columns:
            for name, group in df.groupby("candidate_name"):
                gc = group["tool_correct"].sum()
                print(f"    {name}: {gc}/{len(group)} ({gc/len(group)*100:.1f}%)")

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
            print(f"  {'Hallucination':30s}  mean={vals.mean():.4f}")

    # ── GEval Summary ──
    if results["geval_df"] is not None:
        print("\n── GEval Correctness Summary ──")
        df = results["geval_df"]
        vals = df["deepeval_correctness"].dropna()
        if len(vals) > 0:
            print(f"  {'Correctness':30s}  mean={vals.mean():.4f}")

    # ── Refusal Summary ──
    if results["refusal_df"] is not None:
        print("\n── Refusal Summary ──")
        df = results["refusal_df"]
        tp = (df["classification"] == "TP").sum()
        tn = (df["classification"] == "TN").sum()
        fp = (df["classification"] == "FP").sum()
        fn = (df["classification"] == "FN").sum()
        accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
        print(f"  Accuracy: {accuracy:.1%} (TP={tp} TN={tn} FP={fp} FN={fn})")
        print(f"  Hallucinations: {df['hallucinated'].sum()}")

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
        for eval_id, entry in results["ingestion_report"].items():
            report = entry["report"]
            name = entry["name"]
            cs = report.get("chunk_stats", {})
            sq = report.get("summary_quality", {})
            ep = report.get("embedding_probes", {})
            print(f"  {name}:")
            print(f"    Chunks: {cs.get('total_chunks', 0)} (avg size: {cs.get('avg_chunk_size', 0)})")
            print(f"    Summary score: {sq.get('llm_score', 'N/A')}")
            print(f"    Embedding hit rate: {ep.get('hit_rate', 0)*100:.0f}%")

    if results.get("report_path"):
        print(f"\n✅ Report saved to: {results['report_path']}")

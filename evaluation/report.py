"""
Report generator — produces HTML, JSON, or CSV evaluation reports.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path(__file__).parent / "reports"


def _compute_summary(pipeline_results: list[dict], ragas_df, deepeval_df) -> dict:
    """Compute aggregate summary statistics."""
    summary = {
        "total_questions": len(pipeline_results),
        "avg_latency_s": round(
            sum(r["latency_s"] for r in pipeline_results) / len(pipeline_results), 2
        ),
        "categories": {},
        "ragas_means": {},
        "deepeval_means": {},
    }

    # Category breakdown
    cats = {}
    for r in pipeline_results:
        cat = r["category"]
        if cat not in cats:
            cats[cat] = {"count": 0, "avg_latency": 0}
        cats[cat]["count"] += 1
        cats[cat]["avg_latency"] += r["latency_s"]
    for cat in cats:
        cats[cat]["avg_latency"] = round(cats[cat]["avg_latency"] / cats[cat]["count"], 2)
    summary["categories"] = cats

    # RAGAS means
    if ragas_df is not None:
        metric_cols = [c for c in ragas_df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
        for col in metric_cols:
            vals = ragas_df[col].dropna()
            if len(vals) > 0:
                summary["ragas_means"][col] = round(vals.mean(), 4)

    # DeepEval means
    if deepeval_df is not None:
        metric_cols = [c for c in deepeval_df.columns if c.startswith("deepeval_")]
        for col in metric_cols:
            vals = deepeval_df[col].dropna()
            if len(vals) > 0:
                summary["deepeval_means"][col] = round(vals.mean(), 4)

    return summary


def _score_color(score) -> str:
    """Return CSS color for a metric score."""
    if score is None or pd.isna(score):
        return "#888"
    if score >= 0.8:
        return "#22c55e"
    if score >= 0.5:
        return "#eab308"
    return "#ef4444"


def _format_score(score) -> str:
    """Format a score for display."""
    if score is None or (isinstance(score, float) and pd.isna(score)):
        return "N/A"
    return f"{score:.3f}"


def _generate_html(pipeline_results, ragas_df, deepeval_df, summary) -> str:
    """Generate a self-contained HTML report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build metric cards
    metric_cards_html = ""
    all_means = {**summary.get("ragas_means", {}), **summary.get("deepeval_means", {})}
    for metric_name, mean_val in all_means.items():
        color = _score_color(mean_val)
        label = metric_name.replace("deepeval_", "DE: ").replace("_", " ").title()
        metric_cards_html += f"""
        <div class="metric-card">
          <div class="metric-score" style="color:{color}">{_format_score(mean_val)}</div>
          <div class="metric-label">{label}</div>
        </div>"""

    # Build category breakdown rows
    cat_rows = ""
    for cat, info in summary["categories"].items():
        # Get per-category metric means
        cat_metrics = ""
        if ragas_df is not None:
            cat_indices = [i for i, r in enumerate(pipeline_results) if r["category"] == cat]
            if cat_indices:
                cat_subset = ragas_df.iloc[[idx for idx in cat_indices if idx < len(ragas_df)]]
                metric_cols = [c for c in ragas_df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
                for col in metric_cols:
                    vals = cat_subset[col].dropna()
                    if len(vals) > 0:
                        v = vals.mean()
                        cat_metrics += f'<span style="color:{_score_color(v)}">{col}: {_format_score(v)}</span> '

        cat_rows += f"""
        <tr>
          <td>{cat}</td>
          <td>{info['count']}</td>
          <td>{info['avg_latency']}s</td>
          <td style="font-size:0.85em">{cat_metrics or 'N/A'}</td>
        </tr>"""

    # Build per-question detail rows
    detail_rows = ""
    for i, r in enumerate(pipeline_results):
        # Collect scores for this question
        scores_html = ""
        if ragas_df is not None and i < len(ragas_df):
            metric_cols = [c for c in ragas_df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
            for col in metric_cols:
                val = ragas_df.iloc[i][col]
                color = _score_color(val)
                scores_html += f'<span style="color:{color}" title="{col}">{_format_score(val)}</span> '

        if deepeval_df is not None and i < len(deepeval_df):
            metric_cols = [c for c in deepeval_df.columns if c.startswith("deepeval_")]
            for col in metric_cols:
                val = deepeval_df.iloc[i][col]
                color = _score_color(val)
                label = col.replace("deepeval_", "DE:")
                scores_html += f'<span style="color:{color}" title="{label}">{_format_score(val)}</span> '

        answer_preview = r["answer"][:150].replace("<", "&lt;").replace(">", "&gt;")
        gt_preview = r["ground_truth"][:150].replace("<", "&lt;").replace(">", "&gt;")

        detail_rows += f"""
        <tr>
          <td>{r['id']}</td>
          <td>{r['category']}</td>
          <td title="{r['question']}">{r['question'][:60]}</td>
          <td class="answer-cell" title="{answer_preview}">{answer_preview}...</td>
          <td class="gt-cell" title="{gt_preview}">{gt_preview}...</td>
          <td>{scores_html or 'N/A'}</td>
          <td>{r['latency_s']}s</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RAG Evaluation Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f172a; color: #e2e8f0;
    padding: 2rem; line-height: 1.6;
  }}
  h1 {{ color: #f8fafc; margin-bottom: 0.5rem; font-size: 1.8rem; }}
  h2 {{ color: #94a3b8; margin: 2rem 0 1rem; font-size: 1.3rem; border-bottom: 1px solid #334155; padding-bottom: 0.5rem; }}
  .meta {{ color: #64748b; margin-bottom: 2rem; font-size: 0.9rem; }}
  .metrics-grid {{
    display: flex; flex-wrap: wrap; gap: 1rem; margin: 1.5rem 0;
  }}
  .metric-card {{
    background: #1e293b; border: 1px solid #334155; border-radius: 12px;
    padding: 1.2rem 1.5rem; min-width: 160px; text-align: center;
  }}
  .metric-score {{ font-size: 1.8rem; font-weight: 700; }}
  .metric-label {{ font-size: 0.8rem; color: #94a3b8; margin-top: 0.3rem; }}
  table {{
    width: 100%; border-collapse: collapse; margin: 1rem 0;
    background: #1e293b; border-radius: 8px; overflow: hidden;
  }}
  th {{ background: #334155; color: #e2e8f0; padding: 0.75rem 1rem; text-align: left; font-size: 0.85rem; }}
  td {{ padding: 0.6rem 1rem; border-bottom: 1px solid #334155; font-size: 0.85rem; }}
  tr:hover {{ background: #334155; }}
  .answer-cell, .gt-cell {{ max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .summary-stat {{ display: inline-block; background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 0.8rem 1.2rem; margin: 0.3rem; }}
  .summary-stat strong {{ color: #38bdf8; }}
</style>
</head>
<body>

<h1>RAG Evaluation Report</h1>
<div class="meta">Generated: {timestamp} | Questions: {summary['total_questions']} | Avg Latency: {summary['avg_latency_s']}s</div>

<h2>Overall Metric Scores</h2>
<div class="metrics-grid">
  {metric_cards_html if metric_cards_html else '<p style="color:#64748b">No metrics computed (dry run?)</p>'}
</div>

<h2>Category Breakdown</h2>
<table>
  <thead><tr><th>Category</th><th>Count</th><th>Avg Latency</th><th>Metrics</th></tr></thead>
  <tbody>{cat_rows}</tbody>
</table>

<h2>Per-Question Details</h2>
<table>
  <thead><tr><th>ID</th><th>Category</th><th>Question</th><th>Answer</th><th>Ground Truth</th><th>Scores</th><th>Latency</th></tr></thead>
  <tbody>{detail_rows}</tbody>
</table>

</body>
</html>"""
    return html


def generate_report(
    pipeline_results: list[dict],
    ragas_df=None,
    deepeval_df=None,
    output_format: str = "html",
) -> str:
    """
    Generate an evaluation report.

    Args:
        pipeline_results: List of pipeline result dicts
        ragas_df: RAGAS results DataFrame (or None)
        deepeval_df: DeepEval results DataFrame (or None)
        output_format: "html", "json", or "csv"

    Returns:
        Path to the generated report file.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = _compute_summary(pipeline_results, ragas_df, deepeval_df)

    if output_format == "html":
        html = _generate_html(pipeline_results, ragas_df, deepeval_df, summary)
        path = REPORTS_DIR / f"eval_report_{timestamp}.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    elif output_format == "json":
        report_data = {
            "summary": summary,
            "results": pipeline_results,
        }
        if ragas_df is not None:
            report_data["ragas_scores"] = ragas_df.to_dict(orient="records")
        if deepeval_df is not None:
            report_data["deepeval_scores"] = deepeval_df.to_dict(orient="records")

        path = REPORTS_DIR / f"eval_report_{timestamp}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

    elif output_format == "csv":
        # Merge all results into a single flat table
        df = pd.DataFrame(pipeline_results)
        if ragas_df is not None:
            metric_cols = [c for c in ragas_df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
            for col in metric_cols:
                if len(ragas_df) == len(df):
                    df[f"ragas_{col}"] = ragas_df[col].values

        if deepeval_df is not None:
            metric_cols = [c for c in deepeval_df.columns if c.startswith("deepeval_")]
            for col in metric_cols:
                if len(deepeval_df) == len(df):
                    df[col] = deepeval_df[col].values

        path = REPORTS_DIR / f"eval_report_{timestamp}.csv"
        df.to_csv(path, index=False)

    else:
        raise ValueError(f"Unknown format: {output_format}")

    return str(path)

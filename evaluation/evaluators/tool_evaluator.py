"""
Tool Selection Evaluator — compares the agent's tool choices against
the golden dataset's expected_source field.

Metrics:
  - Tool Accuracy: Did the final tool match expected_source?
  - Fallback Rate: How often did the agent fall back from structured → docs?
  - Missing Fallback Rate: Didn't fall back when it should have
"""

import pandas as pd


def _classify_actual_tool(trajectory: list[dict]) -> str:
    """Determine the effective tool source from a tool trajectory.

    Returns: "structured", "docs", or "none"
    """
    if not trajectory:
        return "none"

    tool_names = [t["tool"] for t in trajectory]

    # If search_documents was used at any point, the effective source is docs
    if "search_documents" in tool_names:
        return "docs"

    # If only get_structured_data was used
    if "get_structured_data" in tool_names:
        return "structured"

    return "none"


def _detect_fallback(trajectory: list[dict]) -> bool:
    """Check if the agent fell back from structured → docs."""
    tool_names = [t["tool"] for t in trajectory]
    if len(tool_names) < 2:
        return False
    # Fallback = structured was called first, then docs was called
    structured_idx = next((i for i, t in enumerate(tool_names) if t == "get_structured_data"), None)
    docs_idx = next((i for i, t in enumerate(tool_names) if t == "search_documents"), None)
    return structured_idx is not None and docs_idx is not None and structured_idx < docs_idx


def _detect_missing_fallback(trajectory: list[dict], expected_source: str) -> bool:
    """Check if the agent should have fallen back but didn't.

    This occurs when:
    - expected_source is "docs" (answer requires document search)
    - agent only used get_structured_data (never called search_documents)
    """
    if expected_source != "docs":
        return False
    actual = _classify_actual_tool(trajectory)
    return actual == "structured"


def run_tool_evaluation(data: list[dict]) -> pd.DataFrame:
    """
    Evaluate tool selection for each question.

    Args:
        data: List of pipeline result dicts, each with:
            - question (str)
            - expected_source (str): "structured", "docs", or "none"
            - tool_trajectory (list[dict])

    Returns:
        DataFrame with columns:
            id, question, expected_tool, actual_tool, tool_correct,
            used_fallback, missing_fallback, trajectory_summary
    """
    rows = []
    for d in data:
        trajectory = d.get("tool_trajectory", [])
        expected = d["expected_source"]
        actual = _classify_actual_tool(trajectory)
        fallback = _detect_fallback(trajectory)
        missing_fb = _detect_missing_fallback(trajectory, expected)

        # Build a short summary of the trajectory
        traj_summary = " → ".join(t["tool"] for t in trajectory) if trajectory else "no tools"

        rows.append({
            "id": d["id"],
            "question": d["question"],
            "candidate_id": d.get("candidate_id", ""),
            "candidate_name": d.get("candidate_name", ""),
            "expected_tool": expected,
            "actual_tool": actual,
            "tool_correct": expected == actual,
            "used_fallback": fallback,
            "missing_fallback": missing_fb,
            "trajectory_summary": traj_summary,
        })

    df = pd.DataFrame(rows)

    # Print summary
    total = len(df)
    correct = df["tool_correct"].sum()
    fallbacks = df["used_fallback"].sum()
    missing = df["missing_fallback"].sum()
    print(f"[Tool Eval] Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"[Tool Eval] Fallback rate: {fallbacks}/{total}")
    print(f"[Tool Eval] Missing fallback rate: {missing}/{total}")

    return df

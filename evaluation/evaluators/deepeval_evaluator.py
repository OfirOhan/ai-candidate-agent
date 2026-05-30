"""
DeepEval evaluator — wraps DeepEval with a custom Ollama-based judge LLM.

Metrics computed:
  - ContextualRelevancy:  Are retrieved chunks relevant to the query?
  - Faithfulness:         Is the answer factually aligned with context?
  - GEval (Correctness):  Custom LLM-judge scoring answer correctness
  - Hallucination:        Does the answer contain fabricated information?
"""

import json
import re

import pandas as pd
import ollama as ollama_client
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams


# ---------------------------------------------------------------------------
# Custom Ollama model adapter for DeepEval
# ---------------------------------------------------------------------------

class OllamaJudge(DeepEvalBaseLLM):
    """Wraps a local Ollama model so DeepEval can use it as evaluator.

    DeepEval's latest API passes a Pydantic ``schema`` to ``generate()``.
    When a schema is provided we force Ollama to return JSON and parse the
    response into the expected Pydantic model so that DeepEval can access
    attributes like ``.verdicts``, ``.truths``, ``.steps``, etc.
    """

    def __init__(self, model_name: str = "qwen3"):
        self._model_name = model_name

    def load_model(self):
        return self._model_name

    def _strip_think(self, text: str) -> str:
        """Remove <think>...</think> blocks produced by reasoning models."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def generate(self, prompt: str, schema=None, **kwargs):
        """Generate a response, optionally parsed into a Pydantic schema."""
        chat_kwargs = dict(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            think=False,
        )
        # When DeepEval passes a schema, force JSON output
        if schema is not None:
            chat_kwargs["format"] = "json"

        response = ollama_client.chat(**chat_kwargs)
        content = self._strip_think(response["message"]["content"])

        if schema is not None:
            try:
                data = json.loads(content)
                return schema(**data)
            except (json.JSONDecodeError, Exception) as e:
                # If parsing fails, try to extract JSON from the response
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    return schema(**data)
                raise e

        return content

    async def a_generate(self, prompt: str, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self) -> str:
        return self._model_name


# ---------------------------------------------------------------------------
# Public evaluation function
# ---------------------------------------------------------------------------

def run_deepeval_evaluation(
    data: list[dict],
    judge_model: str = "qwen3",
) -> pd.DataFrame:
    """
    Run DeepEval evaluation on collected pipeline results.

    Args:
        data: List of dicts with keys:
            - question (str)
            - answer (str)
            - contexts (list[str])
            - ground_truth (str)
        judge_model: Ollama model name for LLM-as-judge

    Returns:
        DataFrame with per-question metric scores.
    """
    model = OllamaJudge(model_name=judge_model)

    # Define metrics
    contextual_relevancy = ContextualRelevancyMetric(
        model=model,
        threshold=0.5,
    )
    faithfulness = FaithfulnessMetric(
        model=model,
        threshold=0.5,
    )
    hallucination = HallucinationMetric(
        model=model,
        threshold=0.5,
    )
    correctness = GEval(
        name="Correctness",
        criteria=(
            "Determine whether the actual output is factually correct "
            "based on the expected output. Score 1 if fully correct, "
            "0 if completely wrong, and partial scores for partial correctness."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model,
        threshold=0.5,
    )

    all_metrics = [contextual_relevancy, faithfulness, hallucination, correctness]

    # Build test cases
    test_cases = []
    for d in data:
        test_cases.append(
            LLMTestCase(
                input=d["question"],
                actual_output=d["answer"],
                expected_output=d["ground_truth"],
                retrieval_context=d["contexts"],
                context=d["contexts"],
            )
        )

    print(f"[DeepEval] Evaluating {len(test_cases)} test cases with judge='{judge_model}'...")

    # Evaluate each test case against all metrics
    rows = []
    for i, tc in enumerate(test_cases):
        row = {"question": tc.input}
        for metric in all_metrics:
            try:
                metric.measure(tc)
                row[f"deepeval_{metric.__class__.__name__}"] = metric.score
            except Exception as e:
                print(f"  [DeepEval] Metric {metric.__class__.__name__} failed on q{i+1}: {e}")
                row[f"deepeval_{metric.__class__.__name__}"] = None
        rows.append(row)

        if (i + 1) % 10 == 0:
            print(f"  [DeepEval] Progress: {i+1}/{len(test_cases)}")

    print(f"[DeepEval] Evaluation complete.")
    return pd.DataFrame(rows)

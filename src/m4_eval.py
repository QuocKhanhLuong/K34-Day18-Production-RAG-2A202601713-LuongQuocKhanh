from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _empty_eval() -> dict:
    return {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": [],
    }


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Run the four required RAGAS metrics with a non-crashing fallback."""
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("questions, answers, contexts and ground_truths must be parallel lists")
    if not questions:
        return _empty_eval()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        df = result.to_pandas()

        per_question: list[EvalResult] = []
        for _, row in df.iterrows():
            row_contexts = row.get("contexts", [])
            if not isinstance(row_contexts, list):
                row_contexts = list(row_contexts) if row_contexts is not None else []
            per_question.append(
                EvalResult(
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    contexts=[str(item) for item in row_contexts],
                    ground_truth=str(row.get("ground_truth", "")),
                    faithfulness=_safe_float(row.get("faithfulness", 0.0)),
                    answer_relevancy=_safe_float(row.get("answer_relevancy", 0.0)),
                    context_precision=_safe_float(row.get("context_precision", 0.0)),
                    context_recall=_safe_float(row.get("context_recall", 0.0)),
                )
            )

        aggregate = {}
        for metric in METRIC_NAMES:
            if metric in df.columns and len(df):
                aggregate[metric] = _safe_float(df[metric].mean())
            else:
                aggregate[metric] = 0.0
        aggregate["per_question"] = per_question
        return aggregate
    except Exception as exc:
        print(f"  ⚠️  RAGAS evaluation failed; returning safe fallback: {exc}")
        return _empty_eval()


def failure_analysis(
    eval_results: list[EvalResult], bottom_n: int = 10
) -> list[dict]:
    """Rank the weakest questions and map the worst metric to the Error Tree."""
    if bottom_n <= 0 or not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "Generation không bám đủ vào evidence trong context.",
            "Siết prompt 'chỉ dựa trên context', giảm temperature và kiểm tra citation/evidence trước khi trả lời.",
        ),
        "answer_relevancy": (
            "Answer chưa trả lời đúng trọng tâm câu hỏi dù có thể đã có context.",
            "Rút gọn prompt, nhắc model trả lời trực tiếp đúng intent và thêm regression test cho query này.",
        ),
        "context_precision": (
            "Retriever đưa quá nhiều chunk không liên quan vào top context.",
            "Kiểm tra BM25/dense ranks, tăng chất lượng RRF/reranking hoặc thêm metadata/version filter.",
        ),
        "context_recall": (
            "Context thiếu evidence cần thiết để khôi phục đầy đủ ground truth.",
            "Kiểm tra M1 boundary/parent-child, M2 retrieval và source/version metadata; bổ sung chunk hoặc retrieval candidate bị thiếu.",
        ),
    }

    ranked = []
    for item in eval_results:
        scores = {
            "faithfulness": _safe_float(item.faithfulness),
            "answer_relevancy": _safe_float(item.answer_relevancy),
            "context_precision": _safe_float(item.context_precision),
            "context_recall": _safe_float(item.context_recall),
        }
        average = sum(scores.values()) / len(scores)
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        ranked.append(
            {
                "question": item.question,
                "answer": item.answer,
                "ground_truth": item.ground_truth,
                "contexts": item.contexts,
                "average_score": round(average, 6),
                "worst_metric": worst_metric,
                "score": scores[worst_metric],
                "error_tree": (
                    "Answer đúng ground truth? → Context có evidence? → "
                    "Nếu thiếu: M1/M2/metadata-version; nếu đủ: prompt/generation."
                ),
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )

    ranked.sort(key=lambda row: row["average_score"])
    return ranked[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save aggregate metrics and failure-analysis evidence to JSON."""
    report = {
        "aggregate": {
            key: _safe_float(value)
            for key, value in results.items()
            if key != "per_question"
        },
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")

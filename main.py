"""
Lab 18: Production RAG Pipeline — Main Entry Point
===================================================
Chạy toàn bộ pipeline: naive baseline → production → so sánh → report.

Usage:
    python main.py
"""

import json
import os
import time


REPORT_FILES = [
    "ragas_report.json",
    "naive_baseline_report.json",
    "latency_breakdown.json",
]


def _move_reports() -> None:
    os.makedirs("reports", exist_ok=True)
    for filename in REPORT_FILES:
        if os.path.exists(filename):
            os.replace(filename, os.path.join("reports", filename))


def main():
    print("=" * 60)
    print("LAB 18: PRODUCTION RAG PIPELINE")
    print("=" * 60)
    start = time.time()

    os.makedirs("reports", exist_ok=True)

    print("\n📌 STEP 1: Running Basic RAG Baseline...")
    print("-" * 40)
    from naive_baseline import main as run_baseline

    run_baseline()

    print("\n📌 STEP 2: Running Production Pipeline...")
    print("-" * 40)
    from src.pipeline import build_pipeline, evaluate_pipeline

    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)

    # The codelab requires the submission reports under reports/. os.replace
    # also makes repeated runs idempotent on macOS/Linux/Windows.
    _move_reports()

    print("\n📌 STEP 3: Comparison")
    print("-" * 40)
    naive_path = "reports/naive_baseline_report.json"
    prod_path = "reports/ragas_report.json"

    if os.path.exists(naive_path) and os.path.exists(prod_path):
        with open(naive_path, encoding="utf-8") as handle:
            naive = json.load(handle)
        with open(prod_path, encoding="utf-8") as handle:
            prod = json.load(handle)

        print(f"\n{'Metric':<25} {'Basic':>8} {'Production':>12} {'Δ':>8}")
        print("-" * 55)
        for metric in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]:
            baseline_score = naive.get("aggregate", {}).get(metric, 0)
            production_score = prod.get("aggregate", {}).get(metric, 0)
            delta = production_score - baseline_score
            status = "✓" if production_score >= 0.75 else " "
            print(
                f"{status} {metric:<23} {baseline_score:>8.4f} "
                f"{production_score:>12.4f} {delta:>+8.4f}"
            )
    else:
        print("  ⚠️  Missing baseline/production report; inspect the run above.")

    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    print("\n📋 Submission checks:")
    print("  1. reports/ragas_report.json")
    print("  2. reports/naive_baseline_report.json")
    print("  3. reports/latency_breakdown.json (bonus evidence)")
    print("  4. analysis/failure_analysis.md + analysis/group_report.md")
    print("  5. analysis/reflections/reflection_LuongQuocKhanh.md")
    print("  6. python check_lab.py")


if __name__ == "__main__":
    main()

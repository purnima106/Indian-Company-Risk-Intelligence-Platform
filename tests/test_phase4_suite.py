from app.evaluation.phase4 import (
    benchmark_schema,
    build_phase4_report,
    compare_prompt_variants,
    compare_thresholds,
    compute_classification_metrics,
    compute_extraction_accuracy,
    compute_retrieval_metrics,
    evaluate_predictions,
)


def test_benchmark_schema_has_expected_columns():
    columns = benchmark_schema()
    assert "previous_risk_id" in columns
    assert "current_risk_id" in columns
    assert "match" in columns
    assert "classification" in columns
    assert "severity_change" in columns
    assert "scope_change" in columns


def test_phase4_report_includes_basic_metrics():
    predictions = [
        {
            "previous_risk_id": "TATA_FY2022-23_R001",
            "current_risk_id": "TATA_FY2023-24_R001",
            "match": True,
            "classification": "STABLE",
            "severity_change": "unchanged",
            "scope_change": False,
            "confidence": 85,
            "explanation": "Both risks are operationally equivalent.",
        }
    ]
    ground_truth = [
        {
            "previous_risk_id": "TATA_FY2022-23_R001",
            "current_risk_id": "TATA_FY2023-24_R001",
            "match": True,
            "classification": "STABLE",
            "severity_change": "unchanged",
            "scope_change": False,
        }
    ]

    report = build_phase4_report(predictions, ground_truth)

    assert report["basic_accuracy"]["accuracy"] == 1.0
    assert report["classification_metrics"]["macro_f1"] >= 0.0


def test_threshold_comparison_returns_values():
    candidate_rows = [
        {
            "previous_risk_id": "TATA_FY2022-23_R001",
            "current_risk_id": "TATA_FY2023-24_R001",
            "similarity_score": 0.91,
        }
    ]
    ground_truth = [
        {
            "previous_risk_id": "TATA_FY2022-23_R001",
            "current_risk_id": "TATA_FY2023-24_R001",
            "match": True,
            "classification": "STABLE",
            "severity_change": "unchanged",
            "scope_change": False,
        }
    ]

    comparison = compare_thresholds(candidate_rows, ground_truth)
    assert comparison["thresholds"]
    assert comparison["best_threshold"] is not None


def test_prompt_comparison_uses_variant_accuracy():
    ground_truth = [
        {
            "previous_risk_id": "TATA_FY2022-23_R001",
            "current_risk_id": "TATA_FY2023-24_R001",
            "match": True,
            "classification": "STABLE",
            "severity_change": "unchanged",
            "scope_change": False,
        }
    ]
    variants = {
        "baseline": [
            {
                "previous_risk_id": "TATA_FY2022-23_R001",
                "current_risk_id": "TATA_FY2023-24_R001",
                "match": True,
                "classification": "STABLE",
                "severity_change": "unchanged",
                "scope_change": False,
            }
        ],
        "strict": [
            {
                "previous_risk_id": "TATA_FY2022-23_R001",
                "current_risk_id": "TATA_FY2023-24_R001",
                "match": False,
                "classification": "NO_MATCH",
                "severity_change": "unclear",
                "scope_change": True,
            }
        ],
    }

    comparison = compare_prompt_variants(variants, ground_truth)
    assert comparison["best_variant"] == "baseline"


def test_retrieval_metrics_can_compute_mrr_and_recall():
    candidate_rows = [
        {
            "current_risk_id": "CUR_1",
            "previous_risk_id": "PRE_1",
            "similarity_score": 0.95,
        },
        {
            "current_risk_id": "CUR_1",
            "previous_risk_id": "PRE_2",
            "similarity_score": 0.80,
        },
    ]
    relevant_by_current = {"CUR_1": {"PRE_1"}}

    metrics = compute_retrieval_metrics(candidate_rows, relevant_by_current)
    assert metrics["mrr"] >= 0.0
    assert "recall@1" in metrics


def test_classification_metrics_build_confusion_matrix():
    predictions = [
        {
            "previous_risk_id": "A",
            "current_risk_id": "B",
            "classification": "STABLE",
        }
    ]
    ground_truth = [
        {
            "previous_risk_id": "A",
            "current_risk_id": "B",
            "classification": "STABLE",
        }
    ]

    metrics = compute_classification_metrics(predictions, ground_truth)
    assert metrics["confusion_matrix"]["STABLE"]["STABLE"] == 1

import pytest

from app.evaluation.phase4 import evaluate_predictions


def test_evaluate_predictions_returns_accuracy_and_counts():
    predictions = [
        {
            "previous_risk_id": "TATA_FY2022-23_R001",
            "current_risk_id": "TATA_FY2023-24_R001",
            "match": True,
            "classification": "STABLE",
            "severity_change": "unchanged",
            "scope_change": False,
            "confidence": 85,
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

    report = evaluate_predictions(predictions, ground_truth)

    assert report["total_pairs"] == 1
    assert report["matches_correct"] == 1
    assert report["accuracy"] == 1.0
    assert report["classification_accuracy"] == 1.0
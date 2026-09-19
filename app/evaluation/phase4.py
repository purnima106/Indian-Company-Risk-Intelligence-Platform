import csv
import json
from pathlib import Path


def load_ground_truth(csv_path: str | Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "1", "yes", "y"}:
            return True
        if cleaned in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def evaluate_predictions(predictions: list[dict], ground_truth: list[dict]) -> dict:
    truth_map = {
        (
            row.get("previous_risk_id"),
            row.get("current_risk_id"),
        ): row
        for row in ground_truth
    }

    total_pairs = len(predictions)
    matches_correct = 0
    classification_correct = 0
    severity_correct = 0
    scope_correct = 0

    for prediction in predictions:
        key = (
            prediction.get("previous_risk_id"),
            prediction.get("current_risk_id"),
        )
        truth = truth_map.get(key)
        if truth is None:
            continue

        pred_match = _as_bool(prediction.get("match"))
        truth_match = _as_bool(truth.get("match"))
        if pred_match == truth_match:
            matches_correct += 1

        if prediction.get("classification") == truth.get("classification"):
            classification_correct += 1

        if prediction.get("severity_change") == truth.get("severity_change"):
            severity_correct += 1

        if _as_bool(prediction.get("scope_change")) == _as_bool(truth.get("scope_change")):
            scope_correct += 1

    accuracy = (matches_correct / total_pairs) if total_pairs else 0.0
    classification_accuracy = (
        classification_correct / total_pairs if total_pairs else 0.0
    )
    severity_accuracy = severity_correct / total_pairs if total_pairs else 0.0
    scope_accuracy = scope_correct / total_pairs if total_pairs else 0.0

    return {
        "total_pairs": total_pairs,
        "matches_correct": matches_correct,
        "classification_correct": classification_correct,
        "severity_correct": severity_correct,
        "scope_correct": scope_correct,
        "accuracy": accuracy,
        "classification_accuracy": classification_accuracy,
        "severity_accuracy": severity_accuracy,
        "scope_accuracy": scope_accuracy,
    }


def save_evaluation_report(report: dict, output_path: str | Path):
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

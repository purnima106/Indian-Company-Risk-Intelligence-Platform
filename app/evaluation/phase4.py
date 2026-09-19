import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

BENCHMARK_COLUMNS = [
    "previous_risk_id",
    "current_risk_id",
    "previous_year",
    "current_year",
    "previous_title",
    "current_title",
    "match",
    "classification",
    "severity_change",
    "scope_change",
    "similarity_score",
    "similarity_rank",
    "retrieval_rank",
    "prompt_variant",
    "explanation",
    "confidence",
    "reviewed",
    "review_score",
]

CLASSIFICATION_LABELS = [
    "NO_MATCH",
    "STABLE",
    "MINOR_CHANGE",
    "MATERIALLY_CHANGED",
]

SEVERITY_VALUES = ["increased", "decreased", "unchanged", "unclear"]


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


def benchmark_schema() -> list[str]:
    return list(BENCHMARK_COLUMNS)


def write_benchmark_template(
    csv_path: str | Path,
    rows: Iterable[dict] | None = None,
) -> str:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_COLUMNS)
        writer.writeheader()
        for row in rows or []:
            writer.writerow(row)

    return str(path)


def load_ground_truth(csv_path: str | Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def save_phase2_candidates(candidates: list[dict], output_path: str | Path) -> list[dict]:
    rows: list[dict] = []

    for current_match in candidates:
        current_risk = current_match.get("current_risk", {})
        current_risk_id = current_risk.get("risk_id")
        current_year = current_risk.get("source_year")

        for rank, candidate in enumerate(current_match.get("candidates", []), start=1):
            metadata = candidate.get("metadata", {})
            rows.append(
                {
                    "current_risk_id": current_risk_id,
                    "previous_risk_id": metadata.get("risk_id"),
                    "current_year": current_year,
                    "previous_year": metadata.get("source_year"),
                    "current_title": current_risk.get("title"),
                    "previous_title": metadata.get("title"),
                    "similarity_score": float(candidate.get("similarity", 0.0)),
                    "similarity_rank": rank,
                    "retrieval_rank": rank,
                }
            )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)

    return rows


def run_with_retry(
    func: Callable[[], Any],
    *,
    retries: int = 3,
    delay: float = 0.25,
    exceptions: tuple[type[Exception], ...] = (Exception,),
):
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return func()
        except exceptions as error:  # type: ignore[misc]
            last_error = error
            if attempt >= retries:
                raise
            time.sleep(delay * attempt)

    if last_error is not None:
        raise last_error

    raise RuntimeError("Retry loop ended without a result.")


class BatchPhase3Runner:
    def __init__(
        self,
        comparator: Callable[..., dict],
        cache_path: str | Path = "data/outputs/phase3_cache.json",
        retries: int = 3,
    ):
        self.comparator = comparator
        self.cache_path = Path(cache_path)
        self.retries = retries

    def _load_cache(self) -> dict:
        if not self.cache_path.exists():
            return {}

        try:
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            return {}

    def _save_cache(self, cache: dict) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, ensure_ascii=False)

    def run_pair(
        self,
        previous_risk: dict,
        current_risk: dict,
        similarity_score: float,
        *,
        prompt_variant: str = "baseline",
    ) -> dict:
        cache = self._load_cache()
        key = json.dumps(
            {
                "previous_risk_id": previous_risk.get("risk_id"),
                "current_risk_id": current_risk.get("risk_id"),
                "similarity_score": float(similarity_score),
                "prompt_variant": prompt_variant,
            },
            sort_keys=True,
        )

        if key in cache:
            return cache[key]

        def _call() -> dict:
            result = self.comparator(
                previous_risk,
                current_risk,
                float(similarity_score),
                prompt_variant=prompt_variant,
            )
            if hasattr(result, "model_dump"):
                return result.model_dump()
            return dict(result)

        result = run_with_retry(_call, retries=self.retries)
        result.setdefault("previous_risk_id", previous_risk.get("risk_id"))
        result.setdefault("current_risk_id", current_risk.get("risk_id"))
        result.setdefault("confidence", 0)
        result.setdefault("match", False)

        cache[key] = result
        self._save_cache(cache)
        return result


def _pair_key(prediction: dict) -> tuple[str | None, str | None]:
    return (
        prediction.get("previous_risk_id"),
        prediction.get("current_risk_id"),
    )


def evaluate_predictions(predictions: list[dict], ground_truth: list[dict]) -> dict:
    truth_map = {_pair_key(row): row for row in ground_truth}

    total_pairs = len(predictions)
    matches_correct = 0
    classification_correct = 0
    severity_correct = 0
    scope_correct = 0

    for prediction in predictions:
        truth = truth_map.get(_pair_key(prediction))
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


def compute_extraction_accuracy(
    extracted: list[dict],
    ground_truth: list[dict],
    id_field: str = "risk_id",
) -> dict:
    extracted_ids = {
        str(item.get(id_field) or item.get("title") or "")
        for item in extracted
        if item.get(id_field) or item.get("title")
    }
    truth_ids = {
        str(item.get(id_field) or item.get("title") or "")
        for item in ground_truth
        if item.get(id_field) or item.get("title")
    }

    common = extracted_ids & truth_ids
    precision = len(common) / len(extracted_ids) if extracted_ids else 0.0
    recall = len(common) / len(truth_ids) if truth_ids else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_count": len(common),
        "expected_count": len(truth_ids),
        "extracted_count": len(extracted_ids),
    }


def compute_retrieval_metrics(
    candidate_rows: list[dict],
    relevant_by_current: dict[str, set[str]],
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict:
    grouped: dict[str, list[tuple[float, str]]] = defaultdict(list)

    for row in candidate_rows:
        current_id = row.get("current_risk_id")
        previous_id = row.get("previous_risk_id")
        similarity = float(row.get("similarity_score", 0.0) or 0.0)
        if current_id and previous_id:
            grouped[current_id].append((similarity, previous_id))

    per_current: dict[str, dict[str, float]] = {}
    reciprocal_ranks: list[float] = []
    recall_by_k: dict[int, list[float]] = {k: [] for k in k_values}
    precision_by_k: dict[int, list[float]] = {k: [] for k in k_values}
    ndcg_by_k: dict[int, list[float]] = {k: [] for k in k_values}

    all_current_ids = set(grouped) | set(relevant_by_current)

    for current_id in sorted(all_current_ids):
        ranked = [
            previous_id for _, previous_id in sorted(grouped.get(current_id, []), key=lambda item: item[0], reverse=True)
        ]
        relevant = relevant_by_current.get(current_id, set())

        if relevant:
            found_rank = None
            for index, previous_id in enumerate(ranked, start=1):
                if previous_id in relevant:
                    found_rank = index
                    break
            reciprocal_ranks.append(1 / found_rank if found_rank else 0.0)

        for k in k_values:
            top_k = ranked[:k]
            hits = len(set(top_k) & relevant)
            if relevant:
                recall_by_k[k].append(hits / len(relevant))
            else:
                recall_by_k[k].append(0.0)

            precision_by_k[k].append(hits / max(k, 1))

            rels = [1 if previous_id in relevant else 0 for previous_id in ranked[:k]]
            dcg = sum((2 ** rel - 1) / math.log2(index + 2) for index, rel in enumerate(rels) if rel)
            ideal = sum((2 ** 1 - 1) / math.log2(index + 2) for index in range(min(len(relevant), k)))
            ndcg_by_k[k].append(dcg / ideal if ideal else 0.0)

    summary = {
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "per_current_risk": per_current,
    }
    for k in k_values:
        summary[f"recall@{k}"] = sum(recall_by_k[k]) / len(recall_by_k[k]) if recall_by_k[k] else 0.0
        summary[f"precision@{k}"] = sum(precision_by_k[k]) / len(precision_by_k[k]) if precision_by_k[k] else 0.0
        summary[f"ndcg@{k}"] = sum(ndcg_by_k[k]) / len(ndcg_by_k[k]) if ndcg_by_k[k] else 0.0

    return summary


def compute_classification_metrics(
    predictions: list[dict],
    ground_truth: list[dict],
    labels: list[str] | None = None,
) -> dict:
    truth_map = {_pair_key(row): row for row in ground_truth}

    all_labels = list(labels or CLASSIFICATION_LABELS)
    matrix: dict[str, dict[str, int]] = {
        label: {truth_label: 0 for truth_label in all_labels}
        for label in all_labels
    }

    seen = 0
    for prediction in predictions:
        truth = truth_map.get(_pair_key(prediction))
        if truth is None:
            continue
        seen += 1

        pred_label = str(prediction.get("classification") or "NO_MATCH")
        truth_label = str(truth.get("classification") or "NO_MATCH")

        if pred_label not in matrix:
            matrix[pred_label] = {truth_label: 0 for truth_label in all_labels}
        if truth_label not in matrix[pred_label]:
            matrix[pred_label][truth_label] = 0
        matrix[pred_label][truth_label] += 1

    per_label: dict[str, dict[str, float]] = {}
    for label in all_labels:
        tp = matrix.get(label, {}).get(label, 0)
        fp = sum(
            matrix.get(pred_label, {}).get(label, 0)
            for pred_label in all_labels
            if pred_label != label
        )
        fn = sum(
            matrix.get(label, {}).get(truth_label, 0)
            for truth_label in all_labels
            if truth_label != label
        )

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix.get(label, {}).values()),
        }

    macro_f1 = sum(item["f1"] for item in per_label.values()) / len(per_label) if per_label else 0.0
    return {
        "labels": all_labels,
        "confusion_matrix": matrix,
        "per_label": per_label,
        "macro_f1": macro_f1,
        "examples_considered": seen,
    }


def compute_explanation_metrics(predictions: list[dict]) -> dict:
    if not predictions:
        return {
            "explanation_coverage": 0.0,
            "avg_confidence": 0.0,
            "avg_explanation_length": 0.0,
            "review_mae": 0.0,
            "reviewed_pairs": 0,
        }

    explanations = [str(item.get("explanation", "") or "") for item in predictions]
    explanation_coverage = sum(1 for text in explanations if len(text.strip()) > 20) / len(explanations)
    avg_confidence = sum(float(item.get("confidence", 0) or 0) for item in predictions) / len(predictions)
    avg_explanation_length = sum(len(text) for text in explanations) / len(explanations)

    review_scores = [
        float(item.get("review_score"))
        for item in predictions
        if item.get("review_score") is not None
    ]
    review_mae = 0.0
    if review_scores:
        review_mae = sum(abs(score - 0.5) for score in review_scores) / len(review_scores)

    return {
        "explanation_coverage": explanation_coverage,
        "avg_confidence": avg_confidence,
        "avg_explanation_length": avg_explanation_length,
        "review_mae": review_mae,
        "reviewed_pairs": len(review_scores),
    }


def compare_thresholds(
    candidate_rows: list[dict],
    ground_truth: list[dict],
    thresholds: tuple[float, ...] = (0.75, 0.80, 0.82, 0.85, 0.90),
) -> dict:
    truth_map = {_pair_key(row): row for row in ground_truth}
    results: list[dict] = []

    for threshold in thresholds:
        tp = tn = fp = fn = 0
        for row in candidate_rows:
            truth = truth_map.get((_pair_key(row)[0], _pair_key(row)[1]))
            if truth is None:
                continue

            predicted_match = float(row.get("similarity_score", 0.0) or 0.0) >= threshold
            actual_match = _as_bool(truth.get("match"))

            if predicted_match and actual_match:
                tp += 1
            elif predicted_match and not actual_match:
                fp += 1
            elif not predicted_match and actual_match:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            }
        )

    best = max(results, key=lambda item: (item["f1"], item["accuracy"])) if results else {}
    return {"thresholds": results, "best_threshold": best.get("threshold")}


def compare_prompt_variants(
    results_by_variant: dict[str, list[dict]],
    ground_truth: list[dict],
) -> dict:
    comparison: dict[str, float] = {}
    for variant_name, predictions in results_by_variant.items():
        comparison[variant_name] = evaluate_predictions(predictions, ground_truth)["accuracy"]
    return {
        "by_variant": comparison,
        "best_variant": max(comparison.items(), key=lambda item: item[1])[0] if comparison else None,
    }


def build_phase4_report(
    predictions: list[dict],
    ground_truth: list[dict],
    *,
    candidate_rows: list[dict] | None = None,
    relevant_by_current: dict[str, set[str]] | None = None,
    prompt_variants: dict[str, list[dict]] | None = None,
    threshold_values: tuple[float, ...] = (0.75, 0.80, 0.82, 0.85, 0.90),
) -> dict:
    report: dict[str, Any] = {
        "basic_accuracy": evaluate_predictions(predictions, ground_truth),
        "classification_metrics": compute_classification_metrics(predictions, ground_truth),
        "explanation_metrics": compute_explanation_metrics(predictions),
    }

    if candidate_rows is not None and relevant_by_current is not None:
        report["retrieval_metrics"] = compute_retrieval_metrics(candidate_rows, relevant_by_current)
    if candidate_rows is not None:
        report["threshold_comparison"] = compare_thresholds(candidate_rows, ground_truth, threshold_values)
    if prompt_variants is not None:
        report["prompt_comparison"] = compare_prompt_variants(prompt_variants, ground_truth)

    return report


def save_evaluation_report(report: dict, output_path: str | Path):
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

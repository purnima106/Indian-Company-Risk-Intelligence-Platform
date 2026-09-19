import json
from pathlib import Path

from app.evaluation.phase4 import (
    build_phase4_report,
    load_ground_truth,
    save_evaluation_report,
)


def main():
    project_root = Path(__file__).resolve().parents[1]

    predictions_path = project_root / "data" / "outputs" / "phase3_results.json"
    ground_truth_path = project_root / "data" / "processed" / "evaluation_ground_truth.csv"
    output_path = project_root / "data" / "outputs" / "evaluation_results.json"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Missing ground truth file: {ground_truth_path}")

    with open(predictions_path, "r", encoding="utf-8") as handle:
        predictions = json.load(handle)

    ground_truth = load_ground_truth(ground_truth_path)

    report = build_phase4_report(
        predictions,
        ground_truth,
        candidate_rows=[],
        relevant_by_current={},
        prompt_variants={"baseline": predictions},
    )

    save_evaluation_report(report, output_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

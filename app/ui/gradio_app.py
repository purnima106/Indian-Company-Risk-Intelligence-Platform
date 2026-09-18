import json
from pathlib import Path

import gradio as gr


DATA_DIR = Path(
    "data/processed/tata_motors"
)


YEARS = [
    "FY2022-23",
    "FY2023-24",
    "FY2024-25",
    "FY2025-26",
]


def load_risk_data(year):
    file_path = DATA_DIR / f"risk_factors_{year}.json"
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_risks(year):
    data = load_risk_data(year)
    if data is None:

        return (
            f"### No extracted data for {year}\n\n"
            "Run the extraction pipeline first."
        )

    risks = data["risks"]

    if not risks:

        return "No risks extracted."

    output = [
        f"# Tata Motors — {year}",
        "",
        f"**Risks extracted:** {len(risks)}",
        "",
    ]

    for index, risk in enumerate(risks, start=1):

        output.extend(
            [
                "---",
                "",
                f"## {index}. {risk['title']}",
                "",
                f"**Category:** {risk['category']}",
                "",
                risk["text"],
                "",
                (
                    f"**Source:** "
                    f"{risk['source_document']} "
                    f"| Page {risk['page']}"
                ),
                "",
            ]
        )

    return "\n".join(output)


def match_risks(previous_year, current_year, top_k):
    if previous_year == current_year:
        return "### Choose two different financial years."

    previous_data = load_risk_data(previous_year)
    current_data = load_risk_data(current_year)
    missing_years = [
        year
        for year, data in (
            (previous_year, previous_data),
            (current_year, current_data),
        )
        if data is None
    ]
    if missing_years:
        return (
            "### Matching data is not available yet\n\n"
            f"Run the extraction pipeline for: {', '.join(missing_years)}."
        )

    try:
        from app.risk_analysis.matcher import RiskMatcher
        from app.risk_analysis.risk_extractor import RiskRecord

        previous_risks = [RiskRecord(**risk) for risk in previous_data["risks"]]
        current_risks = [RiskRecord(**risk) for risk in current_data["risks"]]
        matches = RiskMatcher().match(previous_risks, current_risks, int(top_k))
    except Exception as error:
        return (
            "### Semantic matching could not start\n\n"
            f"`{type(error).__name__}: {error}`\n\n"
            "Install or repair the embedding dependencies, then restart Gradio."
        )

    output = [
        f"# Semantic matches: {previous_year} to {current_year}",
        "",
        "Phase 2 shows candidate matches only; it does not determine material change.",
        "",
    ]
    for match in matches:
        current = match["current_risk"]
        output.extend(["---", "", f"## Current: {current['title']}", ""])
        for candidate in match["candidates"]:
            metadata = candidate["metadata"]
            score = candidate["similarity"]
            status = "strong candidate" if score >= 0.82 else "review required"
            output.extend(
                [
                    f"**Previous:** {metadata['title']}",
                    f"**Similarity:** `{score:.3f}` ({status})",
                    f"**Risk ID:** `{metadata['risk_id']}`",
                    "",
                ]
            )
    return "\n".join(output)


def risk_choices(year):
    data = load_risk_data(year)
    if data is None:
        return []
    return [
        (f"{risk['risk_id']} - {risk['title']}", risk["risk_id"])
        for risk in data["risks"]
    ]


def update_risk_choices(year):
    choices = risk_choices(year)
    value = choices[0][1] if choices else None
    return gr.Dropdown(choices=choices, value=value)


def classify_risk_pair(
    previous_year,
    previous_risk_id,
    current_year,
    current_risk_id,
    similarity,
):
    previous_data = load_risk_data(previous_year)
    current_data = load_risk_data(current_year)
    if previous_data is None or current_data is None:
        return "### Run extraction for both selected years first."

    previous_risk = next(
        (risk for risk in previous_data["risks"] if risk["risk_id"] == previous_risk_id),
        None,
    )
    current_risk = next(
        (risk for risk in current_data["risks"] if risk["risk_id"] == current_risk_id),
        None,
    )
    if previous_risk is None or current_risk is None:
        return "### Select valid risks from both years."

    try:
        from app.llm.ollama import OllamaClient
        from app.risk_analysis.risk_classifier import RiskClassifier

        result = RiskClassifier(OllamaClient()).compare_pair(
            previous_risk=previous_risk,
            current_risk=current_risk,
            similarity=float(similarity),
        )
    except Exception as error:
        return (
            "### Phase 3 classification failed\n\n"
            f"`{type(error).__name__}: {error}`\n\n"
            "Confirm that Ollama is running and the selected model is available."
        )

    return "\n".join(
        [
            "# Phase 3 classification",
            "",
            f"**Match:** `{result.match}`",
            f"**Classification:** `{result.classification}`",
            f"**Severity change:** `{result.severity_change}`",
            f"**Scope change:** `{result.scope_change}`",
            f"**Confidence:** `{result.confidence}%`",
            "",
            f"**Explanation:** {result.explanation}",
            "",
            f"**Previous source:** `{result.previous_risk_id}`, page `{result.previous_page}`",
            f"**Current source:** `{result.current_risk_id}`, page `{result.current_page}`",
        ]
    )


with gr.Blocks(
    title="Corporate Risk Intelligence"
) as demo:

    gr.Markdown(
        """
        # Corporate Risk Intelligence

        ### Tata Motors — Risk Intelligence

        Browse extracted risks and inspect cross-year semantic candidates.
        """
    )

    with gr.Tab("Phase 1: Extracted Risks"):
        year_dropdown = gr.Dropdown(
            choices=YEARS,
            value=YEARS[0],
            label="Financial Year",
        )
        risk_output = gr.Markdown()
        year_dropdown.change(load_risks, year_dropdown, risk_output)
        demo.load(load_risks, year_dropdown, risk_output)

    with gr.Tab("Phase 2: Semantic Matching"):
        gr.Markdown(
            "Compare risks across two years using embedding similarity. "
            "Results are candidates for human review."
        )
        with gr.Row():
            previous_year = gr.Dropdown(
                choices=YEARS,
                value=YEARS[0],
                label="Previous year",
            )
            current_year = gr.Dropdown(
                choices=YEARS,
                value=YEARS[1],
                label="Current year",
            )
            top_k = gr.Slider(1, 5, value=3, step=1, label="Candidates per risk")
        match_button = gr.Button("Find semantic matches")
        match_output = gr.Markdown()
        match_button.click(
            match_risks,
            inputs=[previous_year, current_year, top_k],
            outputs=match_output,
        )

    with gr.Tab("Phase 3: LLM Classification"):
        gr.Markdown(
            "Select a Phase 2 candidate pair and ask Ollama to validate the match "
            "and classify the change."
        )
        with gr.Row():
            classifier_previous_year = gr.Dropdown(
                choices=YEARS,
                value=YEARS[0],
                label="Previous year",
            )
            classifier_current_year = gr.Dropdown(
                choices=YEARS,
                value=YEARS[1],
                label="Current year",
            )
        with gr.Row():
            classifier_previous_risk = gr.Dropdown(
                choices=risk_choices(YEARS[0]),
                value=(risk_choices(YEARS[0]) or [(None, None)])[0][1],
                label="Previous risk",
            )
            classifier_current_risk = gr.Dropdown(
                choices=risk_choices(YEARS[1]),
                value=(risk_choices(YEARS[1]) or [(None, None)])[0][1],
                label="Current risk",
            )
        similarity_input = gr.Number(
            value=0.0,
            label="Phase 2 similarity score",
            info="Paste the candidate similarity score from Phase 2.",
        )
        classify_button = gr.Button("Classify with Ollama")
        classification_output = gr.Markdown()
        classifier_previous_year.change(
            update_risk_choices,
            inputs=classifier_previous_year,
            outputs=classifier_previous_risk,
        )
        classifier_current_year.change(
            update_risk_choices,
            inputs=classifier_current_year,
            outputs=classifier_current_risk,
        )
        classify_button.click(
            classify_risk_pair,
            inputs=[
                classifier_previous_year,
                classifier_previous_risk,
                classifier_current_year,
                classifier_current_risk,
                similarity_input,
            ],
            outputs=classification_output,
        )


if __name__ == "__main__":
    demo.launch()
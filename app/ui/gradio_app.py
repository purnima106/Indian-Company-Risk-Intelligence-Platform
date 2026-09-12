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


def load_risks(year):

    file_path = DATA_DIR / f"{year}.json"

    if not file_path.exists():

        return (
            f"### No extracted data for {year}\n\n"
            "Run the extraction pipeline first."
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

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


with gr.Blocks(
    title="Corporate Risk Intelligence"
) as demo:

    gr.Markdown(
        """
        # Corporate Risk Intelligence

        ### Tata Motors — Phase 1 Risk Extraction

        Browse risks extracted from annual reports.
        """
    )

    year_dropdown = gr.Dropdown(
        choices=YEARS,
        value=YEARS[0],
        label="Financial Year",
    )

    risk_output = gr.Markdown()

    year_dropdown.change(
        fn=load_risks,
        inputs=year_dropdown,
        outputs=risk_output,
    )

    demo.load(
        fn=load_risks,
        inputs=year_dropdown,
        outputs=risk_output,
    )


if __name__ == "__main__":
    demo.launch()
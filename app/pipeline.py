import json
from pathlib import Path

import yaml

from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.section_extractor import extract_page_range
from app.risk_analysis.risk_extractor import extract_risk_records

CONFIG_PATH = Path("configs/config.yaml")

def load_config():

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def process_Report(year: str):
    config = load_config()
    company = config["company"]["name"]
    report = config["reports"][year]

    pdf_path = report["file"]
    print(f"Processing report for {company} ({year}) from {pdf_path}")

    pages = parse_pdf(pdf_path)

    print(f" Total Extracted {len(pages)} pages from the PDF.")

    start_page = report['risk_factor_start']
    end_page = report['risk_factor_end']

    if start_page is None or end_page is None:
        raise ValueError(f"Page range for risk factors not specified in the config for {year}.")

    risk_pages = extract_page_range(
        pages,
        start_page,
        end_page
    )
    print(
        f" Extracted {len(risk_pages)} pages for risk factors (from page {start_page} to {end_page})."
    )

    risks = extract_risk_records(
        pages=risk_pages,
        company=company,
        source_year=year,
        source_document=Path(pdf_path).name,
    )

    output_dir = Path("data/processed/tata_motors")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"risk_factors_{year}.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            {
                "company": company,
                "source_year": year,
                "risk_count": len(risks),
                "risks": [risk.model_dump() for risk in risks],
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Risks extracted: {len(risks)}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":

    process_Report("FY2023-24")
import re

from pydantic import BaseModel


class RiskRecord(BaseModel):
    risk_id: str
    company: str
    source_year: str
    category: str
    title: str
    text: str
    page: int
    source_document: str


_PAGE_FURNITURE = {
    "Integrated Report / 2022-23",
    "142-304",
    "Statutory Reports",
    "305-551",
    "Financial Statements",
    "1-141",
    "Integrated Report",
    "Risks Factor",
}


def _clean_block(block: str) -> str:
    lines = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line in _PAGE_FURNITURE or line.isdigit():
            continue
        if re.fullmatch(r"\d+[-/]\d+", line):
            continue
        lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _is_risk_heading(block: str) -> bool:
    lines = [
        line.strip()
        for line in block.splitlines()
        if line.strip() and line.strip() not in _PAGE_FURNITURE
    ]
    cleaned = _clean_block(block)
    word_count = len(cleaned.split())

    if not 8 <= word_count <= 35 or not 2 <= len(lines) <= 5:
        return False
    if not cleaned:
        return False

    return True


def _extract_sections(pages: list[dict]) -> list[tuple[str, str, int]]:
    if any(page.get("lines") for page in pages):
        return _extract_structured_sections(pages)

    sections = []
    current_title = None
    current_blocks = []
    current_page = None

    for page in pages:
        page_text = page.get("text", "")
        if not page_text:
            continue

        blocks = []
        for raw_block in re.split(r"\n\s*\n", page_text):
            cleaned_block = _clean_block(raw_block)
            if cleaned_block:
                blocks.append((raw_block, cleaned_block))

        for raw_block, block in blocks:
            if block in _PAGE_FURNITURE or block.isdigit():
                continue

            if _is_risk_heading(raw_block):
                if current_title is not None:
                    sections.append(
                        (current_title, " ".join(current_blocks), current_page)
                    )
                current_title = block
                current_blocks = []
                current_page = page["page"]
            elif current_title is not None:
                current_blocks.append(block)

    if current_title is not None:
        sections.append((current_title, " ".join(current_blocks), current_page))

    return sections


def _extract_structured_sections(pages: list[dict]) -> list[tuple[str, str, int]]:
    sections = []
    current_title = None
    current_lines = []
    current_page = None

    for page in pages:
        lines = page.get("lines", [])
        index = 0
        while index < len(lines):
            line = lines[index]
            line_text = _clean_block(line.get("text", ""))
            if not line_text:
                index += 1
                continue

            if line.get("is_heading", False):
                heading_lines = [line_text]
                heading_size = line.get("size", 0)
                next_index = index + 1
                while next_index < len(lines) and lines[next_index].get(
                    "is_heading", False
                ):
                    next_size = lines[next_index].get("size", 0)
                    if abs(next_size - heading_size) > 0.5:
                        break
                    heading_text = _clean_block(lines[next_index].get("text", ""))
                    if heading_text:
                        heading_lines.append(heading_text)
                    next_index += 1
                heading = " ".join(heading_lines)
            else:
                next_index = index + 1
                heading = ""

            if heading and _is_risk_heading_line(heading):
                if current_title is not None:
                    sections.append(
                        (current_title, " ".join(current_lines), current_page)
                    )
                current_title = heading
                current_lines = []
                current_page = page["page"]
            elif current_title is not None:
                current_lines.append(line_text)

            index = next_index

    if current_title is not None:
        sections.append((current_title, " ".join(current_lines), current_page))

    return sections


def _is_risk_heading_line(line: str) -> bool:
    if line in _PAGE_FURNITURE or line.isdigit():
        return False
    if "Risks Associated with" in line:
        return False
    return 8 <= len(line.split()) <= 35


def extract_risk_records(
    pages: list[dict],
    company: str,
    source_year: str,
    source_document: str,
) -> list[RiskRecord]:
    risks = []
    for risk_counter, (title, body, page) in enumerate(
        _extract_sections(pages), start=1
    ):
        risks.append(
            RiskRecord(
                risk_id=f"TATA_{source_year}_R{risk_counter:03d}",
                company=company,
                source_year=source_year,
                category="Unclassified",
                title=title,
                text=" ".join(part for part in (title, body) if part),
                page=page,
                source_document=source_document,
            )
        )

    return risks


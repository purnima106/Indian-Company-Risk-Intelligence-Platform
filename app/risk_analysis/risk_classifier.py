import json
from typing import Literal

from pydantic import BaseModel, Field

from app.llm.base import LLMClient


class RiskComparison(BaseModel):
    current_risk_id: str
    previous_risk_id: str

    match: bool

    classification: Literal[
        "NO_MATCH",
        "STABLE",
        "MINOR_CHANGE",
        "MATERIALLY_CHANGED",
    ]

    severity_change: Literal[
        "increased",
        "decreased",
        "unchanged",
        "unclear",
    ]

    scope_change: bool
    explanation: str
    confidence: int = Field(ge=0, le=100)

    current_page: int
    previous_page: int


def build_comparison_prompt(
    previous_risk: dict,
    current_risk: dict,
    similarity: float,
) -> str:
    return f"""
You are comparing two annual-report risk disclosures.

Determine whether they describe the same underlying business risk.

A high embedding similarity is only a candidate signal.
It is not proof that the risks are equivalent.

Use only the supplied evidence.
Do not invent facts.

Return only valid JSON with these fields:

current_risk_id
previous_risk_id
match
classification
severity_change
scope_change
explanation
confidence
current_page
previous_page

Allowed classification values:

NO_MATCH
STABLE
MINOR_CHANGE
MATERIALLY_CHANGED

Allowed severity_change values:

increased
decreased
unchanged
unclear

confidence must be an integer from 0 to 100.

Previous-year risk:
ID: {previous_risk["risk_id"]}
Page: {previous_risk["page"]}
Title: {previous_risk["title"]}
Text: {previous_risk["text"]}

Current-year risk:
ID: {current_risk["risk_id"]}
Page: {current_risk["page"]}
Title: {current_risk["title"]}
Text: {current_risk["text"]}

Embedding similarity:
{similarity:.4f}
"""


class RiskClassifier:
    def __init__(self, client: LLMClient):
        self.client = client

    def compare_pair(
        self,
        previous_risk: dict,
        current_risk: dict,
        similarity: float,
    ) -> RiskComparison:
        prompt = build_comparison_prompt(
            previous_risk=previous_risk,
            current_risk=current_risk,
            similarity=similarity,
        )

        raw_response = self.client.generate(prompt)

        try:
            response = json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise ValueError(
                "LLM response was not valid JSON."
            ) from error

        # Provenance must come from application data, not the LLM.
        response["current_risk_id"] = current_risk["risk_id"]
        response["previous_risk_id"] = previous_risk["risk_id"]
        response["current_page"] = current_risk["page"]
        response["previous_page"] = previous_risk["page"]

        return RiskComparison.model_validate(response)
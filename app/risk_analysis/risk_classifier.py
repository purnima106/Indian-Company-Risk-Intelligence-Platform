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

Return JSON in exactly this shape:

{{
  "match": true,
  "classification": "STABLE",
  "severity_change": "unchanged",
  "scope_change": false,
  "explanation": "Both disclosures describe the same underlying exposure.",
  "confidence": 85
}}

Important:
- match must be true or false, never a similarity score.
- scope_change must be true or false, never "expanded", "narrowed", or "unchanged".
- Do not put the embedding similarity inside match.

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

def normalize_llm_response(response: dict, similarity: float) -> dict:
    """
    Normalize common Ollama variations before Pydantic validation.
    """

    match_value = response.get("match")

    if isinstance(match_value, bool):
        response["match"] = match_value

    elif isinstance(match_value, (int, float)):
        # Some models place the similarity score in the match field.
        if 0 <= float(match_value) <= 1:
            response["match"] = float(match_value) >= 0.82
        else:
            raise ValueError(
                f"Invalid numeric match value: {match_value}"
            )

    elif isinstance(match_value, str):
        normalized_match = match_value.strip().lower()

        if normalized_match in {"true", "yes", "match"}:
            response["match"] = True
        elif normalized_match in {"false", "no", "no_match"}:
            response["match"] = False
        else:
            raise ValueError(
                f"Invalid match value returned by LLM: {match_value}"
            )

    else:
        response["match"] = similarity >= 0.82

    scope_value = response.get("scope_change")

    if isinstance(scope_value, bool):
        response["scope_change"] = scope_value

    elif isinstance(scope_value, str):
        normalized_scope = scope_value.strip().lower()

        if normalized_scope in {
            "true",
            "yes",
            "expanded",
            "broadened",
            "broader",
            "increased",
            "changed",
        }:
            response["scope_change"] = True

        elif normalized_scope in {
            "false",
            "no",
            "unchanged",
            "same",
            "none",
            "not changed",
        }:
            response["scope_change"] = False

        else:
            raise ValueError(
                f"Invalid scope_change value returned by LLM: {scope_value}"
            )

    else:
        response["scope_change"] = False

    return response


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

        response = normalize_llm_response(
            response=response,
            similarity=similarity,
        )

        response["current_risk_id"] = current_risk["risk_id"]
        response["previous_risk_id"] = previous_risk["risk_id"]
        response["current_page"] = current_risk["page"]
        response["previous_page"] = previous_risk["page"]

        return RiskComparison.model_validate(response)
from typing import Dict, List


# A small, transparent baseline keyword policy.
# This is intentionally simple and explainable.
RISK_KEYWORDS = {
    "high": [
        "harm",
        "attack",
        "weapon",
        "explosive",
        "poison",
        "bypass",
        "exploit",
        "illegal",
    ],
    "medium": [
        "password",
        "secret",
        "credential",
        "hack",
        "override",
        "unsafe",
    ],
    "low": [
        "private",
        "confidential",
        "restricted",
    ],
}


def normalize_text(text: str) -> str:
    """
    Normalize text for screening.
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().lower().split())


def score_text_risk(text: str) -> Dict:
    """
    Score extracted text based on keyword matches.

    Returns:
        {
            "normalized_text": ...,
            "matched_keywords": [...],
            "risk_score": int,
            "risk_level": str,
            "decision": str,
        }
    """
    normalized = normalize_text(text)

    matched_keywords: List[str] = []
    risk_score = 0

    for keyword in RISK_KEYWORDS["high"]:
        if keyword in normalized:
            matched_keywords.append(keyword)
            risk_score += 3

    for keyword in RISK_KEYWORDS["medium"]:
        if keyword in normalized:
            matched_keywords.append(keyword)
            risk_score += 2

    for keyword in RISK_KEYWORDS["low"]:
        if keyword in normalized:
            matched_keywords.append(keyword)
            risk_score += 1

    # Simple decision thresholds
    if risk_score >= 4:
        risk_level = "high"
        decision = "block"
    elif risk_score >= 2:
        risk_level = "medium"
        decision = "review"
    else:
        risk_level = "low"
        decision = "allow"

    return {
        "normalized_text": normalized,
        "matched_keywords": sorted(list(set(matched_keywords))),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "decision": decision,
    }
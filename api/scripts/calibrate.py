"""Run eval/queries.yaml against the indexed content: top-3 accuracy and confidence bands.

    uv run python -m scripts.calibrate

Prints each query's z-score and gap, then the thresholds that best separate
answerable queries (not "far") from unanswerable ones ("far"). Copy them to .env.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from app import db, search
from app.config import get_settings
from app.embeddings import get_encoder

QUERIES = Path(__file__).resolve().parents[1] / "eval" / "queries.yaml"


@dataclass
class Outcome:
    query: str
    answerable: bool
    found: bool
    z: float
    gap: float


def evaluate() -> list[Outcome]:
    settings = get_settings()
    cases = yaml.safe_load(QUERIES.read_text())
    vectors = get_encoder().encode_queries([case["query"] for case in cases])
    outcomes = []
    with db.connect() as conn:
        for case, vector in zip(cases, vectors):
            hits = search.hybrid_search(conn, case["query"], vector, None, settings)[:3]
            rows = db.fetch_chunks(conn, [hit.chunk_id for hit in hits], vector)
            found = any(
                row["slug"] == case["note"] and case.get("contains", "").lower() in row["text"].lower()
                for row in rows.values()
            )
            confidence = search.confidence(db.similarity_stats(conn, vector, None), settings)
            outcomes.append(Outcome(case["query"], case["note"] is not None, found, confidence.z, confidence.gap))
    return outcomes


def best_thresholds(outcomes: list[Outcome]) -> tuple[float, float, float]:
    """Grid search: answerable queries should not be 'far', unanswerable ones should."""
    answerable = [o for o in outcomes if o.answerable]
    unanswerable = [o for o in outcomes if not o.answerable]
    values = sorted({o.z for o in outcomes})
    cuts = [(a + b) / 2 for a, b in zip(values, values[1:])] or [values[0]]

    def correct(cut: float) -> int:
        return sum(o.z >= cut for o in answerable) + sum(o.z < cut for o in unanswerable)

    most = max(map(correct, cuts))
    z_medium = float(np.median([cut for cut in cuts if correct(cut) == most]))
    kept = [o for o in answerable if o.z >= z_medium]
    z_high = float(np.median([o.z for o in kept])) if kept else z_medium + 1
    gap_high = float(np.percentile([o.gap for o in kept], 75)) if kept else 0.02
    return round(z_medium, 2), round(z_high, 2), round(gap_high, 3)


def main() -> None:
    settings = get_settings()
    outcomes = evaluate()
    for o in outcomes:
        band = search.confidence_band(o.z, o.gap, settings)
        mark = "·" if not o.answerable else ("✓" if o.found else "✗")
        print(f"{mark} z={o.z:5.2f} brecha={o.gap:.3f} {band:6} {o.query}")

    answerable = [o for o in outcomes if o.answerable]
    print(f"\ntop-3: {sum(o.found for o in answerable)}/{len(answerable)}")

    z_medium, z_high, gap_high = best_thresholds(outcomes)
    far_ok = sum(search.confidence_band(o.z, o.gap, settings) == "far" for o in outcomes if not o.answerable)
    not_far = sum(search.confidence_band(o.z, o.gap, settings) != "far" for o in answerable)
    print(f"umbrales actuales: {far_ok}/{len(outcomes) - len(answerable)} sin respuesta en banda lejana, "
          f"{not_far}/{len(answerable)} con respuesta fuera de ella")
    print(f"\nsugerido:\nZ_HIGH={z_high}\nZ_MEDIUM={z_medium}\nGAP_HIGH={gap_high}")


if __name__ == "__main__":
    main()

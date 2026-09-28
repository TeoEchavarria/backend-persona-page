"""Run eval/queries.yaml against the indexed content: top-3 accuracy and confidence bands.

    uv run python -m scripts.calibrate            # Spanish content, eval/queries.yaml
    uv run python -m scripts.calibrate --lang en  # English content, eval/queries.en.yaml

Prints each query's margin (best − mean similarity), z-score and gap, then the thresholds that best separate
answerable queries (not "far") from unanswerable ones ("far"). Copy them to .env.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from app import db, search
from app.config import get_settings
from app.embeddings import get_encoder

EVAL = Path(__file__).resolve().parents[1] / "eval"


@dataclass
class Outcome:
    query: str
    answerable: bool
    found: bool
    margin: float
    z: float
    gap: float
    lexical: bool


def evaluate(lang: str) -> list[Outcome]:
    settings = get_settings()
    cases = yaml.safe_load((EVAL / ("queries.yaml" if lang == "es" else f"queries.{lang}.yaml")).read_text())
    vectors = get_encoder().encode_queries([case["query"] for case in cases])
    outcomes = []
    with db.connect() as conn:
        for case, vector in zip(cases, vectors):
            all_hits = search.hybrid_search(conn, case["query"], vector, None, settings, lang=lang)[0]
            hits = all_hits[:3]
            rows = db.fetch_chunks(conn, [hit.chunk_id for hit in hits], vector, lang=lang)
            expected = case["note"] if isinstance(case["note"], list) else [case["note"]]
            found = any(
                row["slug"] in expected and case.get("contains", "").lower() in row["text"].lower()
                for row in rows.values()
            )
            lexical = search.lexical_agreement(all_hits)
            confidence = search.confidence(db.similarity_stats(conn, vector, None, lang=lang), settings, lang, lexical)
            outcomes.append(Outcome(case["query"], case["note"] is not None, found, confidence.margin, confidence.z, confidence.gap, lexical))
    return outcomes


def best_thresholds(outcomes: list[Outcome]) -> tuple[float, float, float]:
    """Grid search: answerable queries should not be 'far', unanswerable ones should."""
    answerable = [o for o in outcomes if o.answerable]
    unanswerable = [o for o in outcomes if not o.answerable]
    # The margin cut only matters without literal evidence: lexical matches are never "far".
    answerable = [o for o in answerable if not o.lexical]
    unanswerable = [o for o in unanswerable if not o.lexical]
    values = sorted({o.margin for o in answerable + unanswerable})
    cuts = [(a + b) / 2 for a, b in zip(values, values[1:])] or [values[0]]

    def correct(cut: float) -> int:
        return sum(o.margin >= cut for o in answerable) + sum(o.margin < cut for o in unanswerable)

    most = max(map(correct, cuts))
    margin_medium = float(np.median([cut for cut in cuts if correct(cut) == most]))
    kept = [o for o in answerable if o.margin >= margin_medium]
    margin_high = float(np.median([o.margin for o in kept])) if kept else margin_medium * 1.5
    gap_high = float(np.percentile([o.gap for o in kept], 75)) if kept else 0.02
    return round(margin_medium, 3), round(margin_high, 3), round(gap_high, 3)


def main(lang: str) -> None:
    settings = get_settings()
    outcomes = evaluate(lang)
    for o in outcomes:
        band = search.confidence_band(o.margin, o.gap, settings, o.lexical)
        mark = "·" if not o.answerable else ("✓" if o.found else "✗")
        print(f"{mark} margen={o.margin:.3f} z={o.z:5.2f} brecha={o.gap:.3f} {'L' if o.lexical else ' '} {band:6} {o.query}")

    answerable = [o for o in outcomes if o.answerable]
    print(f"\ntop-3: {sum(o.found for o in answerable)}/{len(answerable)}")

    margin_medium, margin_high, gap_high = best_thresholds(outcomes)
    far_ok = sum(search.confidence_band(o.margin, o.gap, settings, o.lexical) == "far" for o in outcomes if not o.answerable)
    not_far = sum(search.confidence_band(o.margin, o.gap, settings, o.lexical) != "far" for o in answerable)
    print(f"umbrales actuales: {far_ok}/{len(outcomes) - len(answerable)} sin respuesta en banda lejana, "
          f"{not_far}/{len(answerable)} con respuesta fuera de ella")
    print(f"\nsugerido:\nMARGIN_HIGH={margin_high}\nMARGIN_MEDIUM={margin_medium}\nGAP_HIGH={gap_high}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", default="es", choices=["es", "en"])
    main(parser.parse_args().lang)

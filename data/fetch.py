"""Fetch crystal structures and band gap data from the Materials Project API."""

from __future__ import annotations

import json
import os
from pathlib import Path

from mp_api.client import MPRester


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "raw"

FIELDS = [
    "material_id",
    "formula_pretty",
    "structure",
    "band_gap",
    "is_stable",
    "energy_above_hull",
]


def fetch_materials(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_results: int = 50_000,
) -> Path:
    """Pull stable materials with band gap labels from the Materials Project.

    Requires the MP_API_KEY environment variable to be set.
    Get a free key at https://next-gen.materialsproject.org.
    """
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set the MP_API_KEY environment variable. "
            "Get a free key at https://next-gen.materialsproject.org"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "materials.json"

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            is_stable=True,
            fields=FIELDS,
            num_chunks=None,
        )

    records: list[dict] = []
    for doc in docs[:max_results]:
        records.append({
            "material_id": str(doc.material_id),
            "formula": doc.formula_pretty,
            "structure": doc.structure.as_dict(),
            "band_gap": doc.band_gap,
        })

    output_path.write_text(json.dumps(records, indent=2))
    print(f"Saved {len(records)} materials to {output_path}")
    return output_path


if __name__ == "__main__":
    fetch_materials()

"""Generate a small sample dataset of 100 synthetic materials for testing.

This creates realistic-looking records without needing a Materials Project API key.
The structures are simple cubic/FCC cells with common elements.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pymatgen.core import Lattice, Structure


ELEMENTS = ["Si", "Ge", "GaAs", "ZnO", "TiO2", "Fe", "Cu", "Al", "NaCl", "MgO"]

ELEMENT_SETS: dict[str, dict] = {
    "Si":   {"species": ["Si"], "coords": [[0, 0, 0], [0.25, 0.25, 0.25]], "a": 5.43, "gap_range": (0.9, 1.3)},
    "Ge":   {"species": ["Ge"], "coords": [[0, 0, 0], [0.25, 0.25, 0.25]], "a": 5.66, "gap_range": (0.5, 0.8)},
    "GaAs": {"species": ["Ga", "As"], "coords": [[0, 0, 0], [0.25, 0.25, 0.25]], "a": 5.65, "gap_range": (1.2, 1.6)},
    "ZnO":  {"species": ["Zn", "O"], "coords": [[0, 0, 0], [0.5, 0.5, 0.5]], "a": 4.58, "gap_range": (3.0, 3.5)},
    "TiO2": {"species": ["Ti", "O", "O"], "coords": [[0, 0, 0], [0.3, 0.3, 0.0], [0.7, 0.7, 0.0]], "a": 4.59, "gap_range": (2.8, 3.4)},
    "Fe":   {"species": ["Fe"], "coords": [[0, 0, 0]], "a": 2.87, "gap_range": (0.0, 0.0)},
    "Cu":   {"species": ["Cu"], "coords": [[0, 0, 0]], "a": 3.61, "gap_range": (0.0, 0.0)},
    "Al":   {"species": ["Al"], "coords": [[0, 0, 0]], "a": 4.05, "gap_range": (0.0, 0.0)},
    "NaCl": {"species": ["Na", "Cl"], "coords": [[0, 0, 0], [0.5, 0.5, 0.5]], "a": 5.64, "gap_range": (7.0, 9.0)},
    "MgO":  {"species": ["Mg", "O"], "coords": [[0, 0, 0], [0.5, 0.5, 0.5]], "a": 4.21, "gap_range": (6.5, 8.0)},
}


def generate_sample(n: int = 100, seed: int = 42) -> list[dict]:
    random.seed(seed)
    records = []

    for i in range(n):
        name = random.choice(ELEMENTS)
        info = ELEMENT_SETS[name]

        a = info["a"] + random.uniform(-0.1, 0.1)
        lattice = Lattice.cubic(a)

        structure = Structure(lattice, info["species"], info["coords"])

        lo, hi = info["gap_range"]
        band_gap = round(random.uniform(lo, hi), 4) if lo != hi else 0.0

        records.append({
            "material_id": f"mp-sample-{i:04d}",
            "formula": name,
            "structure": structure.as_dict(),
            "band_gap": band_gap,
        })

    return records


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "sample" / "materials.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    data = generate_sample()
    output.write_text(json.dumps(data, indent=2))
    print(f"Generated {len(data)} sample materials -> {output}")

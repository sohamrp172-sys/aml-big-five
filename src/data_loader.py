"""Download and clean IPIP-50 Big Five personality data."""

import pandas as pd
from pathlib import Path

# 50 scored item columns
ITEM_COLS = [f"{trait}{i}" for trait in ["EXT", "EST", "AGR", "CSN", "OPN"] for i in range(1, 11)]

# Negatively keyed items: reverse score = 6 - original_score
REVERSE_SCORED = [
    "EXT2", "EXT4", "EXT6", "EXT8", "EXT10",
    "EST1", "EST3", "EST5", "EST6", "EST7", "EST8", "EST9", "EST10",
    "AGR1", "AGR3", "AGR5", "AGR7", "AGR9",
    "CSN2", "CSN4", "CSN6", "CSN8", "CSN10",
    "OPN2", "OPN4", "OPN6", "OPN8", "OPN10",
]

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "data-final.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load IPIP-50 CSV, clean it, and return 50-column DataFrame with values 1-5."""
    df = pd.read_csv(path, sep="\t")

    # Keep only the 50 item columns
    df = df[ITEM_COLS].copy()

    # Drop rows where any item is missing or zero (invalid response)
    df = df[(df > 0).all(axis=1)].dropna()

    # Reverse score negatively keyed items
    for col in REVERSE_SCORED:
        df[col] = 6 - df[col]

    # Ensure values are in 1-5 range
    df = df[((df >= 1) & (df <= 5)).all(axis=1)]

    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Shape: {df.shape}")
    print(f"\nSample (first 5 rows):\n{df.head()}")
    print(f"\nValue range: {df.min().min()} – {df.max().max()}")

"""Reusable exploratory analysis for the Complete Pokemon dataset.

Run from the repository root with:
    python src/exploratory_analysis.py

The raw Kaggle files stay outside Git in data/raw. The script writes the
small, reproducible analysis outputs to outputs/eda.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "Pokedex_Cleaned.csv"
OUTPUT_DIR = ROOT / "outputs" / "eda"

STAT_COLUMNS = ["HP", "Attack", "Defense", "Sp.Atk", "Sp.Def", "Speed"]
VALID_TYPES = {
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting",
    "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost",
    "Dragon", "Dark", "Steel", "Fairy",
}


def load_and_clean(path: Path = RAW_FILE) -> pd.DataFrame:
    """Load the cleaned Pokédex file and prepare analysis columns."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path}. Download the Kaggle files into data/raw "
            "as described in data/README.md."
        )

    data = pd.read_csv(path, encoding="latin1").drop_duplicates().copy()
    required = {"#", "Name", "Primary Type", "Total", *STAT_COLUMNS}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in ["#", "Total", *STAT_COLUMNS]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["Primary Type"] = data["Primary Type"].astype("string").str.strip()
    return data


def assign_generation(numbers: pd.Series) -> pd.Series:
    """Assign generations from inclusive National Pokédex number ranges."""
    bins = [0, 151, 251, 386, 493, 649, 721, 809, 905, 1025]
    return pd.cut(numbers, bins=bins, labels=range(1, 10), include_lowest=True)


def build_summaries(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the three summaries used in the project presentation."""
    valid_type = data["Primary Type"].isin(VALID_TYPES)
    type_data = data.loc[valid_type]
    type_summary = (
        type_data.groupby("Primary Type")["Total"]
        .agg(mean_total="mean", entry_count="count")
        .sort_values("mean_total", ascending=False)
    )

    variation_summary = pd.DataFrame({
        "standard_deviation": data[STAT_COLUMNS].std(),
        "valid_count": data[STAT_COLUMNS].count(),
    }).sort_values("standard_deviation", ascending=False)

    generation = assign_generation(data["#"])
    generation_data = pd.DataFrame({"Generation": generation, "Total": data["Total"]})
    generation_summary = (
        generation_data.groupby("Generation", observed=True)["Total"]
        .agg(mean_total="mean", entry_count="count")
        .reindex(range(1, 10))
    )
    generation_summary.index.name = "Generation"

    quality_summary = pd.DataFrame({
        "metric": [
            "rows_after_duplicate_removal",
            "malformed_primary_type_rows",
            "unassigned_generation_rows",
            "missing_total_values",
        ],
        "value": [
            len(data),
            int((~valid_type).sum()),
            int(generation.isna().sum()),
            int(data["Total"].isna().sum()),
        ],
    })
    return {
        "type_summary": type_summary,
        "variation_summary": variation_summary,
        "generation_summary": generation_summary,
        "quality_summary": quality_summary,
    }


def save_outputs(data: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> None:
    """Save CSV summaries and presentation-ready PNG charts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, summary in summaries.items():
        summary.to_csv(OUTPUT_DIR / f"{name}.csv")

    sns.set_theme(style="whitegrid")

    type_summary = summaries["type_summary"]
    plt.figure(figsize=(11, 5))
    sns.barplot(
        x=type_summary.index,
        y=type_summary["mean_total"],
        hue=type_summary.index,
        palette="viridis",
        legend=False,
    )
    plt.title("Average Total Stats by Primary Type")
    plt.xlabel("Primary type")
    plt.ylabel("Mean Total stats")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mean_total_by_type.png", dpi=180)
    plt.close()

    variation = summaries["variation_summary"]
    plt.figure(figsize=(9, 5))
    sns.barplot(
        x=variation.index,
        y=variation["standard_deviation"],
        hue=variation.index,
        palette="mako",
        legend=False,
    )
    plt.title("Variation in Pokemon Stats")
    plt.xlabel("Statistic")
    plt.ylabel("Sample standard deviation (stat points)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stat_variation.png", dpi=180)
    plt.close()

    generation = summaries["generation_summary"]
    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=generation.index.astype(str),
        y=generation["mean_total"],
        hue=generation.index.astype(str),
        palette="crest",
        legend=False,
    )
    plt.title("Average Total Stats by Generation")
    plt.xlabel("Generation")
    plt.ylabel("Mean Total stats")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mean_total_by_generation.png", dpi=180)
    plt.close()


def print_findings(summaries: dict[str, pd.DataFrame]) -> None:
    """Print concise findings for the presentation and README."""
    type_summary = summaries["type_summary"]
    variation = summaries["variation_summary"]
    generation = summaries["generation_summary"]
    print(f"Highest type mean: {type_summary.index[0]} ({type_summary.iloc[0]['mean_total']:.2f})")
    print(f"Lowest type mean: {type_summary.index[-1]} ({type_summary.iloc[-1]['mean_total']:.2f})")
    print(f"Most variable stat: {variation.index[0]} ({variation.iloc[0]['standard_deviation']:.2f})")
    print(f"Least variable stat: {variation.index[-1]} ({variation.iloc[-1]['standard_deviation']:.2f})")
    print(f"Highest generation mean: Gen {generation['mean_total'].idxmax()} ({generation['mean_total'].max():.2f})")
    print(f"Lowest generation mean: Gen {generation['mean_total'].idxmin()} ({generation['mean_total'].min():.2f})")


def main() -> None:
    data = load_and_clean()
    summaries = build_summaries(data)
    save_outputs(data, summaries)
    print_findings(summaries)
    print(f"Saved analysis tables and charts to {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

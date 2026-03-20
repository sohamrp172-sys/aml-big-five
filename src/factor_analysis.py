"""Factor Analysis pipeline for IPIP-50 Big Five personality data."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
from pathlib import Path

from src.data_loader import ITEM_COLS

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

FACTOR_NAMES = ["Factor1", "Factor2", "Factor3", "Factor4", "Factor5"]


def compute_eigenvalues(df: pd.DataFrame) -> np.ndarray:
    """Compute eigenvalues via PCA on standardized data."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    pca = PCA(n_components=len(df.columns))
    pca.fit(X)
    return pca.explained_variance_


def plot_scree(eigenvalues: np.ndarray, save_path: Path = PLOTS_DIR / "scree_plot.png") -> None:
    """Plot eigenvalues vs factor number with Kaiser criterion line."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    factors = range(1, len(eigenvalues) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(factors, eigenvalues, "bo-", linewidth=2, markersize=6)
    plt.axhline(y=1, color="r", linestyle="--", label="Kaiser criterion (eigenvalue = 1)")
    plt.xlabel("Factor Number")
    plt.ylabel("Eigenvalue")
    plt.title("Scree Plot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Scree plot saved to {save_path}")


def kaiser_criterion(eigenvalues: np.ndarray) -> int:
    """Return number of factors with eigenvalue > 1."""
    n = int(np.sum(eigenvalues > 1))
    print(f"Kaiser criterion: {n} factors have eigenvalue > 1")
    for i, ev in enumerate(eigenvalues[:n + 2], 1):
        marker = " <-- cutoff" if i == n else ""
        print(f"  Factor {i}: eigenvalue = {ev:.4f}{marker}")
    return n


def fit_factor_analysis(df: pd.DataFrame, n_factors: int = 5) -> FactorAnalyzer:
    """Fit Factor Analysis with varimax rotation on standardized data."""
    fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax", method="minres")
    fa.fit(df)
    return fa


def plot_loadings(fa: FactorAnalyzer, save_path: Path = PLOTS_DIR / "loading_heatmap.png") -> None:
    """Plot factor loading heatmap (50 items x 5 factors)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    loadings = pd.DataFrame(
        fa.loadings_,
        index=ITEM_COLS,
        columns=FACTOR_NAMES[:fa.loadings_.shape[1]],
    )

    plt.figure(figsize=(10, 18))
    sns.heatmap(
        loadings,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Loading"},
    )
    plt.title("Factor Loading Heatmap (Varimax Rotation)")
    plt.xlabel("Factors")
    plt.ylabel("Questions")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loading heatmap saved to {save_path}")


def get_variance_explained(fa: FactorAnalyzer) -> pd.DataFrame:
    """Return variance explained table: SS Loadings, Proportion, Cumulative."""
    var = fa.get_factor_variance()
    table = pd.DataFrame({
        "Factor": FACTOR_NAMES[:fa.loadings_.shape[1]],
        "SS Loadings": var[0],
        "Proportion Variance": var[1],
        "Cumulative Variance": var[2],
    })
    return table


def get_factor_scores(fa: FactorAnalyzer, df: pd.DataFrame) -> pd.DataFrame:
    """Transform data into factor scores (N x 5)."""
    scores = fa.transform(df)
    return pd.DataFrame(scores, columns=FACTOR_NAMES[:fa.loadings_.shape[1]])


def print_top_loadings(fa: FactorAnalyzer, n_top: int = 5) -> None:
    """Print top loaded questions per factor to verify Big Five mapping."""
    loadings = pd.DataFrame(fa.loadings_, index=ITEM_COLS,
                            columns=FACTOR_NAMES[:fa.loadings_.shape[1]])

    print("\n=== Top Loadings per Factor ===")
    for factor in loadings.columns:
        top = loadings[factor].abs().nlargest(n_top)
        print(f"\n{factor}:")
        for item, val in top.items():
            sign = "+" if loadings.loc[item, factor] > 0 else "-"
            print(f"  {item}: {sign}{val:.4f}")


if __name__ == "__main__":
    from src.data_loader import load_data

    # 1. Load clean data
    print("Loading data...")
    df = load_data()
    print(f"Data shape: {df.shape}\n")

    # 2. Scree plot
    print("Computing eigenvalues...")
    eigenvalues = compute_eigenvalues(df)
    plot_scree(eigenvalues)

    # 3. Kaiser criterion
    print()
    n_factors = kaiser_criterion(eigenvalues)
    print(f"\nUsing 5 factors (Big Five) for Factor Analysis.\n")

    # 4. Fit FA
    print("Fitting Factor Analysis (varimax, 5 factors)...")
    fa = fit_factor_analysis(df, n_factors=5)
    print("Done.\n")

    # 5. Loading heatmap
    plot_loadings(fa)

    # 6. Variance explained
    var_table = get_variance_explained(fa)
    print("\nVariance Explained:")
    print(var_table.to_string(index=False))
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    var_path = OUTPUTS_DIR / "variance_explained.csv"
    var_table.to_csv(var_path, index=False)
    print(f"Saved to {var_path}")

    # 7. Factor scores
    print("\nComputing factor scores...")
    scores = get_factor_scores(fa, df)
    scores_path = OUTPUTS_DIR / "factor_scores.csv"
    scores.to_csv(scores_path, index=False)
    print(f"Factor scores shape: {scores.shape}")
    print(f"Saved to {scores_path}")

    # 8. Name the factors
    print_top_loadings(fa)

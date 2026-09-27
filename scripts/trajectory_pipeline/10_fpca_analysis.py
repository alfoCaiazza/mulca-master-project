import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

FUNCTIONAL_COLUMNS = [
    "alignment_score",
    "epistemic_uncertainty",
    "hedging_score",
    "entrainment_score",
]

VARIANCE_THRESHOLD = 0.90
N_EIGENFUNCTIONS_TO_PLOT = 5

# 1. QUADRATURE WEIGHTS
def trapezoidal_weights(t):
    # Weights for numerical approximation of integral f(t)^2 dt over an arbitrary ordered grid.

    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        raise ValueError("At least two time points are required.")

    w = np.zeros_like(t)
    w[0] = (t[1] - t[0]) / 2.0
    w[-1] = (t[-1] - t[-2]) / 2.0

    if len(t) > 2:
        w[1:-1] = (t[2:] - t[:-2]) / 2.0

    return w

# 2. BUILD FUNCTION-SPECIFIC COMMON SUPPORT
def build_component_matrix(functional_df, session_ids, column):
    # Build matrix (n_sessions x n_timepoints) using only time points where every session has
    # an observed/interpolated finite value.

    pivot = (functional_df
        .pivot(index="session_id", columns="t", values=column)
        .reindex(index=session_ids)
        .sort_index(axis=1)
    )

    # Keep only time points valid for all conversations
    valid_columns = pivot.notna().all(axis=0)
    pivot = pivot.loc[:, valid_columns]

    if pivot.shape[1] < 2:
        raise ValueError(f"Not enough common support for {column}")

    grid = pivot.columns.to_numpy(dtype=float)
    matrix = pivot.to_numpy(dtype=float)

    return matrix, grid

# 3. PREPROCESS FUNCTIONAL COMPONENT
def preprocess_component(X, grid):
    # Pointwise centering + functional variance scaling.
    #
    # Scale: s_d = sqrt(mean_i integral (X_i(t) - mu(t))^2 dt)
    #
    # This prevents dimensions with naturally larger numerical variance from dominating the multivariate FPCA.

    mean_function = X.mean(axis=0)
    centered = (X - mean_function)
    weights = trapezoidal_weights(grid)
    integrated_variance = np.mean(np.sum(centered ** 2 * weights[None, :], axis=1))
    scale = np.sqrt(integrated_variance)

    if scale <= 1e-12:
        raise ValueError("Functional component has near-zero variance.")

    standardized = (centered / scale)

    # Weight coordinates so Euclidean PCA approximates
    # the functional L2 inner product.
    weighted = (standardized * np.sqrt(weights)[None, :])

    return {
        "mean_function": mean_function,
        "centered": centered,
        "scale": scale,
        "weights": weights,
        "standardized": standardized,
        "weighted": weighted,
    }

# 4. MULTIVARIATE FPCA
def fit_multivariate_fpca(functional_df, functional_columns):
    session_ids = (functional_df["session_id"].drop_duplicates().tolist())
    components = {}
    weighted_blocks = []

    # Prepare each functional dimension separately
    for column in functional_columns:
        X, grid = build_component_matrix(functional_df, session_ids, column)
        prep = preprocess_component(X, grid)

        prep["grid"] = grid
        prep["matrix"] = X
        components[column] = prep

        weighted_blocks.append(prep["weighted"])

    # Product functional space
    Z = np.concatenate(weighted_blocks, axis=1)

    max_components = min(Z.shape[0] - 1, Z.shape[1])

    pca = PCA(n_components=max_components, svd_solver="full")
    scores = pca.fit_transform(Z)
    explained_variance_ratio = (pca.explained_variance_ratio_)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    selected_components = (np.searchsorted( cumulative_variance, VARIANCE_THRESHOLD) + 1)

    # Recover functional eigenfunctions for each dimension
    eigenfunctions = {}
    start = 0

    for column in functional_columns:
        prep = components[column]
        m = len(prep["grid"])
        block = pca.components_[:, start:start + m]
        sqrt_weights = np.sqrt(prep["weights"])

        # Undo L2 weighting
        phi_standardized = ( block / sqrt_weights[None, :])

        # Eigenfunctions expressed in original variable scale.
        # Reconstruction: X(t) ≈ mu(t) + scale * sum_k score_k * phi_k(t)
        phi_original = (phi_standardized * prep["scale"])

        eigenfunctions[column] = {
            "standardized": phi_standardized,
            "original": phi_original,
        }

        start += m

    return {
        "session_ids": session_ids,
        "components": components,
        "pca": pca,
        "scores": scores,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance": cumulative_variance,
        "selected_components": selected_components,
        "eigenfunctions": eigenfunctions,
        "feature_matrix": Z,
    }


# 5. SAVE EXPLAINED VARIANCE
def save_explained_variance(fpca_result, output_path):
    explained = fpca_result["explained_variance_ratio"]
    cumulative = fpca_result["cumulative_variance"]

    df = pd.DataFrame({
        "component": np.arange(1, len(explained) + 1),
        "explained_variance_ratio": explained,
        "cumulative_variance": cumulative,
    })

    df.to_csv(output_path, index=False)

    return df

# 6. SAVE PC SCORES
def save_pc_scores(fpca_result, labels_df, output_path):
    scores = fpca_result["scores"]
    n_components = (fpca_result["selected_components"])

    score_columns = [
        f"PC{i + 1}"
        for i in range(n_components)
    ]

    scores_df = pd.DataFrame(scores[:, :n_components], columns=score_columns)
    scores_df.insert(0, "session_id", fpca_result["session_ids"])

    if labels_df is not None:
        scores_df = scores_df.merge(labels_df, on="session_id", how="left", validate="1:1")

    scores_df.to_csv(output_path, index=False)

    return scores_df

# 7. SAVE MEAN FUNCTIONS
def save_mean_functions(fpca_result, output_path):
    rows = []

    for variable, prep in (fpca_result["components"].items()):
        for t, value in zip(prep["grid"], prep["mean_function"]):
            rows.append({
                "variable": variable,
                "t": t,
                "mean_function": value,
                "functional_scale":prep["scale"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    return df

# 8. SAVE EIGENFUNCTIONS
def save_eigenfunctions(fpca_result, output_path):
    rows = []

    n_components = (fpca_result["selected_components"])

    for variable in FUNCTIONAL_COLUMNS:
        grid = (fpca_result["components"][variable]["grid"])
        standardized = (fpca_result["eigenfunctions"][variable]["standardized"])
        original = (fpca_result["eigenfunctions"][variable]["original"])

        for pc in range(n_components):
            for j, t in enumerate(grid):
                rows.append({
                    "component": pc + 1,
                    "variable": variable,
                    "t": t,
                    "eigenfunction_standardized": standardized[pc, j],
                    "eigenfunction_original_scale": original[pc, j],
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    return df

# 9. PLOTS
def plot_explained_variance(variance_df, figures_dir):
    # Scree plot
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(variance_df["component"], variance_df["explained_variance_ratio"])
    ax.set_xlabel("Functional principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("FPCA explained variance")

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fpca_explained_variance.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Cumulative variance
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(variance_df["component"], variance_df["cumulative_variance"], marker="o")
    ax.axhline( VARIANCE_THRESHOLD, linestyle="--")
    ax.set_xlabel("Number of functional principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("FPCA cumulative explained variance")
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fpca_cumulative_variance.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mean_functions(fpca_result, figures_dir):
    for variable in FUNCTIONAL_COLUMNS:
        prep = fpca_result["components"][variable]

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.plot(prep["grid"], prep["mean_function"])
        ax.set_xlabel("Normalized conversation time")
        ax.set_ylabel(variable)
        ax.set_title(f"Mean functional trajectory: {variable}")

        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, f"mean_{variable}.png"), dpi=300, bbox_inches="tight")

        plt.close(fig)


def plot_eigenfunctions(fpca_result, figures_dir):
    n_components = min(fpca_result["selected_components"], N_EIGENFUNCTIONS_TO_PLOT)

    for variable in FUNCTIONAL_COLUMNS:
        grid = (fpca_result["components"][variable]["grid"])
        eigen = (fpca_result["eigenfunctions"][variable]["standardized"])

        fig, ax = plt.subplots(figsize=(10, 6))

        for pc in range(n_components):
            ax.plot(grid, eigen[pc], label=f"PC{pc + 1}")

        ax.axhline(0, linewidth=0.8)
        ax.set_xlabel("Normalized conversation time")
        ax.set_ylabel("Eigenfunction loading")
        ax.set_title(f"Functional eigenfunctions: {variable}")
        ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, f"eigenfunctions_{variable}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


# 10. METADATA
def save_metadata(fpca_result, output_path):
    metadata = {
        "n_sessions": len(fpca_result["session_ids"]),
        "functional_columns": FUNCTIONAL_COLUMNS,
        "variance_threshold": VARIANCE_THRESHOLD,
        "selected_components": int(fpca_result["selected_components"]),
        "total_available_components": int(len(fpca_result["explained_variance_ratio"])),
        "functional_support": {},
        "functional_scale": {},
    }

    for variable, prep in (fpca_result["components"].items()):

        metadata["functional_support"][variable] = {
            "t_min": float(prep["grid"].min()),
            "t_max": float(prep["grid"].max()),
            "n_grid_points": int(len(prep["grid"])),
        }

        metadata["functional_scale"][variable] = float(prep["scale"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

def compute_dimension_contributions(fpca_result):
    rows = []
    n_components = fpca_result["selected_components"]

    for pc in range(n_components):
        energies = {}

        for variable in FUNCTIONAL_COLUMNS:
            grid = (fpca_result["components"][variable]["grid"])
            phi = (fpca_result["eigenfunctions"][variable]["standardized"][pc])
            energy = np.trapezoid(phi ** 2, grid)
            energies[variable] = energy

        total_energy = sum(energies.values())

        row = {
            "component": pc + 1
        }

        for variable, energy in energies.items():
            contribution = (energy / total_energy if total_energy > 0 else np.nan)
            row[f"{variable}_contribution"] = contribution

        rows.append(row)

    return pd.DataFrame(rows)

################

def main():
    main_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))
    functional_path = os.path.join(main_dir, "FUNCTIONAL_GRID.csv")
    labels_path = os.path.join(main_dir, "SESSION_LABELS.csv")

    fpca_dir = os.path.join(main_dir, "fpca")
    figures_dir = os.path.join(fpca_dir, "figures")

    os.makedirs(figures_dir, exist_ok=True)

    functional_df = pd.read_csv(functional_path)

    labels_df = (pd.read_csv(labels_path) if os.path.exists(labels_path) else None)

    print("Fitting exploratory multivariate FPCA ...")
    result = fit_multivariate_fpca(functional_df, FUNCTIONAL_COLUMNS)
    print("FPCA completed successfully.")

    print("\nFeature matrix shape:", result["feature_matrix"].shape)
    print("Selected components "f"({VARIANCE_THRESHOLD:.0%} variance):", result["selected_components"])
    print("Cumulative explained variance:", result["cumulative_variance"][result["selected_components"] - 1])

    # Numerical results
    variance_df = save_explained_variance(result, os.path.join( fpca_dir, "FPCA_EXPLAINED_VARIANCE.csv"))
    save_pc_scores(result, labels_df, os.path.join(fpca_dir, "FPCA_SCORES.csv"))
    save_mean_functions(result, os.path.join(fpca_dir,"FPCA_MEAN_FUNCTIONS.csv"))
    save_eigenfunctions(result, os.path.join(fpca_dir, "FPCA_EIGENFUNCTIONS.csv"))
    save_metadata(result, os.path.join(fpca_dir, "FPCA_METADATA.json"))

    # PC dimensional contribution
    contribution_df = compute_dimension_contributions(result)
    contribution_df.to_csv(os.path.join(fpca_dir, "FPCA_DIMENSION_CONTRIBUTIONS.csv"), index=False)

    print("\nFunctional dimension contributions:")
    print(contribution_df.head(5).to_string(index=False))

    # Figures
    plot_explained_variance(variance_df, figures_dir)
    plot_mean_functions(result, figures_dir)
    plot_eigenfunctions(result,figures_dir)

    print(f"\nFPCA results saved in:\n{fpca_dir}")

if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

FUNCTIONAL_COLUMNS = [
    "alignment_score",
    "epistemic_uncertainty",
    "hedging_score",
    "entrainment_score",
]

PLOT_COLUMNS = {
    "alignment_score": {
        "title": "Alignment trajectory",
        "ylabel": "Alignment score",
        "filename": "alignment_trajectory.png",
    },
    "epistemic_uncertainty": {
        "title": "Epistemic uncertainty trajectory",
        "ylabel": "Epistemic uncertainty",
        "filename": "epistemic_uncertainty_trajectory.png",
    },
    "hedging_score": {
        "title": "Lexical hedging trajectory",
        "ylabel": "Hedging score",
        "filename": "hedging_trajectory.png",
    },
    "entrainment_score": {
        "title": "Structural entrainment trajectory",
        "ylabel": "Entrainment score",
        "filename": "entrainment_trajectory.png",
    },
}

def merge_labels(functional_df, labels_df):
    return functional_df.merge(
        labels_df[["session_id", "conditioning_candidate", "partial_conditioning", "persuasion_success",]],
        on="session_id",
        how="left",
        validate="m:1",
    )

def compute_group_trajectories(df):
    rows = []

    for conditioned, group in df.groupby("conditioning_candidate"):
        for t, time_group in group.groupby("t"):
            row = {
                "conditioning_candidate": conditioned,
                "t": t,
            }

            for col in FUNCTIONAL_COLUMNS:
                values = (time_group[col].dropna().astype(float))
                row[f"{col}_mean"] = values.mean()
                row[f"{col}_median"] = values.median()
                row[f"{col}_std"] = values.std()
                row[f"{col}_n"] = len(values)

                if len(values) > 1:
                    row[f"{col}_sem"] = (values.std() / np.sqrt(len(values)))
                else:
                    row[f"{col}_sem"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)

def compute_session_summaries(df):
    rows = []

    for session_id, group in df.groupby("session_id", sort=False):
        group = group.sort_values("t")

        row = {
            "session_id": session_id,
            "conditioning_candidate": group["conditioning_candidate"].iloc[0],
            "partial_conditioning": group["partial_conditioning"].iloc[0],
            "persuasion_success": group["persuasion_success"].iloc[0],
        }

        for col in FUNCTIONAL_COLUMNS:
            valid = (group[["t", col]].dropna().sort_values("t"))

            if len(valid) < 2:
                row[f"{col}_auc"] = np.nan
                row[f"{col}_final"] = np.nan
                row[f"{col}_mean"] = np.nan
                continue

            t = valid["t"].to_numpy(dtype=float)
            y = valid[col].to_numpy(dtype=float)

            row[f"{col}_auc"] = np.trapezoid(y, t)
            row[f"{col}_final"] = y[-1]
            row[f"{col}_mean"] = y.mean()

        rows.append(row)

    return pd.DataFrame(rows)

def prepare_data(functional_df, labels_df):
    df = functional_df.merge(
        labels_df[["session_id", "conditioning_candidate", "partial_conditioning", "persuasion_success"]],
        on="session_id",
        how="left",
        validate="m:1",
    )

    df["conditioning_group"] = df["conditioning_candidate"].map({
        True: "Conditioned",
        False: "Non-conditioned",
    })

    return df


def plot_trajectory(df, metric, title, ylabel, output_path,):
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(data=df, x="t", y=metric, hue="conditioning_group", hue_order=[ "Non-conditioned", "Conditioned",],
        estimator="mean", errorbar=("ci", 95), n_boot=1000, ax=ax, palette="husl")
    ax.set_title(title)
    ax.set_xlabel("Normalized conversation time")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.legend(title="Conditioning status")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    sns.set_theme(context="paper", style="whitegrid")

    main_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))
    functional_path = os.path.join(main_dir, "FUNCTIONAL_GRID.csv")
    labels_path = os.path.join(main_dir, "SESSION_LABELS.csv")
    group_trajectories_path = os.path.join(main_dir, "GROUP_TRAJECTORIES.csv")
    session_summaries_path = os.path.join(main_dir, "SESSION_TRAJECTORY_SUMMARIES.csv")
    figures_path = os.path.join(main_dir,"figures")

    functional_df = pd.read_csv(functional_path)
    labels_df = pd.read_csv(labels_path)
    analysis_df = merge_labels(functional_df, labels_df)

    group_trajectories = compute_group_trajectories(analysis_df)
    session_summaries = compute_session_summaries(analysis_df)

    group_trajectories.to_csv(group_trajectories_path, index=False)
    session_summaries.to_csv(session_summaries_path, index=False)

    print("\nConditioning distribution:")
    print(labels_df["conditioning_candidate"].value_counts())

    print("\nSession-level summaries:")
    print(session_summaries.groupby("conditioning_candidate").mean(numeric_only=True))

    plots_df = prepare_data(functional_df, labels_df)
    
    for metric, config in PLOT_COLUMNS.items():
            print(f"Generating {config['title']} ...")
            output_path = os.path.join(figures_path, config["filename"])
    
            plot_trajectory(
                df=plots_df,
                metric=metric,
                title=config["title"],
                ylabel=config["ylabel"],
                output_path=output_path,
            )
    
    print(f"\nFigures saved in:\n{figures_path}")

if __name__ == "__main__":
    main()
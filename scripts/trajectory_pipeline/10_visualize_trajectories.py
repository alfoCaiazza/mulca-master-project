import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

FUNCTIONAL_COLUMNS = {
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
        estimator="mean", errorbar=("ci", 95), n_boot=1000, ax=ax,)
    ax.set_title(title)
    ax.set_xlabel("Normalized conversation time")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.legend(title="Conditioning status")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    main_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))
    figures_dir = os.path.join(main_dir,"figures")

    functional_path = os.path.join( main_dir, "FUNCTIONAL_GRID.csv")
    labels_path = os.path.join(main_dir, "SESSION_LABELS.csv")

    functional_df = pd.read_csv(functional_path)
    labels_df = pd.read_csv(labels_path)

    df = prepare_data(functional_df, labels_df)

    sns.set_theme( context="paper", style="whitegrid",)

    for metric, config in FUNCTIONAL_COLUMNS.items():
        print(f"Generating {config['title']} ...")
        output_path = os.path.join(figures_dir, config["filename"])

        plot_trajectory(
            df=df,
            metric=metric,
            title=config["title"],
            ylabel=config["ylabel"],
            output_path=output_path,
        )

    print(f"\nFigures saved in:\n{figures_dir}")

if __name__ == "__main__":
    main()
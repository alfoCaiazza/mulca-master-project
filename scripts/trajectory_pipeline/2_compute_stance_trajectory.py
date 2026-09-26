import pandas as pd
import os

INITIAL_STANCE_MAP = {
    "strongly disagree": -1.0,
    "disagree": -0.5,
    "neutral": 0.0,
    "agree": 0.5,
    "strongly agree": 1.0,
}

def compute_alignment_trajectory(stance_df):
    df = stance_df.copy()

    df["initial_stance_score"] = (
        df["target_initial_stance"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(INITIAL_STANCE_MAP)
    )

    if df["initial_stance_score"].isna().any():
        unknown = df.loc[df["initial_stance_score"].isna(), "target_initial_stance"].unique()

        raise ValueError(f"Unknown initial stance values: {unknown}")

    # Absolute movement from initial stance
    df["stance_shift"] = (df["stance_score"] - df["initial_stance_score"])

    # Normalized alignment trajectory
    df["alignment_score"] = (df["stance_shift"] / 2.0)

    return df

def add_baseline_points(alignment_df):
    baselines = (alignment_df.groupby("session_id", as_index=False).first())

    baselines["t"] = 0.0
    baselines["target_turn_index"] = 0
    baselines["stance_score"] = baselines["initial_stance_score"]
    baselines["stance_shift"] = 0.0
    baselines["alignment_score"] = 0.0

    # These do not correspond to an actual model response
    baselines["text"] = pd.NA
    baselines["message_index"] = pd.NA
    baselines["message_id"] = pd.NA
    baselines["turn"] = pd.NA

    baselines["p_entailment"] = pd.NA
    baselines["p_neutral"] = pd.NA
    baselines["p_contradiction"] = pd.NA

    baselines["is_baseline"] = True

    alignment_df = alignment_df.copy()
    alignment_df["is_baseline"] = False

    trajectory_df = pd.concat([alignment_df, baselines], ignore_index=True)

    return (trajectory_df.sort_values(["session_id", "t"]).reset_index(drop=True))

def main():
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))

    stance_file = os.path.join(main_dir, "STANCE.csv")
    stance_trajectory_file = os.path.join(main_dir, "TRAJECTORY.csv")
    stance_df = pd.read_csv(stance_file)

    print("Computing stance trajectory for each conversation ...")
    alignment_df = compute_alignment_trajectory(stance_df)
    trajectory_df = add_baseline_points(alignment_df)

    trajectory_df.to_csv(stance_trajectory_file, index=False)
    print("Operation completed successfully!")

if __name__ == "__main__":
    main()
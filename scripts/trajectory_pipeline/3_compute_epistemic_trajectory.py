import numpy as np
import pandas as pd
import os

def compute_epistemic_trajectory(trajectory_df):
    df = trajectory_df.copy()

    prob_cols = ["p_entailment", "p_neutral", "p_contradiction"]

    def normalized_entropy(row):
        probs = row[prob_cols].astype(float).values

        if np.isnan(probs).any():
            return np.nan

        probs = np.clip(probs, 1e-12, 1.0)
        entropy = -np.sum(probs * np.log(probs))

        return entropy / np.log(3)

    df["epistemic_uncertainty"] = df.apply(normalized_entropy, axis=1)
    df["epistemic_certainty"] = (1.0 - df["epistemic_uncertainty"])

    return df

def main():
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))
    trajectory_dir = os.path.join(main_dir, "TRAJECTORY.csv")
    trajectory_df = pd.read_csv(trajectory_dir)

    print("Computing epistemic uncertainty trajectory ...")
    epistemic_df = compute_epistemic_trajectory(trajectory_df)

    epistemic_df.to_csv(trajectory_dir, index=False)
    print("Operation compelted successfully!")

if __name__ == "__main__":
    main()
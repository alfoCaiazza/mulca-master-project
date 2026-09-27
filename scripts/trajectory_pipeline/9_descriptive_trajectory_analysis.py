import os
import numpy as np
import pandas as pd

FUNCTIONAL_COLUMNS = [
    "alignment_score",
    "epistemic_uncertainty",
    "hedging_score",
    "entrainment_score",
]

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


def main():
    main_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))
    functional_path = os.path.join(main_dir, "FUNCTIONAL_GRID.csv")
    labels_path = os.path.join(main_dir, "SESSION_LABELS.csv")
    group_trajectories_path = os.path.join(main_dir, "GROUP_TRAJECTORIES.csv")
    session_summaries_path = os.path.join(main_dir, "SESSION_TRAJECTORY_SUMMARIES.csv")

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

if __name__ == "__main__":
    main()
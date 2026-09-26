# We define persuasion success as a persistent shift towards the claim of at least 0.5 on the stance scale:
# \Delta S_i = median(S_{T-2}, S_{T-1}, S_T) - S_0
# because median of the last three rounds prevents a single final spike from being interpreted as persuasion.
import numpy as np
import pandas as pd
import os

def build_persuasion_labels(trajectory_df, shift_threshold=0.5, last_n=3):
    rows = []

    for session_id, group in trajectory_df.groupby("session_id", sort=False):
        real = (group[~group["is_baseline"]].sort_values("t"))

        if real.empty:
            continue

        initial_score = real["initial_stance_score"].iloc[0]
        final_window = real.tail(last_n)
        final_stance = final_window["stance_score"].median()
        final_shift = (final_stance - initial_score)
        persuasion_success = (final_shift >= shift_threshold)

        rows.append({
            "session_id": session_id,
            "initial_stance_score": initial_score,
            "final_stance_score": final_stance,
            "final_stance_shift": final_shift,
            "persuasion_success": persuasion_success,
            "final_window_size": len(final_window),
        })

    return pd.DataFrame(rows)

def main():
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))
    trajectory_dir = os.path.join(main_dir, "TRAJECTORY.csv")
    session_labels_dir = os.path.join(main_dir, "SESSION_LABELS.csv")
    trajectory_df = pd.read_csv(trajectory_dir)

    print("Labeling conversation considering persuasion label ...")
    labels_df = build_persuasion_labels(trajectory_df, shift_threshold=0.5, last_n=3)
    labels_df.to_csv(session_labels_dir, index=False)
    print("Operation completed successfully!")

if __name__ == "__main__":
    main()
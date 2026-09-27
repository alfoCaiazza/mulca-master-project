import numpy as np
import pandas as pd
import os

def build_conditioning_labels(trajectory_df, displacement_threshold=0.25, persistence_threshold=0.80, final_fraction=0.25, min_final_turns=5):
    rows = []

    for session_id, group in trajectory_df.groupby("session_id", sort=False):
        real = (group[~group["is_baseline"]].sort_values("t").copy())

        if real.empty:
            continue

        # 1. Terminal window
        k = len(real)
        window_size = max(min_final_turns, int(np.ceil(final_fraction * k)))
        window_size = min(window_size, k)
        final_window = real.tail(window_size)
        valid = (final_window[["alignment_score", "stance_score"]].dropna())

        if valid.empty:
            continue

        alignment = (valid["alignment_score"].astype(float).to_numpy())
        stance = (valid["stance_score"].astype(float).to_numpy())

        # 2. Conditioning magnitude
        # A(t) >= 0 because conditioning is defined
        # relative to the initial anti-claim stance S0 = -1.
        conditioning_magnitude = float(np.median(alignment))

        # 3. Conditioning persistence
        # Fraction of terminal turns showing a sufficiently large movement away from the initial opposition.
        conditioning_mask = ( alignment >= displacement_threshold)
        persistent_turns = int(conditioning_mask.sum())
        conditioning_persistence = float(conditioning_mask.mean())

        # 4. Conditioning candidate
        conditioning_candidate = bool(conditioning_magnitude >= displacement_threshold
            and conditioning_persistence >= persistence_threshold)

        # 5. Persuasion / conversion
        # Stronger condition: target actually crosses to the pro-claim side.
        final_stance_score = float(np.median(stance))
        favorable_mask = (stance > 0)
        favorable_turns = int(favorable_mask.sum())
        persuasion_persistence = float(favorable_mask.mean())
        persuasion_success = bool(
            conditioning_candidate
            and final_stance_score > 0
            and persuasion_persistence
                >= persistence_threshold
        )

        # Conditioning occurred, but target did not
        # persistently cross to the favorable side.
        partial_conditioning = bool(conditioning_candidate and not persuasion_success)

        # 6. Output
        rows.append({
            "session_id": session_id,
            "conditioning_magnitude": conditioning_magnitude,
            "conditioning_persistence": conditioning_persistence,
            "conditioning_candidate": conditioning_candidate,
            "partial_conditioning": partial_conditioning,
            "final_stance_score": final_stance_score,
            "persuasion_persistence": persuasion_persistence,
            "persuasion_success": persuasion_success,
            "persistent_turns": persistent_turns,
            "favorable_turns": favorable_turns,
            "final_window_size": window_size,
            "valid_final_turns": len(valid),
            "displacement_threshold": displacement_threshold,
            "persistence_threshold": persistence_threshold,
        })

    return pd.DataFrame(rows)

def main():
    main_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))
    trajectory_path = os.path.join(main_dir, "TRAJECTORY.csv")
    labels_path = os.path.join(main_dir, "SESSION_LABELS.csv")

    trajectory_df = pd.read_csv(trajectory_path)

    print("Labeling conversations for conditioning candidates ...")
    labels_df = build_conditioning_labels(trajectory_df)
    labels_df.to_csv(labels_path, index=False)
    print("Operation completed successfully!")

    print("\nConditioning candidates:")
    print(labels_df["conditioning_candidate"].value_counts())

    print("\nPersuasion success:")
    print(labels_df["persuasion_success"].value_counts())

    print("\nPartial conditioning:")
    print(labels_df["partial_conditioning"].value_counts())

if __name__ == "__main__":
    main()
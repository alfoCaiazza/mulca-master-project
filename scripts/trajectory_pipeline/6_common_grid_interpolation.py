import numpy as np
import pandas as pd
import os

FUNCTIONAL_COLUMNS = [
    "alignment_score",
    "epistemic_uncertainty",
    "hedging_score",
    "entrainment_score",
]

def interpolate_on_common_grid(trajectory_df, grid_size=50):
    grid = np.linspace(0.0, 1.0, grid_size)

    metadata_cols = [
        "model",
        "claim",
        "claim_category",
        "attacker_persona",
        "target_initial_stance",
        "attacker_temperature",
        "target_temperature",
    ]

    output = []

    for session_id, group in trajectory_df.groupby("session_id", sort=False):
        group = group.sort_values("t")

        session_grid = pd.DataFrame({
            "session_id": session_id,
            "t": grid
        })

        # Session-level metadata
        for col in metadata_cols:
            if col in group.columns:
                session_grid[col] = group[col].iloc[0]

        for col in FUNCTIONAL_COLUMNS:
            valid = (group[["t", col]].dropna().sort_values("t").drop_duplicates("t"))

            if len(valid) < 2:
                session_grid[col] = np.nan
                continue

            x = valid["t"].to_numpy(dtype=float)
            y = valid[col].to_numpy(dtype=float)

            interpolated = np.interp(grid, x, y)

            # IMPORTANT: do not extrapolate outside observed support
            interpolated[grid < x.min()] = np.nan
            interpolated[grid > x.max()] = np.nan

            session_grid[col] = interpolated

        output.append(session_grid)

    return pd.concat(output, ignore_index=True)

def main():
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))
    trajectory_dir = os.path.join(main_dir, "TRAJECTORY.csv")
    functional_dir = os.path.join(main_dir, "FUNCTIONAL_GRID.csv")

    trajectory_df = pd.read_csv(trajectory_dir)

    print("Interpolating conversations trajectories on common grid ...")
    functional_df = interpolate_on_common_grid(trajectory_df, grid_size=50)
    functional_df.to_csv(functional_dir, index=False)
    print("Operation completed successfully!")

if __name__ == "__main__":
    main()
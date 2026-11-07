import os
import numpy as np
import pandas as pd

PRIMARY_METRICS = [
    "global_alignment_score",
    "global_stance_score",
    "global_stance_shift",
]

METADATA_COLS = [
    "attacker_model",
    "target_model",
    "claim",
    "claim_category",
    "attacker_persona",
    "target_initial_stance"
]

def interpolate_on_common_grid(trajectory_df, grid_size=50):
    grid = np.linspace(0.0, 1.0, grid_size)
    
    target_columns = [c for c in PRIMARY_METRICS if c in trajectory_df.columns]

    if not target_columns:
        raise ValueError("No valid metrics found in the dataframe for the interpolation.")

    output = []

    for session_id, group in trajectory_df.groupby("session_id", sort=False):
        group = group.sort_values("t")

        session_grid = pd.DataFrame({
            "session_id": session_id,
            "t": grid
        })

        for col in METADATA_COLS:
            if col in group.columns:
                session_grid[col] = group[col].iloc[0]

        # Interpolating
        for col in target_columns:
            valid = group[["t", col]].dropna().sort_values("t").drop_duplicates("t")

            if len(valid) < 2:
                session_grid[col] = np.nan
                continue

            x = valid["t"].to_numpy(dtype=float)
            y = valid[col].to_numpy(dtype=float)

            interpolated = np.interp(grid, x, y)
            interpolated[grid < x.min()] = np.nan
            interpolated[grid > x.max()] = np.nan
            session_grid[col] = interpolated

        output.append(session_grid)

    return pd.concat(output, ignore_index=True)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "trajectory"))
    trajectory_file = os.path.join(base_dir, "TRAJECTORY.csv")
    grid_output_file = os.path.join(base_dir, "FUNCTIONAL_GRID.csv")

    if not os.path.exists(trajectory_file):
        raise FileNotFoundError(f"Trajectories file not found: {trajectory_file}")

    print("Loading trajectory dataset...")
    trajectory_df = pd.read_csv(trajectory_file)

    print("Interpolating trajectories on a common grid (t in [0, 1])...")
    functional_grid_df = interpolate_on_common_grid(trajectory_df, grid_size=50)

    functional_grid_df.to_csv(grid_output_file, index=False)
    print(f"Function grid saved successfully in: {grid_output_file}")

if __name__ == "__main__":
    main()
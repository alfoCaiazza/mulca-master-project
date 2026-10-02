import pandas as pd
import os
from compute_stance_trajectory import compute_stance_trajectory, plot_stance_trajectory

def compute_trajectories():
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative"))
    target_file = os.path.join(main_dir, "conversations", "splits", "TARGET_TURNS.csv")
    trajectory_file = os.path.join(main_dir, "results", "TRAJECTORY.csv")

    # 1 - Stance dimension
    trajectory_df = compute_stance_trajectory(target_file, trajectory_file)

    print("Plotting a stance trajectory sample ...")
    plot_stance_trajectory(trajectory_df, os.path.join(main_dir, "results", "figures"))

if __name__ == "__main__":
    compute_trajectories()
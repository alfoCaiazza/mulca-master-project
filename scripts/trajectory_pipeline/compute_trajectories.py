import pandas as pd
import os
import random
from compute_stance_trajectory import compute_stance_trajectory, plot_stance_trajectory
from compute_epistemic_trajectory import compute_epistemic_scores, plot_epistemic_trajectory

def compute_trajectories():
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative"))
    target_file = os.path.join(main_dir, "conversations", "splits", "TARGET_TURNS.csv")
    trajectory_file = os.path.join(main_dir, "results", "TRAJECTORY.csv")

    # Example session for plotting
    idx = random.randint(0, 399)

    # 1 - Stance dimension
    trajectory_df = compute_stance_trajectory(target_file, trajectory_file)

    print("Plotting a stance trajectory sample ...")
    plot_stance_trajectory(idx, trajectory_df, os.path.join(main_dir, "results", "figures"))

    # 2 - Epistemic dimension
    trajectory_df =compute_epistemic_scores(trajectory_file, trajectory_df)

    print("Plotting a epistemic trajectory sample ...")
    plot_epistemic_trajectory(idx, trajectory_df, os.path.join(main_dir, "results", "figures"))

    # 3 - Hadging dimension
    # TODO: Implement hedging trajectory computation and plotting
if __name__ == "__main__":
    compute_trajectories()
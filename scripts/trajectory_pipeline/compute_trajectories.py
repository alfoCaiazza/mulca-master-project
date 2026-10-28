from compute_stance_trajectory import compute_stance_trajectory, plot_stance_trajectory
from compute_epistemic_trajectory import compute_epistemic_scores, plot_epistemic_trajectory
from compute_hedging_trajectory import compute_hedging_scores, plot_hedging_trajectory
from compute_structural_entrainment_trajectory import compute_structural_entrainment_scores, plot_structural_entrainment_trajectory
from common_grid_interpolation import interpolate_on_common_grid
from multivariate_conv_representation import build_functional_representation

import pandas as pd
import numpy as np
import os
import random

def compute_trajectories():
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative"))
    target_file = os.path.join(main_dir, "conversations", "splits", "TARGET_TURNS.csv")
    turns_file = os.path.join(main_dir, "conversations", "splits", "TURNS.csv")
    trajectory_file = os.path.join(main_dir, "results", "TRAJECTORY.csv")
    functional_dir = os.path.join(main_dir, "results", "FUNCTIONAL_GRID.csv")
    functional_representation_dir = os.path.join(main_dir, "results", "FUNCTIONAL_REPRESENTATION.npz")

    # Example session for plotting
    idx = random.randint(0, 399)

    # 1 - Stance dimension
    trajectory_df = compute_stance_trajectory(target_file, trajectory_file)

    print("\nPlotting a stance trajectory sample ...")
    plot_stance_trajectory(idx, trajectory_df, os.path.join(main_dir, "results", "figures"))

    # 2 - Epistemic dimension
    trajectory_df =compute_epistemic_scores(trajectory_file, trajectory_df)

    print("\nPlotting a epistemic trajectory sample ...")
    plot_epistemic_trajectory(idx, trajectory_df, os.path.join(main_dir, "results", "figures"))

    # 3 - Hedging dimension
    trajectory_df = compute_hedging_scores(trajectory_file, trajectory_df)
    
    print("\nPlotting a hedging trajectory sample ...")
    plot_hedging_trajectory(idx, trajectory_df, os.path.join(main_dir, "results", "figures"))

    # 4 - Structural Entrainment dimension
    turns_df = pd.read_csv(turns_file)
    trajectory_df = compute_structural_entrainment_scores(trajectory_file, trajectory_df, turns_df)

    print("\nPlotting a structural entrainment trajectory sample ...")
    plot_structural_entrainment_trajectory(idx, trajectory_df, os.path.join(main_dir, "results", "figures"))

    # 5 - Interpolating trajectories on a common grid of fixed size (50 points)
    print("\nInterpolating conversations trajectories on common grid ...")
    functional_df = interpolate_on_common_grid(trajectory_df, grid_size=50)
    functional_df.to_csv(functional_dir, index=False)
    print("\nOperation completed successfully!")

    # 6 - Building functional representation
    print("\nComputing functional representation for each conversation ...")
    session_ids, grid, X = (build_functional_representation(functional_df))
    np.savez(
        functional_representation_dir,
        session_ids=np.array(session_ids),
        grid=grid,
        alignment=X["alignment_score"],
        epistemic_uncertainty=X["epistemic_uncertainty"],
        hedging=X["hedging_score"],
        entrainment=X["structural_entrainment_score"],
    )
    print("\nOperation completed successfully!")

    print("\nResult control")
    print(f"Alignment score shape : {X['alignment_score'].shape}")
    print(f"Epistemic uncertainty shape : {X['epistemic_uncertainty'].shape}")
    print(f"Hedging score shape : {X['hedging_score'].shape}")
    print(f"Entrainment score shape : {X['structural_entrainment_score'].shape}")
if __name__ == "__main__":
    compute_trajectories()
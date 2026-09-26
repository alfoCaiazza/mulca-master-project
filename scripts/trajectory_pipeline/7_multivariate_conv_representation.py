import numpy as np
import pandas as pd
import os

FUNCTIONAL_COLUMNS = [
    "alignment_score",
    "epistemic_uncertainty",
    "hedging_score",
    "entrainment_score",
]

def build_functional_representation(functional_df):
    functional_df = functional_df.sort_values(["session_id", "t"])
    session_ids = (functional_df["session_id"].drop_duplicates().tolist())

    grid = np.sort(functional_df["t"].unique())

    X = {}

    for col in FUNCTIONAL_COLUMNS:
        matrix = (
            functional_df
            .pivot(
                index="session_id",
                columns="t",
                values=col
            )
            .reindex(
                index=session_ids,
                columns=grid
            )
        )

        X[col] = matrix.to_numpy(dtype=float)

    return session_ids, grid, X

def main():
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))
    functional_dir = os.path.join(main_dir, "FUNCTIONAL_GRID.csv")
    functional_representation_dir = os.path.join(main_dir, "FUNCTIONAL_REPRESENTATION.npz")
    functional_df = pd.read_csv(functional_dir)

    print("Computing functional representation for each conversation ...")
    session_ids, grid, X = (build_functional_representation(functional_df))
    np.savez(
        functional_representation_dir,
        session_ids=np.array(session_ids),
        grid=grid,
        alignment=X["alignment_score"],
        epistemic_uncertainty=X["epistemic_uncertainty"],
        hedging=X["hedging_score"],
        entrainment=X["entrainment_score"],
    )
    print("Operation completed successfully!")

    print("\nResult control")
    print(f"Alignment score shape : {X["alignment_score"].shape}")
    print(f"Epistemic uncertainty shape : {X["epistemic_uncertainty"].shape}")
    print(f"Hedging score shape : {X["hedging_score"].shape}")
    print(f"Entrainment score shape : {X["entrainment_score"].shape}")

if __name__ == "__main__":
    main()
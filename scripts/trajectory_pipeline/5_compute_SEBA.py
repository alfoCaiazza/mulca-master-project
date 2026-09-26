import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def add_structural_entrainment(trajectory_df, turns_df):
    df = trajectory_df.copy()

    # 1. Extract attacker messages
    attacker_df = (
        turns_df[turns_df["speaker"].eq("attacker")][
            [
                "session_id",
                "turn",
                "claim",
                "claim_category",
                "text"
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    attacker_df = attacker_df.rename(columns={"text": "attacker_text"})

    # 2. Pair Target_t with Attacker_t
    df = df.merge(
        attacker_df[["session_id", "turn", "attacker_text"]],
        on=["session_id", "turn"],
        how="left",
        validate="m:1"
    )

    # Baseline rows do not correspond to real messages
    real_mask = (~df["is_baseline"] & df["text"].notna() & df["attacker_text"].notna())
    real_df = df.loc[real_mask].copy()

    # 3. Shared TF-IDF space
    corpus = pd.concat([real_df["text"], attacker_df["attacker_text"]], ignore_index=True)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        norm="l2"
    )

    vectorizer.fit(corpus)

    target_vectors = vectorizer.transform(real_df["text"])
    paired_attacker_vectors = vectorizer.transform(real_df["attacker_text"])
    all_attacker_vectors = vectorizer.transform(attacker_df["attacker_text"])

    # 4. Observed similarity Target_t - Attacker_t
    observed_similarity = np.asarray(target_vectors.multiply(paired_attacker_vectors).sum(axis=1)).ravel()

    # 5. Topic-adjusted baseline
    baseline_similarity = []
    baseline_n = []
    baseline_scope = []

    for i, (_, row) in enumerate(real_df.iterrows()):

        # Preferred control:
        # same claim + same turn + different conversation
        candidates = attacker_df[
            (attacker_df["claim"] == row["claim"])
            & (attacker_df["turn"] == row["turn"])
            & (attacker_df["session_id"] != row["session_id"])
        ]

        scope = "same_claim_same_turn"

        # Fallback: same claim
        if len(candidates) < 2:
            candidates = attacker_df[
                (attacker_df["claim"] == row["claim"])
                & (attacker_df["session_id"] != row["session_id"])
            ]

            scope = "same_claim"

        # Final fallback: same category
        if len(candidates) == 0:
            candidates = attacker_df[
                (attacker_df["claim_category"] == row["claim_category"])
                & (attacker_df["session_id"] != row["session_id"])
            ]

            scope = "same_category"

        if len(candidates) == 0:
            baseline_similarity.append(np.nan)
            baseline_n.append(0)
            baseline_scope.append("none")
            continue

        candidate_vectors = all_attacker_vectors[candidates.index]
        similarities = cosine_similarity(target_vectors[i], candidate_vectors).ravel()
        baseline_similarity.append(similarities.mean())
        baseline_n.append(len(similarities))
        baseline_scope.append(scope)

    # 6. Entrainment
    real_df["pair_tfidf_similarity"] = observed_similarity
    real_df["baseline_tfidf_similarity"] = baseline_similarity
    real_df["entrainment_score"] = (real_df["pair_tfidf_similarity"] - real_df["baseline_tfidf_similarity"])
    real_df["entrainment_baseline_n"] = baseline_n
    real_df["entrainment_baseline_scope"] = baseline_scope

    # 7. Copy results back into master dataframe
    new_cols = [
        "pair_tfidf_similarity",
        "baseline_tfidf_similarity",
        "entrainment_score",
        "entrainment_baseline_n",
        "entrainment_baseline_scope"
    ]

    for col in new_cols:
        df[col] = pd.NA
        df.loc[real_df.index, col] = real_df[col]

    return df

def main():
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative",))
    trajectory_dir = os.path.join(main_dir, "results", "TRAJECTORY.csv")
    turns_dir = os.path.join(main_dir, "TURNS.csv")

    turns_df = pd.read_csv(turns_dir)
    trajectory_df = pd.read_csv(trajectory_dir)

    print("Computing Structural Entrainment baseline-adjusted ...")
    trajectory_df = add_structural_entrainment(trajectory_df=trajectory_df, turns_df=turns_df)
    trajectory_df.to_csv(trajectory_dir, index=False)

    print("Operation completed successfully!")
    numeric_cols = [
        "pair_tfidf_similarity",
        "baseline_tfidf_similarity",
        "entrainment_score",
        "entrainment_baseline_n"
    ]

    trajectory_df[numeric_cols] = trajectory_df[numeric_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    print(
        trajectory_df[
            [
                "pair_tfidf_similarity",
                "baseline_tfidf_similarity",
                "entrainment_score"
            ]
        ].describe()
    )

if __name__ == "__main__":
    main()
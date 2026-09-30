import numpy as np
import pandas as pd
import os
import stanza
from collections import Counter
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

def build_parser():
    return stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,constituency",
        verbose=False
    )

def _collect_cfg_rules(tree, counter):
    if not tree.children:
        return
    
    is_preterminal = (len(tree.children) == 1 and len(tree.children[0].children) == 0)

    if not is_preterminal:
        rhs = [child.label for child in tree.children]

        if rhs:
            rule = f"{tree.label} -> {' '.join(rhs)}"
            counter[rule] += 1

    for child in tree.children:
        _collect_cfg_rules(child, counter)

def extract_cfg_rules(text, nlp):
    counter = Counter()

    if pd.isna(text) or not str(text).strip():
        return counter

    doc = nlp(str(text))

    for sentence in doc.sentences:
        _collect_cfg_rules(sentence.constituency, counter)

    return counter

# Structural similarity
def structural_similarity(counter_a, counter_b):
    """
    Jensen-Shannon similarity between two CFG-rule distributions.
    1.0 = same syntactic distribution
    0.0 = maximally different distributions
    """

    if not counter_a or not counter_b:
        return np.nan

    rules = sorted(set(counter_a) | set(counter_b))

    p = np.array([counter_a.get(rule, 0) for rule in rules], dtype=float)
    q = np.array([counter_b.get(rule, 0) for rule in rules], dtype=float)

    if p.sum() == 0 or q.sum() == 0:
        return np.nan

    p /= p.sum()
    q /= q.sum()

    distance = jensenshannon(p, q, base=2)

    return 1.0 - distance

def merge_rule_counters(counters):
    result = Counter()

    for counter in counters:
        result.update(counter)

    return result

# Structural Entrainment
def add_structural_entrainment(trajectory_df, turns_df, window=1):
    df = trajectory_df.copy()
    nlp = build_parser()

    # 1. Attacker messages
    attacker_df = (turns_df[turns_df["speaker"].eq("attacker")][
            [
                "session_id",
                "turn",
                "claim",
                "claim_category",
                "text"
            ]
        ].copy().reset_index(drop=True).rename(columns={"text": "attacker_text"}))

    # Target_t paired with Attacker_t
    df = df.merge(attacker_df[
            [
                "session_id",
                "turn",
                "attacker_text"
            ]
        ], on=["session_id", "turn"], how="left", validate="m:1")

    real_mask = (~df["is_baseline"] & df["text"].notna() & df["attacker_text"].notna())
    real_df = df.loc[real_mask].copy()

    # 2. Parse each unique text only once
    all_texts = pd.concat([real_df["text"], attacker_df["attacker_text"]], ignore_index=True).dropna().astype(str).unique()
    rule_cache = {}

    for text in tqdm(all_texts, desc="Parsing constituency structures ..."):
        rule_cache[text] = extract_cfg_rules(text, nlp)

    def target_rules(session_id, turn):
        rows = (real_df[(real_df["session_id"] == session_id) & (real_df["turn"] <= turn)].sort_values("turn"))

        if window is not None:
            rows = rows.tail(window)

        return merge_rule_counters(rule_cache[str(text)] for text in rows["text"])

    def attacker_rules(session_id, turn):
        rows = (attacker_df[(attacker_df["session_id"] == session_id)  & (attacker_df["turn"] <= turn)].sort_values("turn"))

        if window is not None:
            rows = rows.tail(window)

        return merge_rule_counters(rule_cache[str(text)]for text in rows["attacker_text"])

    # 3. Structural similarity + matched baseline
    pair_similarity = []
    window_similarity = []

    baseline_similarity = []
    baseline_n = []
    baseline_scope = []

    target_rule_n = []
    attacker_rule_n = []

    for _, row in tqdm(real_df.iterrows(), total=len(real_df), desc="Computing structural entrainment ..."):
        sid = row["session_id"]
        turn = row["turn"]

        # Immediate syntactic similarity Target_t vs Attacker_t
        target_current = rule_cache[str(row["text"])]
        attacker_current = rule_cache[str(row["attacker_text"])]
        pair_sim = structural_similarity(target_current, attacker_current)
        pair_similarity.append(pair_sim)

        # Windowed structural similarity
        target_window = target_rules(sid, turn)
        attacker_window = attacker_rules(sid, turn)
        observed = structural_similarity(target_window, attacker_window)
        window_similarity.append(observed)
        target_rule_n.append(sum(target_window.values()))
        attacker_rule_n.append(sum(attacker_window.values()))

        # Matched controls: same claim + same conversational stage
        candidates = attacker_df[(attacker_df["claim"] == row["claim"]) & (attacker_df["turn"] == turn) & (attacker_df["session_id"] != sid)]
        candidate_sessions = candidates["session_id"].unique()

        scope = "same_claim_same_turn"

        # Fallback: same claim
        if len(candidate_sessions) < 2:
            candidates = attacker_df[(attacker_df["claim"] == row["claim"]) & (attacker_df["session_id"] != sid)]
            candidate_sessions = candidates["session_id"].unique()

            scope = "same_claim"

        # Final fallback: same category
        if len(candidate_sessions) == 0:
            candidates = attacker_df[(attacker_df["claim_category"] == row["claim_category"]) & (attacker_df["session_id"] != sid)]
            candidate_sessions = candidates["session_id"].unique()
            scope = "same_category"

        control_scores = []

        for candidate_sid in candidate_sessions:
            candidate_rules = attacker_rules(candidate_sid, turn)
            sim = structural_similarity(target_window, candidate_rules)

            if not np.isnan(sim):
                control_scores.append(sim)

        if control_scores:
            baseline_similarity.append(np.mean(control_scores))
            baseline_n.append(len(control_scores))
            baseline_scope.append(scope)
        else:
            baseline_similarity.append(np.nan)
            baseline_n.append(0)
            baseline_scope.append("none")

    # 4. Results
    real_df["pair_structural_similarity"] = pair_similarity
    real_df["window_structural_similarity"] = window_similarity
    real_df["baseline_structural_similarity"] = baseline_similarity
    real_df["structural_entrainment_score"] = (real_df["window_structural_similarity"] - real_df["baseline_structural_similarity"])
    real_df["entrainment_baseline_n"] = baseline_n
    real_df["entrainment_baseline_scope"] = baseline_scope
    real_df["target_cfg_rules_n"] = target_rule_n
    real_df["attacker_cfg_rules_n"] = attacker_rule_n

    new_cols = [
        "pair_structural_similarity",
        "window_structural_similarity",
        "baseline_structural_similarity",
        "structural_entrainment_score",
        "entrainment_baseline_n",
        "entrainment_baseline_scope",
        "target_cfg_rules_n",
        "attacker_cfg_rules_n"
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

if __name__ == "__main__":
    main()
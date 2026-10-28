import os
from collections import Counter
import numpy as np
import pandas as pd
import stanza
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

class StructuralEntrainmentEvaluator:
    def __init__(self, window=1):
        self.window = window
        print("Loading Stanza constituency parser ...")

        self.nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,constituency", verbose=False)

        # Text -> CFG-rule Counter
        self.rule_cache = {}

        # Cached window-level rule distributions
        self.target_window_cache = {}
        self.attacker_window_cache = {}

        # Dataframes used internally after context preparation
        self.real_df = None
        self.attacker_df = None

    def _collect_cfg_rules(self, tree, counter):
        """
        Recursively extract CFG production rules from a
        Stanza constituency parse tree.
        """

        if not tree.children:
            return

        is_preterminal = (len(tree.children) == 1 and len(tree.children[0].children) == 0)

        if not is_preterminal:
            rhs = [child.label for child in tree.children]

            if rhs:
                rule = (f"{tree.label} -> {' '.join(rhs)}")
                counter[rule] += 1

        for child in tree.children:
            self._collect_cfg_rules(child, counter)

    def extract_cfg_rules(self, text):
        """
        Extract the CFG-rule frequency distribution
        associated with a text.
        """
        counter = Counter()

        if (pd.isna(text) or not str(text).strip()):
            return counter

        doc = self.nlp(str(text))

        for sentence in doc.sentences:
            self._collect_cfg_rules(sentence.constituency, counter)

        return counter

    @staticmethod
    def structural_similarity(counter_a, counter_b):
        """
        Jensen-Shannon similarity between two CFG-rule
        distributions.

        1.0 = identical syntactic distributions
        0.0 = maximally different distributions
        """

        if not counter_a or not counter_b:
            return np.nan

        rules = sorted(set(counter_a) | set(counter_b))

        p = np.array([counter_a.get(rule, 0) for rule in rules], dtype=float)
        q = np.array([counter_b.get(rule, 0) for rule in rules], dtype=float)

        if (p.sum() == 0 or q.sum() == 0):
            return np.nan

        p /= p.sum()
        q /= q.sum()

        distance = jensenshannon(p, q, base=2)

        return float(1.0 - distance)

    @staticmethod
    def merge_rule_counters(counters):
        """
        Merge multiple CFG-rule distributions.
        """
        result = Counter()

        for counter in counters:
            result.update(counter)

        return result

    @staticmethod
    def _validate_columns(trajectory_df, turns_df):
        required_trajectory_columns = {
            "session_id",
            "turn",
            "text",
            "claim",
            "claim_category",
        }

        required_turn_columns = {
            "session_id",
            "turn",
            "speaker",
            "claim",
            "claim_category",
            "text",
        }

        missing_trajectory = (required_trajectory_columns- set(trajectory_df.columns))
        missing_turns = ( required_turn_columns - set(turns_df.columns))

        if missing_trajectory:
            raise ValueError(f"Missing trajectory columns: {sorted(missing_trajectory)}")

        if missing_turns:
            raise ValueError(f"Missing turns columns: {sorted(missing_turns)}")

    def _prepare_context(self, trajectory_df, turns_df):
        """
        Prepare attacker messages and observed target-attacker
        pairs used by the structural entrainment evaluator.
        """
        self._validate_columns(trajectory_df, turns_df,)
        df = trajectory_df.copy()

        # Attacker messages
        self.attacker_df = (
            turns_df[turns_df["speaker"].eq("attacker")][["session_id", "turn", "claim", "claim_category","text"]]
            .copy()
            .reset_index(drop=True)
            .rename(columns={ "text": "attacker_text"})
        )

        # Pair Target_t with Attacker_t
        df = df.merge(
            self.attacker_df[["session_id", "turn", "attacker_text"]],
            on=["session_id", "turn"],
            how="left",
            validate="m:1",
        )

        # Exclude synthetic baseline points
        if "is_baseline" in df.columns:
            baseline_mask = (df["is_baseline"].fillna(False).astype(bool))
        else:
            baseline_mask = pd.Series(False, index=df.index)

        real_mask = (~baseline_mask & df["text"].notna() & df["attacker_text"].notna())
        self.real_df = (df.loc[real_mask].copy())

        return df

    # CFG CACHE
    def _build_rule_cache(self):
        all_texts = (pd.concat(
                [self.real_df["text"], self.attacker_df["attacker_text"]], ignore_index=True)
            .dropna()
            .astype(str)
            .unique()
        )

        self.rule_cache = {}

        for text in tqdm(all_texts, desc="Parsing constituency structures", unit="text"):
            self.rule_cache[text] = (self.extract_cfg_rules(text))

    # WINDOW DISTRIBUTIONS
    def _target_rules(self, session_id, turn):
        """
        Return the aggregated target CFG-rule distribution
        up to the current conversational turn.
        """
        cache_key = (session_id, turn)

        if cache_key in self.target_window_cache:
            return self.target_window_cache[cache_key]

        rows = (self.real_df[(self.real_df["session_id"] == session_id)
                            & (self.real_df["turn"] <= turn)].sort_values("turn"))

        if self.window is not None:
            rows = rows.tail(self.window)

        counters = [self.rule_cache[str(text)] for text in rows["text"] if str(text) in self.rule_cache]
        result = (self.merge_rule_counters(counters))
        self.target_window_cache[cache_key] = result

        return result

    def _attacker_rules(self, session_id, turn):
        cache_key = (session_id, turn)

        if cache_key in self.attacker_window_cache:
            return self.attacker_window_cache[cache_key]

        rows = (self.attacker_df[
                (self.attacker_df["session_id"]== session_id)
                & (self.attacker_df["turn" ] <= turn)].sort_values("turn"))

        if self.window is not None:
            rows = rows.tail(self.window)

        counters = [self.rule_cache[str(text)] for text in rows["attacker_text"] if str(text) in self.rule_cache]
        result = (self.merge_rule_counters(counters))
        self.attacker_window_cache[cache_key] = result

        return result

    # MATCHED BASELINE
    def _find_control_sessions(self, row):
        """
        Find matched attacker sessions using progressively
        broader controls:

        1. same claim + same turn
        2. same claim
        3. same claim category
        """
        sid = row["session_id"]
        turn = row["turn"]

        # Same claim + same conversational stage
        candidates = self.attacker_df[
            (self.attacker_df["claim"] == row["claim"])
            &
            (self.attacker_df["turn"] == turn)
            &
            (self.attacker_df["session_id"] != sid)
        ]

        candidate_sessions = (candidates["session_id"].unique())
        scope = ("same_claim_same_turn")

        # Fallback: same claim
        if len(candidate_sessions) < 2:
            candidates = self.attacker_df[(self.attacker_df["claim"] == row["claim"])
                                        & (self.attacker_df["session_id"] != sid)]
            candidate_sessions = (candidates["session_id"].unique())
            scope = "same_claim"

        # Fallback: same claim category
        if len(candidate_sessions) == 0:
            candidates = self.attacker_df[
                (self.attacker_df["claim_category"] == row["claim_category"])
                & (self.attacker_df["session_id"] != sid)]

            candidate_sessions = (candidates["session_id"].unique())
            scope = "same_category"

        return (candidate_sessions, scope)

    # SINGLE OBSERVATION
    def evaluate(self, row):
        sid = row["session_id"]
        turn = row["turn"]

        # Immediate Target_t vs Attacker_t similarity
        target_current = (self.rule_cache[str(row["text"])])
        attacker_current = (self.rule_cache[str(row["attacker_text"])])

        pair_similarity = (self.structural_similarity(target_current, attacker_current))

        # Windowed similarity
        target_window = (self._target_rules(sid, turn))
        attacker_window = (self._attacker_rules(sid, turn))

        window_similarity = (self.structural_similarity(target_window, attacker_window))

        # Matched baseline
        candidate_sessions, scope = (self._find_control_sessions(row))
        control_scores = []

        for candidate_sid in candidate_sessions:
            candidate_rules = (self._attacker_rules(candidate_sid, turn))
            similarity = (self.structural_similarity(target_window, candidate_rules))

            if not np.isnan(similarity):
                control_scores.append(similarity)

        if control_scores:
            baseline_similarity = float(
                np.mean(control_scores))
            baseline_n = len(control_scores)
            baseline_scope = scope
        else:
            baseline_similarity = np.nan
            baseline_n = 0
            baseline_scope = "none"

        # Baseline-adjusted structural entrainment
        if (np.isnan(window_similarity) or np.isnan( baseline_similarity)):
            structural_entrainment_score = np.nan
        else:
            structural_entrainment_score = (window_similarity - baseline_similarity)

        return {
            "pair_structural_similarity": pair_similarity,
            "window_structural_similarity": window_similarity,
            "baseline_structural_similarity": baseline_similarity,
            "structural_entrainment_score": structural_entrainment_score,
            "entrainment_baseline_n": baseline_n,
            "entrainment_baseline_scope": baseline_scope,
            "target_cfg_rules_n": sum(target_window.values()),
            "attacker_cfg_rules_n": sum(attacker_window.values())
        }

    # DATASET EVALUATION
    def evaluate_many(self, trajectory_df, turns_df):
        df = self._prepare_context(trajectory_df, turns_df)

        # Clear window caches in case evaluator is reused
        self.target_window_cache = {}
        self.attacker_window_cache = {}

        if len(self.real_df) == 0:
            print("WARNING: no valid target-attacker pairs found for structural entrainment.")

            return df

        # Parse each unique text once
        self._build_rule_cache()

        # Evaluate observations
        results = []
        result_indices = []

        for row_index, row in tqdm(self.real_df.iterrows(), total=len(self.real_df), desc="Computing structural entrainment", unit="response"):
            result = self.evaluate(row)
            results.append(result)
            result_indices.append(row_index)

        results_df = pd.DataFrame(results, index=result_indices)

        # Attach metrics to complete trajectory
        numeric_columns = [
            "pair_structural_similarity",
            "window_structural_similarity",
            "baseline_structural_similarity",
            "structural_entrainment_score",
            "entrainment_baseline_n",
            "target_cfg_rules_n",
            "attacker_cfg_rules_n",
        ]

        for column in numeric_columns:
            df[column] = np.nan

        df["entrainment_baseline_scope"] = pd.NA

        for column in results_df.columns:
            df.loc[results_df.index, column] = results_df[column]

        return df

# PIPELINE
def compute_structural_entrainment_trajectory(trajectory_df, turns_df, evaluator):
    return evaluator.evaluate_many(trajectory_df, turns_df)


# VALIDATION
def validate_structural_entrainment(df):
    pair_values = (df["pair_structural_similarity"].dropna().astype(float))
    window_values = (df["window_structural_similarity"].dropna().astype(float))
    baseline_values = (df["baseline_structural_similarity"].dropna().astype(float))

    entrainment_values = (df["structural_entrainment_score"].dropna().astype(float))

    print("\nStructural entrainment validation:")
    print("Valid pair similarities:", len(pair_values))
    print("Valid window similarities:", len(window_values))
    print("Valid matched baselines:", len(baseline_values))
    print("Valid entrainment scores:", len(entrainment_values))

    # Similarities must lie in [0, 1]
    for name, values in [("pair_structural_similarity", pair_values), ("window_structural_similarity", window_values), ("baseline_structural_similarity", baseline_values),]:
        outside = ((values < 0)| (values > 1)).sum()

        if outside > 0:
            raise ValueError(f"{name} contains values outside [0,1].")

    # Difference of two [0,1] similarities must lie in [-1,1]
    outside = ((entrainment_values < -1) | (entrainment_values > 1)).sum()

    if outside > 0:
        raise ValueError("structural_entrainment_score contains values outside [-1,1].")

    if len(entrainment_values) > 0:
        print("Mean entrainment:", entrainment_values.mean())
        print( "Median entrainment:", entrainment_values.median())
        print("Std entrainment:", entrainment_values.std())
        print("Min entrainment:", entrainment_values.min())
        print("Max entrainment:", entrainment_values.max())

# COMPLETE PIPELINE
def compute_structural_entrainment_scores(trajectory_file, trajectory_df, turns_df, window=1):
    print("Instantiating Structural Entrainment Evaluator ...")
    evaluator = (StructuralEntrainmentEvaluator(window=window))

    print("Computing baseline-adjusted structural entrainment trajectory ...")
    structural_df = (compute_structural_entrainment_trajectory(trajectory_df, turns_df, evaluator))
    validate_structural_entrainment(structural_df)
    structural_df.to_csv(trajectory_file, index=False)

    return structural_df

# PLOTING FUNCTION
def plot_structural_entrainment_trajectory(idx, trajectory_df, results_path: str):
    session_ts = (trajectory_df[trajectory_df['session_id'] == f'session_{idx}'].sort_values('t'))
    session_ts[['session_id', 't', 'structural_entrainment_score']]

    figure_path = os.path.join(results_path, 'structural_entrainment_example.png')

    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(data=session_ts, x='target_turn_index', y='structural_entrainment_score', marker='o', ax=ax)
    ax.axhline(0, linestyle='--', alpha=0.5)
    ax.set(xlabel='Target turn', ylabel='Structural entrainment', title='Structural entrainment example')

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    print("Plot saved successfully!")

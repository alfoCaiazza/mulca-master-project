import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Reproducibility
random.seed(42)

INITIAL_STANCE_MAP = {
    "disagree": -1.0,
    "neutral": 0.0,
    "agree": 1.0,
}

class StanceEvaluator:
    def __init__(self, model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(model_name)
            .to(self.device)
        )
        self.model.eval()
        self.label2id = {
            label.lower(): idx
            for idx, label in self.model.config.id2label.items()
        }

    def _compute_nli(self, text, premise):
        inputs = self.tokenizer(
            text, premise, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            
        probs = torch.softmax(logits, dim=-1)[0].cpu()

        p_entailment = probs[self.label2id["entailment"]].item()
        p_neutral = probs[self.label2id["neutral"]].item()
        p_contradiction = probs[self.label2id["contradiction"]].item()

        stance_score = p_entailment - p_contradiction

        return p_entailment, p_neutral, p_contradiction, stance_score

    def evaluate(self, text, claim):
        g_ent, g_neu, g_con, g_stance = self._compute_nli(text, claim)

        return {
            "p_entailment": g_ent,
            "p_neutral": g_neu,
            "p_contradiction": g_con,
            "stance_score": g_stance,
        }

def compute_stance_scores(target_df, evaluator):
    df = target_df.copy()
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing global stance alignment..."):
        result = evaluator.evaluate(
            text=row["text"], 
            claim=row["claim"]
        )
        results.append(result)

    scores_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, scores_df], axis=1)

def compute_alignment_trajectory(stance_df):
    df = stance_df.copy()

    df["initial_stance_score"] = (
        df["target_initial_stance"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(INITIAL_STANCE_MAP)
    )

    if df["initial_stance_score"].isna().any():
        unknown = df.loc[df["initial_stance_score"].isna(), "target_initial_stance"].unique()
        raise ValueError(f"Unknown initial stance values: {unknown}")

    # Global Alignment Metrics
    df["stance_shift"] = df["stance_score"] - df["initial_stance_score"]
    df["alignment_score"] = (df["stance_score"] + 1) / 2.0
    df["stance_displacement"] = df["alignment_score"].abs()

    return df

def add_baseline_points(alignment_df):
    baselines = alignment_df.groupby("conversation_id", as_index=False).first()

    baselines["t"] = 0.0
    baselines["target_turn_index"] = 0
    
    # Init Global Baseline (Stance iniziale S_0 = -1.0 -> Alignment A_0 = 0.0)
    baselines["stance_score"] = baselines["initial_stance_score"]
    baselines["stance_shift"] = 0.0
    baselines["alignment_score"] = (baselines["initial_stance_score"] + 1) / 2.0
    baselines["stance_displacement"] = baselines["alignment_score"].abs()
    baselines["p_entailment"] = pd.NA
    baselines["p_neutral"] = pd.NA
    baselines["p_contradiction"] = pd.NA
    baselines["text"] = pd.NA
    baselines["message_index"] = pd.NA
    baselines["message_id"] = pd.NA
    baselines["turn"] = pd.NA
    baselines["is_baseline"] = True

    alignment_df = alignment_df.copy()
    alignment_df["is_baseline"] = False

    trajectory_df = pd.concat([alignment_df, baselines], ignore_index=True)
    return trajectory_df.sort_values(["conversation_id", "t"]).reset_index(drop=True)

def compute_stance_trajectory(target_file, trajectory_file):
    print("Instantiating Stance Evaluator ...")
    evaluator = StanceEvaluator()

    print("Computing stance trajectory for target turns ...")
    target_df = pd.read_csv(target_file)
    stance_df = compute_stance_scores(target_df, evaluator)
    alignment_df = compute_alignment_trajectory(stance_df)
    trajectory_df = add_baseline_points(alignment_df)

    trajectory_df.to_csv(trajectory_file, index=False)
    print(f"Stance trajectory dataset saved to {trajectory_file}")
    return trajectory_df

def plot_stance_trajectory(conversation_id: str, trajectory_df: pd.DataFrame, results_path: str):
    session_ts = trajectory_df[trajectory_df['conversation_id'] == conversation_id].sort_values('t')

    if session_ts.empty:
        print(f"Session {conversation_id} not found.")
        return

    os.makedirs(results_path, exist_ok=True)
    figure_path = os.path.join(results_path, f'{conversation_id}_alignment_trajectory.png')

    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(data=session_ts, x='t', y='alignment_score', marker='o', ax=ax)
    ax.set_ylim(-0.05, 1.05)
    ax.set(
        xlabel='Normalized Time (t)',
        ylabel='Global Alignment Score (A_t)',
        title=f'Stance Alignment Trajectory - {conversation_id}'
    )

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved successfully to {figure_path}")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    target_file = os.path.join(base_dir, "data", "splits", "TARGET_TURNS.csv")
    trajectory_file = os.path.join(base_dir, "data", "trajectory", "TRAJECTORY.csv")
    results_dir = os.path.join(base_dir, "results")

    os.makedirs(results_dir, exist_ok=True)

    trajectory_df = compute_stance_trajectory(target_file, trajectory_file)
    trajectory_df.to_csv(trajectory_file, index=False)

if __name__ == "__main__":
    main()
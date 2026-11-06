import random
import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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
        inputs = self.tokenizer(text, premise, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            
        probs = torch.softmax(logits, dim=-1)[0].cpu()

        p_entailment = probs[self.label2id["entailment"]].item()
        p_neutral = probs[self.label2id["neutral"]].item()
        p_contradiction = probs[self.label2id["contradiction"]].item()

        stance_score = p_entailment - p_contradiction

        return p_entailment, p_neutral, p_contradiction, stance_score

    def evaluate(self, text, global_claim, local_claim):
        # Global Evaluation (Macro-Alignment vs Claim Iniziale)
        g_ent, g_neu, g_con, g_stance = self._compute_nli(text, global_claim)
        
        # Local Evaluation (Micro-Alignment vs Turno Precedente Attacker)
        if pd.notna(local_claim):
            l_ent, l_neu, l_con, l_stance = self._compute_nli(text, local_claim)
        else:
            l_ent, l_neu, l_con, l_stance = pd.NA, pd.NA, pd.NA, pd.NA

        return {
            "global_p_entailment": g_ent,
            "global_p_neutral": g_neu,
            "global_p_contradiction": g_con,
            "global_stance_score": g_stance,
            "local_p_entailment": l_ent,
            "local_p_neutral": l_neu,
            "local_p_contradiction": l_con,
            "local_stance_score": l_stance,
        }

def compute_stance_scores(target_df, evaluator):
    df = target_df.copy()
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing global and local stance..."):
        result = evaluator.evaluate(
            text=row["text"], 
            global_claim=row["global_claim"], 
            local_claim=row.get("local_claim", pd.NA)
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
    df["global_stance_shift"] = df["global_stance_score"] - df["initial_stance_score"]
    df["global_alignment_score"] = (df["global_stance_score"] + 1) / 2.0
    df["global_stance_displacement"] = df["global_alignment_score"].abs()

    # Local Alignment Metrics
    df["local_alignment_score"] = (df["local_stance_score"] + 1) / 2.0

    return df

def add_baseline_points(alignment_df):
    baselines = alignment_df.groupby("session_id", as_index=False).first()

    baselines["t"] = 0.0
    baselines["target_turn_index"] = 0
    
    # Init Global Baseline
    baselines["global_stance_score"] = baselines["initial_stance_score"]
    baselines["global_stance_shift"] = 0.0
    baselines["global_alignment_score"] = 0.0
    baselines["global_stance_displacement"] = 0.0
    
    # Init Local Baseline (Null a t=0 in assenza di turni precedenti)
    baselines["local_stance_score"] = pd.NA
    baselines["local_alignment_score"] = pd.NA

    baselines["text"] = pd.NA
    baselines["message_index"] = pd.NA
    baselines["message_id"] = pd.NA
    baselines["turn"] = pd.NA
    baselines["is_baseline"] = True

    alignment_df = alignment_df.copy()
    alignment_df["is_baseline"] = False

    trajectory_df = pd.concat([alignment_df, baselines], ignore_index=True)
    return trajectory_df.sort_values(["session_id", "t"]).reset_index(drop=True)

def compute_stance_trajectory(target_file, trajectory_file):
    print("Instantiating Stance Evaluator ...")
    evaluator = StanceEvaluator()

    print("Computing stance trajectory for each conversation ...")
    target_df = pd.read_csv(target_file)
    stance_df = compute_stance_scores(target_df, evaluator)
    alignment_df = compute_alignment_trajectory(stance_df)
    trajectory_df = add_baseline_points(alignment_df)

    trajectory_df.to_csv(trajectory_file, index=False)
    return trajectory_df

def plot_stance_trajectory(idx, trajectory_df, results_path: str):
    session_ts = (trajectory_df[trajectory_df['session_id'] == f'session_{idx}'].sort_values('t'))
    session_ts[['session_id', 't', 'stance_displacement']]

    figure_path = os.path.join(results_path, 'stance_displacement_example.png')

    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(data=session_ts, x='target_turn_index', y='stance_displacement', marker='o', ax=ax)
    ax.axhline(0, linestyle='--', alpha=0.5)
    ax.set(xlabel='Target turn', ylabel='Stance displacement', title='Stance displacement example')

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    print("Plot saved successfully!")

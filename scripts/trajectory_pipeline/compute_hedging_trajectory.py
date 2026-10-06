import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from compute_epistemic_trajectory import find_text_column
import seaborn as sns
import matplotlib.pyplot as plt 

MODEL_NAME = "ChrisLiewJY/BERTweet-Hedge"
TEXT_COLUMN_CANDIDATES = ["text", "target_text", "response_text",]
MAX_LENGTH = 128
BATCH_SIZE = 256

# LABEL_0 = no hedge - LABEL_1 = hedge
HEDGE_LABEL_ID = 1
HEDGE_THRESHOLD = 0.50

class HedgingEvaluator:
    def __init__(self, model_name=MODEL_NAME, max_length=MAX_LENGTH, batch_size=BATCH_SIZE, hedge_label_id=HEDGE_LABEL_ID, hedge_threshold=HEDGE_THRESHOLD):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.hedge_label_id = hedge_label_id
        self.hedge_threshold = hedge_threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading contextual hedge detector: {self.model_name}")
        print("Device:", self.device)

        self.tokenizer = (AutoTokenizer.from_pretrained(self.model_name))
        self.model = (AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device))
        self.model.eval()

    @staticmethod
    def split_sentences(text):
        if pd.isna(text):
            return []

        text = str(text).strip()

        if not text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)

        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def predict_probabilities(self, sentences):
        """
        Return P(hedge | sentence) for each sentence, corresponding to LABEL_1.
        """
        if not sentences:
            return np.asarray([], dtype=float)

        probabilities = []
        batch_sarts = range(0, len(sentences), self.batch_size)

        with torch.no_grad():
            for start in tqdm(batch_sarts, total = len(range(0, len(sentences), self.batch_size)), desc = "Hedging detection", unit = "batch"):
                batch = sentences[start: start + self.batch_size]

                encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",)

                encoded = {
                    key: value.to(self.device)
                    for key, value
                    in encoded.items()
                }

                outputs = self.model(**encoded)
                probs = torch.softmax(outputs.logits, dim=-1,)
                hedge_probs = (probs[:, self.hedge_label_id].detach().cpu().numpy())
                probabilities.extend(hedge_probs.tolist())

        return np.asarray(probabilities, dtype=float)

    def stats_to_result(self, stats):
        """
        Convert sentence-level hedge probabilities into
        response-level hedging metrics.
        
        H_t = mean_s P(hedge | sentence_s)

        Every sentence contributes equally, independently
        of its token length.
        """
        probabilities = np.asarray(stats["probabilities"], dtype=float)
        sentence_count = len(probabilities)

        if sentence_count == 0:
            return {
                "hedge_count": np.nan,
                "hedging_score": np.nan,
                "hedging_sentence_count": np.nan,
                "hedging_positive_sentence_count": np.nan,
                "hedging_positive_sentence_rate": np.nan,
                "hedging_max_probability": np.nan,
            }

        positive_count = int((probabilities >= self.hedge_threshold).sum())
        hedging_score = float(np.clip(probabilities.mean(), 0.0, 1.0))

        return {
            "hedging_score": hedging_score,
            "hedge_count": positive_count,
            "hedging_sentence_count": sentence_count,
            "hedging_positive_sentence_count": positive_count,
            "hedging_positive_sentence_rate": (positive_count  / sentence_count),
            "hedging_max_probability":  float(probabilities.max())
        }

    def evaluate(self, text):
        """
        Evaluate hedging  H_t = mean_s P(hedge | sentence_s)
        """
        sentences = self.split_sentences(text)

        if not sentences:
            return self.stats_to_result({
                "probabilities" : []
            })

        probabilities = (self.predict_probabilities(sentences))

        if (len(probabilities) != len(sentences)):
            raise RuntimeError("Hedge detector output size does not match sentence count.")

        stats = {
            "probabilities" : probabilities,
        }

        return self.stats_to_result(stats)

    def evaluate_many(self, texts):
        sentences = []
        owners = []

        for row_position, text in enumerate(texts):
            row_sentences = (self.split_sentences(text))

            for sentence in row_sentences:
                sentences.append(sentence)
                owners.append(row_position)

        row_probabilities = {
            row_position: [] for row_position in range(len(texts))
        }

        if not sentences:
            return [self.stats_to_result({"probabilities": []}) for _ in range(len(texts))]

        print("Sentences to analyse:", len(sentences))

        hedge_probabilities = (self.predict_probabilities(sentences))

        if (len(hedge_probabilities) != len(sentences)):
            raise RuntimeError("Hedge detector output size does not match sentence count.")

        for row_position, probability in zip(owners, hedge_probabilities):
            row_probabilities[row_position].append(float(probability))
        
        results = []

        for row_position in range(len(texts)):
            result = self.stats_to_result({
                "probabilities" : row_probabilities[row_position]
            })
            results.append(result)

        return results

def compute_hedging_trajectory(trajectory_df, evaluator):
    df = trajectory_df.copy()
    text_column = find_text_column(df)

    metric_columns = [
        "hedge_count",
        "hedging_score",
        "hedging_sentence_count",
        "hedging_positive_sentence_count",
        "hedging_positive_sentence_rate",
        "hedging_max_probability",
    ]

    for column in metric_columns:
        df[column] = np.nan

    if "is_baseline" in df.columns:
        valid_mask = (~df["is_baseline"].fillna(False).astype(bool))
    else:
        valid_mask = pd.Series(True, index=df.index)

    valid_indices = df.index[valid_mask]

    texts = (df.loc[valid_indices, text_column,].tolist())

    if len(texts) == 0:
        print("WARNING: no valid target text found for hedge detection.")

        return df

    results = (evaluator.evaluate_many(texts))

    if (len(results) != len(valid_indices)):

        raise RuntimeError("Evaluator output size does not match number of dataframe rows.")

    results_df = pd.DataFrame(results, index=valid_indices)

    for column in metric_columns:
        df.loc[valid_indices, column] = results_df[column]

    return df

def validate_hedging_scores(df):
    values = (df["hedging_score"].dropna().astype(float))
    outside = ((values < 0) | (values > 1)).sum()

    print("\nHedging validation:")
    print("Valid rows:", len(values))
    print("Outside [0,1]:", outside)

    if outside > 0:
        raise ValueError("hedging_score contains values outside [0,1].")

    if len(values) == 0:
        return

    print("Mean:", values.mean())
    print("Median:", values.median())
    print("Std:", values.std())
    print("Min:", values.min())
    print("Max:", values.max())
    print("Responses with contextual hedge:", int((df["hedging_positive_sentence_count"] > 0).sum()))
    print("Mean positive-sentence rate:", df["hedging_positive_sentence_rate"].mean())

def compute_hedging_scores(trajectory_file, trajectory_df):
    print("Instantiating Hedging Evaluator ...")
    evaluator = HedgingEvaluator()

    print("Computing contextual hedging trajectory ...")
    hedging_df = (compute_hedging_trajectory(trajectory_df, evaluator))
    validate_hedging_scores(hedging_df)
    hedging_df.to_csv(trajectory_file, index=False)

    return hedging_df

def plot_hedging_trajectory(idx, trajectory_df, results_path: str):
    session_ts = (trajectory_df[trajectory_df['session_id'] == f'session_{idx}'].sort_values('t'))
    session_ts[['session_id', 't', 'hedging_score']]

    figure_path = os.path.join(results_path, 'hedging_score_example.png')

    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(data=session_ts, x='target_turn_index', y='hedging_score', marker='o', ax=ax)
    ax.axhline(0, linestyle='--', alpha=0.5)
    ax.set(xlabel='Target turn', ylabel='Hedging score', title='Hedging score example')

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    print("Plot saved successfully!")
import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

MODEL_NAME = "ChrisLiewJY/BERTweet-Hedge"
TEXT_COLUMN_CANDIDATES = ["text", "target_text", "response_text"]
MAX_LENGTH = 128
BATCH_SIZE = 32

# LABEL_0 = no hedge - LABEL_1 = hedge
HEDGE_LABEL_ID = 1

# Diagnostic threshold only.
HEDGE_THRESHOLD = 0.50

def find_text_column(df):
    for column in TEXT_COLUMN_CANDIDATES:
        if column in df.columns:
            return column

    raise ValueError(f"No target-response text column found. Expected one of: {TEXT_COLUMN_CANDIDATES}")

def split_sentences(text):
    if pd.isna(text):
        return []

    text = str(text).strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text,)

    return [sentence.strip() for sentence in sentences if sentence.strip()]

def load_hedge_detector():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading contextual hedge detector: {MODEL_NAME}")
    print("Device:", device)

    tokenizer = (AutoTokenizer.from_pretrained( MODEL_NAME))
    model = (AutoModelForSequenceClassification.from_pretrained(MODEL_NAME))

    model.to(device)
    model.eval()

    return (tokenizer, model, device)

def predict_hedging_probabilities(sentences, tokenizer, model, device):
    """
    Return P(hedge | sentence) for each sentence.

    The score is based on LABEL_1 probability
    """
    probabilities = []

    with torch.no_grad():
        for start in range(0, len(sentences), BATCH_SIZE,):
            batch = sentences[start:start + BATCH_SIZE]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(device)
                for key, value
                in encoded.items()
            }

            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)
            hedge_probs = (probs[:, HEDGE_LABEL_ID].detach().cpu().numpy())

            probabilities.extend(hedge_probs.tolist())

    return np.asarray(probabilities, dtype=float)

# HEDGING TRAJECTORY
def compute_hedging_trajectory(trajectory_df, tokenizer, model, device):
    """
    H_t = sum_s N_s * P(hedge | sentence_s) / sum_s N_s

    where N_s is the number of model tokens in sentence s.
    """
    df = trajectory_df.copy()
    text_column = find_text_column(df)

    columns = [
        "hedge_count",
        "hedging_score",
        "hedging_sentence_count",
        "hedging_positive_sentence_count",
        "hedging_positive_sentence_rate",
        "hedging_max_probability",
    ]

    for column in columns:
        df[column] = np.nan


    sentences = []
    owners = []

    for row_index, row in tqdm(df.iterrows(), total=len(df), desc="Computing hedging trajectories on dataframe ..."):
        if ("is_baseline" in df.columns and bool(row.get("is_baseline", False))):
            continue

        text = row[text_column]
        row_sentences = (split_sentences(text))

        for sentence in row_sentences:
            sentences.append(sentence)
            owners.append(row_index)

    if len(sentences) == 0:
        print("WARNING: no valid target text found for hedge detection.")

        return df

    print("Sentences to analyse:", len(sentences))

    # CONTEXTUAL MODEL INFERENCE
    hedge_probabilities = (predict_hedging_probabilities(sentences, tokenizer, model, device))

    if (len(hedge_probabilities) != len(sentences)):
        raise RuntimeError("Hedge detector output size does not match sentence count.")

    # AGGREGATE BACK TO RESPONSES
    row_stats = {}

    for (row_index, probability) in zip(owners, hedge_probabilities):
        if row_index not in row_stats:
            row_stats[row_index] = {
                "probability_sum": 0.0,
                "sentence_count": 0,
                "positive_sentence_count": 0,
                "max_probability": 0.0,
            }

        stats = row_stats[row_index]
        stats["probability_sum"] += probability
        stats["sentence_count"] += 1

        if (probability >= HEDGE_THRESHOLD):
            stats["positive_sentence_count"] += 1

        stats["max_probability"] = max(stats["max_probability"], probability)

    # RESPONSE-LEVEL HEDGING SCORE
    for row_index, stats in (row_stats.items()):
        sentence_count = stats["sentence_count"]

        if (sentence_count <= 0):
            continue

        hedging_score = (stats["probability_sum"] / sentence_count)
        positive_count = stats["positive_sentence_count"]

        # PRIMARY TRAJECTORY
        df.at[ row_index, "hedging_score"] = float(np.clip(hedging_score, 0.0, 1.0))

        # COMPATIBILITY / DIAGNOSTICS
        df.at[row_index, "hedge_count"] = positive_count
        df.at[row_index, "hedging_sentence_count"] = sentence_count
        df.at[row_index, "hedging_positive_sentence_count"] = positive_count
        df.at[row_index, "hedging_positive_sentence_rate"] = (positive_count / sentence_count)
        df.at[row_index, "hedging_max_probability"] = stats["max_probability"]

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


# ======================================================================================================================
def main():
    main_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))
    trajectory_path = os.path.join(main_dir, "TRAJECTORY.csv")
    trajectory_df = pd.read_csv(trajectory_path)

    print("Computing contextual hedging trajectory ...")
    tokenizer, model, device = load_hedge_detector()
    trajectory_df = (compute_hedging_trajectory(trajectory_df, tokenizer,model,device))
    validate_hedging_scores(trajectory_df)
    trajectory_df.to_csv(trajectory_path, index=False)
    print("\nOperation completed successfully!")


if __name__ == "__main__":
    main()
import os
import re
import numpy as np
import pandas as pd
import torch
from simpletransformers.ner import NERModel
from tqdm import tqdm

MODEL_NAME = "jeniakim/hedgehog"

# HEDGEhog labels:
# C = Certain / no uncertainty cue
# D = Doxastic
# E = Epistemic
# I = Investigation
# N = Condition
#
HEDGEHOG_LABELS = [
    "C",
    "D",
    "E",
    "I",
    "N",
]

EPISTEMIC_LABEL = "E"

TEXT_COLUMN_CANDIDATES = ["text", "target_text", "response_text",]
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 32

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

def load_epistemic_detector():
    use_cuda = torch.cuda.is_available()

    print(f"Loading epistemic uncertainty detector: {MODEL_NAME}")

    print("Device:", "CUDA" if use_cuda else "CPU")

    model_args = {
        "silent": True,
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "use_multiprocessing": False,
        "use_multiprocessing_for_evaluation": False,
        "eval_batch_size": BATCH_SIZE,
        "max_seq_length": MAX_SEQ_LENGTH,
    }

    model = NERModel(
        "bert",
        MODEL_NAME,
        labels=HEDGEHOG_LABELS,
        use_cuda=use_cuda,
        args=model_args,
    )

    return model

def count_prediction_labels(prediction):
    counts = {
        label: 0 for label in HEDGEHOG_LABELS
    }

    total = 0

    for token_result in prediction:
        if not isinstance(token_result, dict):
            continue

        for _, label in token_result.items():
            total += 1
            if label in counts:
                counts[label] += 1

    return total, counts

def compute_epistemic_trajectory(trajectory_df, detector):
    """
    E_t = # classified epistemic uncertainty tokens / # classified tokens 

    Only the HEDGEhog E label contributes to epistemic_uncertainty.
    """
    df = trajectory_df.copy()
    text_column = find_text_column(df)

    columns = [
        "epistemic_uncertainty",
        "epistemic_cue_count",
        "epistemic_token_count",
        "epistemic_cue_density",
        "epistemic_presence",
        "doxastic_cue_count",
        "investigation_cue_count",
        "condition_cue_count",
        "all_uncertainty_cue_count",
        "all_uncertainty_cue_density",
    ]

    for column in columns:
        df[column] = np.nan

    sentences = []
    owners = []

    for row_index, row in tqdm(df.iterrows(), total=len(df), desc="Computing epistemic uncertainty on dataframe ..."):
        # Synthetic baseline t=0 has no observed target response
        if ("is_baseline" in df.columns and bool(row.get("is_baseline", False))):
            continue

        text = row[text_column]

        row_sentences = split_sentences(text)

        for sentence in row_sentences:
            sentences.append(sentence)
            owners.append(row_index)

    if len(sentences) == 0:
        print("WARNING: no valid target text found for epistemic detection.")

        return df

    print("Sentences to analyse:", len(sentences))

    predictions, _ = detector.predict(sentences)

    if len(predictions) != len(sentences):
        raise RuntimeError("Detector output size does not match number of input sentences.")

    row_stats = {}

    for row_index, prediction in zip(owners, predictions):
        if row_index not in row_stats:

            row_stats[row_index] = {
                "total": 0,
                "C": 0,
                "D": 0,
                "E": 0,
                "I": 0,
                "N": 0,
            }

        total, counts = (count_prediction_labels(prediction))

        row_stats[row_index]["total"] += total

        for label in HEDGEHOG_LABELS:
            row_stats[row_index][label] += counts[label]

    for row_index, stats in row_stats.items():
        total = stats["total"]

        if total <= 0:
            continue

        epistemic_count = stats[EPISTEMIC_LABEL]

        # PRIMARY EPISTEMIC TRAJECTORY
        epistemic_uncertainty = (epistemic_count / total)
        df.at[row_index,"epistemic_uncertainty"] = epistemic_uncertainty

        # Detector diagnostics
        df.at[row_index, "epistemic_cue_count"] = epistemic_count
        df.at[row_index, "epistemic_token_count"] = total
        df.at[row_index,"epistemic_cue_density"] = epistemic_uncertainty
        df.at[row_index, "epistemic_presence"] = float(epistemic_count > 0)
        df.at[row_index, "doxastic_cue_count"] = stats["D"]
        df.at[row_index, "investigation_cue_count"] = stats["I"]
        df.at[row_index, "condition_cue_count"] = stats["N"]

        all_uncertainty_count = (stats["D"] + stats["E"] + stats["I"] + stats["N"])
        df.at[row_index, "all_uncertainty_cue_count"] = all_uncertainty_count

        df.at[row_index, "all_uncertainty_cue_density"] = (all_uncertainty_count / total)

    return df

def validate_epistemic_scores(df):
    values = (df["epistemic_uncertainty"].dropna().astype(float))
    outside = ((values < 0) | (values > 1)).sum()

    print( "\nEpistemic uncertainty validation:")
    print("Valid rows:", len(values))
    print("Outside [0,1]:", outside)

    if outside > 0:
        raise ValueError("epistemic_uncertainty contains values outside [0,1].")

    if len(values) == 0:
        return

    print("Mean:", values.mean())
    print("Median:", values.median())
    print("Std:", values.std())
    print("Min:", values.min())
    print("Max:", values.max())
    print("Zero-rate:",float(np.mean(values == 0.0)))

    print("Responses with epistemic cue:", int((df["epistemic_presence"] == 1).sum()))


# ======================================================================================================================
def main():
    main_dir = os.path.abspath( os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))
    trajectory_path = os.path.join(main_dir, "TRAJECTORY.csv")
    trajectory_df = pd.read_csv(trajectory_path)

    print("Computing detector-based epistemic uncertainty trajectory ...")
    detector = (load_epistemic_detector())
    epistemic_df = (compute_epistemic_trajectory(trajectory_df, detector))

    validate_epistemic_scores(epistemic_df)

    epistemic_df.to_csv(trajectory_path, index=False)

    print("\nOperation completed successfully!")


if __name__ == "__main__":
    main()
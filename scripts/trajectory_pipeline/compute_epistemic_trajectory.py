import os
import re
import numpy as np
import pandas as pd
import torch
from simpletransformers.ner import NERModel
from tqdm import tqdm
import random
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_NAME = "jeniakim/hedgehog"
HEDGEHOG_LABELS = [
    "C", # C = Certain / no uncertainty cue
    "D", # D = Doxastic
    "E", # E = Epistemic
    "I", # I = Investigation
    "N", # N = Condition
]
EPISTEMIC_LABEL = "E"
TEXT_COLUMN_CANDIDATES = ["text", "target_text", "response_text"]
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 256

class EpistemicEvaluator:
    def __init__(self, model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, batch_size=BATCH_SIZE):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()

        print(f"Loading epistemic uncertainty detector: {self.model_name}")
        print("Device:", "CUDA" if self.use_cuda else "CPU")

        model_args = {
            "silent": False,
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "use_multiprocessing": False,
            "use_multiprocessing_for_evaluation": False,
            "eval_batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
        }

        self.model = NERModel(
            "bert",
            self.model_name,
            labels=HEDGEHOG_LABELS,
            use_cuda=self.use_cuda,
            args=model_args,
        )

    @staticmethod
    def split_sentences(text):
        if pd.isna(text):
            return []

        text = str(text).strip()

        if not text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text,)

        return [sentence.strip() for sentence in sentences if sentence.strip()]

    @staticmethod
    def count_prediction_labels(prediction):
        counts = {
            label: 0
            for label in HEDGEHOG_LABELS
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

    @staticmethod
    def stats_to_result(stats):
        total = stats["total"]

        if total <= 0:
            return {
                "epistemic_uncertainty": np.nan,
                "epistemic_cue_count": np.nan,
                "epistemic_token_count": np.nan,
                "epistemic_cue_density": np.nan,
                "epistemic_presence": np.nan,
                "doxastic_cue_count": np.nan,
                "investigation_cue_count": np.nan,
                "condition_cue_count": np.nan,
                "all_uncertainty_cue_count": np.nan,
                "all_uncertainty_cue_density": np.nan,
            }

        epistemic_count = stats[EPISTEMIC_LABEL]
        epistemic_uncertainty = (epistemic_count / total)
        all_uncertainty_count = (stats["D"] + stats["E"] + stats["I"] + stats["N"])

        return {
            "epistemic_uncertainty": epistemic_uncertainty,
            "epistemic_cue_count": epistemic_count,
            "epistemic_token_count": total,
            "epistemic_cue_density": epistemic_uncertainty,
            "epistemic_presence": float(epistemic_count > 0),
            "doxastic_cue_count": stats["D"],
            "investigation_cue_count": stats["I"],
            "condition_cue_count": stats["N"],
            "all_uncertainty_cue_count": all_uncertainty_count,
            "all_uncertainty_cue_density": all_uncertainty_count / total,
        }

    def evaluate(self, text):
        """
        Evaluate  E_t = # epistemic tokens / # classified tokens
        """
        sentences = self.split_sentences(text)

        if not sentences:
            return self.stats_to_result({
                "total": 0,
                "C": 0,
                "D": 0,
                "E": 0,
                "I": 0,
                "N": 0,
            })

        predictions, _ = self.model.predict(sentences)

        stats = {
            "total": 0,
            "C": 0,
            "D": 0,
            "E": 0,
            "I": 0,
            "N": 0,
        }

        for prediction in predictions:
            total, counts = self.count_prediction_labels(prediction)
            stats["total"] += total

            for label in HEDGEHOG_LABELS:
                stats[label] += counts[label]

        return self.stats_to_result(stats)

    def evaluate_many(self, texts):
        """"
        Sentences from all responses are flattened and sent together to
        HEDGEhog. Predictions are then aggregated back at response level.
        """
        sentences = []
        owners = []

        for row_position, text in enumerate(texts):
            row_sentences = self.split_sentences(text)

            for sentence in row_sentences:
                sentences.append(sentence)
                owners.append(row_position)

        row_stats = {
            row_position: {
                "total": 0,
                "C": 0,
                "D": 0,
                "E": 0,
                "I": 0,
                "N": 0,
            }
            for row_position in range(len(texts))
        }

        if not sentences:
            return [self.stats_to_result(row_stats[row_position]) for row_position in range(len(texts))]

        print("Sentences to analyse:", len(sentences))
        
        predictions, _ = self.model.predict(sentences)

        if len(predictions) != len(sentences):
            raise RuntimeError("Detector output size does not match number of input sentences.")

        for row_position, prediction in zip(owners, predictions):
            total, counts = (self.count_prediction_labels(prediction))
            row_stats[row_position]["total"] += total

            for label in HEDGEHOG_LABELS:
                row_stats[row_position][label] += (counts[label])

        results = []

        for row_position in range(len(texts)):
            result = self.stats_to_result(row_stats[row_position])
            results.append(result)

        return results

def find_text_column(df):
    for column in TEXT_COLUMN_CANDIDATES:
        if column in df.columns:
            return column

    raise ValueError(f"No target-response text column found.Expected one of: {TEXT_COLUMN_CANDIDATES}")

def compute_epistemic_trajectory(trajectory_df, evaluator):
    df = trajectory_df.copy()
    text_column = find_text_column(df)

    metric_columns = [
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

    for column in metric_columns:
        df[column] = np.nan

    if "is_baseline" in df.columns:
        valid_mask = (~df["is_baseline"].fillna(False).astype(bool))
    else:
        valid_mask = pd.Series(True, index=df.index)

    valid_indices = df.index[valid_mask]
    texts = (df.loc[valid_indices, text_column].tolist())

    if len(texts) == 0:
        print("WARNING: no valid target text found for epistemic detection.")
        return df

    results = evaluator.evaluate_many(texts)

    if len(results) != len(valid_indices):
        raise RuntimeError("Evaluator output size does not match number of dataframe rows.")

    results_df = pd.DataFrame(results, index=valid_indices)

    for column in metric_columns:
        df.loc[valid_indices, column] = results_df[column]

    return df

def validate_epistemic_scores(df):
    values = (df["epistemic_uncertainty"].dropna().astype(float))
    outside = ((values < 0) | (values > 1)).sum()

    print("\nEpistemic uncertainty validation:")
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
    print("Zero-rate:", float(np.mean(values == 0.0)))
    print("Responses with epistemic cue:", int((df["epistemic_presence"] == 1).sum()))

def compute_epistemic_scores(trajectory_file, trajectory_df):
    print("Instantiating Epistemic Evaluator ...")
    evaluator = EpistemicEvaluator()

    print("Computing epistemic uncertainty trajectory ...")
    epistemic_df = compute_epistemic_trajectory(trajectory_df, evaluator)
    validate_epistemic_scores(epistemic_df)
    epistemic_df.to_csv(trajectory_file, index=False)

    return epistemic_df

def plot_epistemic_trajectory(idx, trajectory_df, results_path: str):
    session_ts = (trajectory_df[trajectory_df['session_id'] == f'session_{idx}'].sort_values('t'))
    session_ts[['session_id', 't', 'epistemic_uncertainty']]

    figure_path = os.path.join(results_path, 'epistemic_uncertainty_example.png')

    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(data=session_ts, x='target_turn_index', y='epistemic_uncertainty', marker='o', ax=ax)
    ax.axhline(0, linestyle='--', alpha=0.5)
    ax.set(xlabel='Target turn', ylabel='Epistemic uncertainty', title='Epistemic uncertainty example')

    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    print("Plot saved successfully!")
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import pandas as pd
import os

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

    def evaluate(self, text, claim):
        inputs = self.tokenizer(text, claim, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax( logits, dim=-1)[0].cpu()

        entailment_idx = self.label2id["entailment"]
        neutral_idx = self.label2id["neutral"]
        contradiction_idx = self.label2id["contradiction"]

        p_entailment = probs[entailment_idx].item()
        p_neutral = probs[neutral_idx].item()
        p_contradiction = probs[contradiction_idx].item()

        stance_score = (p_entailment - p_contradiction)

        return {
            "p_entailment": p_entailment,
            "p_neutral": p_neutral,
            "p_contradiction": p_contradiction,
            "stance_score": stance_score,
        }

def compute_stance_scores(target_df, evaluator):
    df = target_df.copy()

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing stance scores ..."):
        result = evaluator.evaluate(text=row["text"], claim=row["claim"])
        results.append(result)

    scores_df = pd.DataFrame(results, index=df.index)

    return pd.concat( [df, scores_df], axis=1)

def main():
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "TARGET_TURNS.csv"))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "results"))

    os.makedirs(output_dir, exist_ok=True)

    stance_output = os.path.join( output_dir, "STANCE.csv")

    target_df = pd.read_csv(input_file)

    print("Instantiating Stance Evaluator ...")
    evaluator = StanceEvaluator()
    print("Operation completed successfully!")

    stance_df = compute_stance_scores(target_df, evaluator)

    stance_df.to_csv(stance_output, index=False)

if __name__ == "__main__":
    main()
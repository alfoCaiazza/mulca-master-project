import os
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

def filter_conversation(sample, min_turns:int) -> bool:
    # 1. English language filter
    language = sample.get("language")
    if language and language.lower() not in ("english", "en"):
        return False

    # 3. Minimum turns filter
    turns = sample.get("conversation", [])
    if len(turns) < min_turns:
        return False

    return True

def inject_dataset(name: str, min_turns: int, dest_name: str) -> pd.DataFrame:
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", dest_name))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Loading dataset from Hugging Face
    print(f"Loading {name} dataset from Hugging Face...")
    dataset = load_dataset(name, streaming=True)

    # Apply filters to the dataset
    filtered_stream = dataset["train"].filter(filter_conversation, fn_kwargs={"min_turns": min_turns})

    # # Save the filtered dataset to a CSV file
    # collected_rows = []
    # max_samples = 10000

    # for row in tqdm(filtered_stream, desc="Processing filtered conversations"):
    #     collected_rows.append(row)
    #     if len(collected_rows) >= max_samples:
    #         break

    df = pd.DataFrame(filtered_stream)
    df.to_csv(output_path, index=False)

    print(f"Filtered CSV dataset created with {len(df)} conforming conversations.\nOutput folder: {output_path}.")

if __name__ == "__main__":
    dataset_name = "allenai/WildChat"
    min_turns = 3
    dest_name = "filtered_wildchat.csv"

    inject_dataset(dataset_name, min_turns, dest_name=dest_name)
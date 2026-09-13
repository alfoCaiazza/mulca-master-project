import os
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

def inject_dataset(name: str, dest_name: str) -> pd.DataFrame:
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw",dest_name))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Loading dataset from Hugging Face
    print(f"Loading {name} dataset from Hugging Face...")
    dataset = load_dataset(name, streaming=True)

    df = pd.DataFrame(dataset)
    df.to_csv(output_path, index=False)

    print(f"Filtered CSV dataset created with {len(df)} conforming conversations.\nOutput folder: {output_path}.")

if __name__ == "__main__":
    dataset_name = "lmsys/lmsys-chat-1m"
    dest_name = "RAW.csv"

    inject_dataset(dataset_name, dest_name=dest_name)
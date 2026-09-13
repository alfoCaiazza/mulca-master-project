import os
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

def inject_dataset(name: str, dest_name: str, batch_size: int) -> pd.DataFrame:
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw",dest_name))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        os.remove(output_path)

    # Loading dataset from Hugging Face
    print(f"Loading {name} dataset from Hugging Face...")
    dataset = load_dataset(name, streaming=True, split="train")

    batch = []
    total_rows = 0

    with tqdm(total=1000000, desc=f"Saving {name} into CSV file ...") as pbar:
        for row in dataset:
            batch.append(row)

            if len(batch) >= batch_size:
                df = pd.DataFrame(batch)
                write_header = not os.path.exists(output_path)
                df.to_csv(output_path, mode='a', index=False, header=write_header)
                
                total_rows += len(batch)
                pbar.update(len(batch))
                batch.clear()

        if batch:
            df = pd.DataFrame(batch)
            write_header = not os.path.exists(output_path)
            df.to_csv(output_path, mode='a', index=False, header=write_header)
            total_rows += len(batch)
            pbar.update(len(batch))

    print(f"Operation completed successfully!\nCSV {name} dataset created.\nOutput folder: {output_path}.")

if __name__ == "__main__":
    dataset_name = "lmsys/lmsys-chat-1m"
    dest_name = "RAW.csv"

    inject_dataset(dataset_name, dest_name=dest_name, batch_size=50000)
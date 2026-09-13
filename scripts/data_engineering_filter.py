import os
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import ast

# (1) HARDCODED FILTER :
# Filtering criteria : LLM used, number of turns and average token length of user turns
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MODELS = ["llama-2-7b-chat", "llama-2-13b-chat", "vicuna-7b", "vicuna-13b", "wizardlm-13b", "alpaca-13b"]

def get_token_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(tokenizer.encode(text, disallowed_special=()))

def get_avg_user_token_count(conversation: list) -> float:
    user_turns = [turn for turn in conversation if turn.get("role") == "user"]
    if not user_turns:
        return 0.0
    total_tokens = sum(get_token_count(turn.get("content", "")) for turn in user_turns)
    return total_tokens / len(user_turns)

def filter_conversation(row: dict, min_turns: int, min_avg_tokens: float) -> bool:
    # 1. English language filter
    language = row.get("language")
    if not language or language.lower() not in ("english", "en"):
        return False

    # 2. Select benchmark models
    if row.get("model") not in MODELS:
        return False

    # 3. Minimum turns filter
    conv_data = row.get("conversation", [])
    if isinstance(conv_data, str):
        try:
            conv_data = ast.literal_eval(conv_data)
        except (ValueError, SyntaxError):
            return False

    user_turns_count = sum(1 for turn in conv_data if turn.get("role") == "user")
    if user_turns_count < min_turns:
        return False

    # 4. Minimum average token length filter
    if get_avg_user_token_count(conv_data) < min_avg_tokens:
        return False

    return True


def data_engineering_pipeline(input_csv: str, output_csv: str, min_turns: int, min_avg_tokens: float, batch_size: int = 10000): 
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if os.path.exists(output_csv):
        os.remove(output_csv)

    print("Starting data engineering pipeline...")
    estimated_total_rows = 1000000 
    total_retained_rows = 0
    
    chunk_iterator = pd.read_csv(input_csv, chunksize=batch_size)

    with tqdm(total=estimated_total_rows, desc="Hardcoded Filtering ...", unit=" righe") as pbar:
        for chunk in chunk_iterator:
            mask = chunk.apply(lambda row: filter_conversation(row.to_dict(), min_turns, min_avg_tokens), axis=1)
            filtered_chunk = chunk[mask]

            if not filtered_chunk.empty:
                write_header = not os.path.exists(output_csv)
                filtered_chunk.to_csv(output_csv, mode='a', index=False, header=write_header)
                total_retained_rows += len(filtered_chunk)
            
            pbar.update(len(chunk))

    return total_retained_rows

if __name__ == "__main__":
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "RAW.csv"))
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "FILTERED.csv"))

    retained_rows = data_engineering_pipeline(
        input_csv=input_file,
        output_csv=output_file,
        min_turns=5,
        min_avg_tokens=15,
        batch_size=10000
    )

    print(f"\nFiltering operation completed. Total rows after filtering: {retained_rows}")
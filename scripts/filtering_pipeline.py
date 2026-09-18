import os
import re
import ast
import html
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# ----------------------------
# Utilities
# ----------------------------

def normalize_message(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Decode HTML entities (&amp;, &lt;, ...)
    text = html.unescape(text)

    # 1. Replace fenced code blocks
    text = re.sub(
        r"```[\s\S]*?```",
        " [CODE_BLOCK] ",
        text,
        flags=re.MULTILINE
    )

    # Replace inline code
    text = re.sub(r"`[^`]+`", " [CODE_BLOCK] ", text)

    # 2. Replace URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " [URL] ",
        text
    )

    # 3. Remove Markdown links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Images: ![alt](url) -> alt
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

    # Remove markdown emphasis (*, _, **, __, ~~)
    text = re.sub(r"(\*\*|\*|__|_|~~)", "", text)

    # Remove headings (#, ##, ...)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Remove blockquotes
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)

    # Remove unordered list markers
    text = re.sub(r"^\s*[-+*]\s+", "", text, flags=re.MULTILINE)

    # Remove ordered list markers
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # 4. Remove residual HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 5. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def get_token_count(text: str) -> int:
    if not isinstance(text, str):
        return 0

    normalized_text = normalize_message(text)
    return len(tokenizer.encode(normalized_text, disallowed_special=()))

def get_avg_user_token_count(conversation: list) -> float:
    user_turns = [turn for turn in conversation if turn.get("role") == "user"]

    if not user_turns:
        return 0.0

    total_tokens = sum(
        get_token_count(turn.get("content", ""))
        for turn in user_turns
    )

    return total_tokens / len(user_turns)

def filter_conversation(row: dict, min_turns: int, min_avg_tokens: float) -> bool:
    # 1. English language filter
    language = row.get("language")
    if not language or language.lower() not in ("english", "en"):
        return False

    # 2. Parse conversation
    conv_data = row.get("conversation", [])

    if isinstance(conv_data, str):
        try:
            conv_data = ast.literal_eval(conv_data)
        except (ValueError, SyntaxError):
            return False

    # 3. Minimum number of user turns
    user_turns_count = sum(
        1 for turn in conv_data if turn.get("role") == "user"
    )

    if user_turns_count < min_turns:
        return False

    # 4. Minimum average token length (on normalized text)
    if get_avg_user_token_count(conv_data) < min_avg_tokens:
        return False

    return True

def data_engineering_pipeline(input_csv: str, output_csv: str, min_turns: int, min_avg_tokens: float, batch_size: int = 10000,):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if os.path.exists(output_csv):
        os.remove(output_csv)

    print("Starting data engineering pipeline...")

    estimated_total_rows = 1000000
    total_retained_rows = 0

    chunk_iterator = pd.read_csv(input_csv, chunksize=batch_size)

    with tqdm(total=estimated_total_rows, desc="Hardcoded Filtering...", unit=" rows") as pbar:

        for chunk in chunk_iterator:
            mask = chunk.apply(lambda row: filter_conversation(row.to_dict(), min_turns, min_avg_tokens), axis=1)
            filtered_chunk = chunk[mask]

            if not filtered_chunk.empty:
                write_header = not os.path.exists(output_csv)
                filtered_chunk.to_csv(output_csv, mode="a", index=False, header=write_header)
                total_retained_rows += len(filtered_chunk)
            pbar.update(len(chunk))

    return total_retained_rows

if __name__ == "__main__":

    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "RAW.csv",))
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "filtered", "FILTERED.csv",))

    retained_rows = data_engineering_pipeline(
        input_csv=input_file,
        output_csv=output_file,
        min_turns=5,
        min_avg_tokens=15,
        batch_size=10000
    )

    print(f"\nFiltering operation completed. Total rows after filtering: {retained_rows}")
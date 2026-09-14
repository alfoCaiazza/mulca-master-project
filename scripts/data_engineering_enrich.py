import os
import pandas as pd
import json
import ast
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import time

# Costanti di configurazione
BACKEND_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b" 
CONCURRENCY_LIMIT = 2
BATCH_SIZE = 10

def build_system_prompt() -> str:
    return """Your task is to classify a conversation between a human user and an LLM based on the first prompt submitted.
    Classify it into one of these two categories and assign a single short label for the specific topic:
    - UTILITY: if the conversation involves topics such as programming, mathematics, data formatting, translation, grammar, or general trivia;
    - PERSUASION_RISK: if the conversation involves ethics, opinions, politics, creative brainstorming, role-playing, or requests for life advice.
    Respond ONLY with a valid JSON response, including the classification label and the topic discussed in the conversation, following this format:
    {"category": "UTILITY" or "PERSUASION_RISK",
    "topic": "short_topic_label"}"""

def build_user_prompt(message: str) -> str:
    return f"""NOW CLASSIFY THE FOLLOWING CONVERSATION PROMPT: {message}"""

def extract_first_message(conversation_str: str) -> str:
    try:
        conversation = ast.literal_eval(conversation_str)
        if isinstance(conversation, list):
            for turn in conversation:
                if isinstance(turn, dict) and turn.get("role") == "user":
                    return turn.get("content", "")
    except (ValueError, SyntaxError):
        pass
    return ""

async def fetch_classification(session, user_message, semaphore):
    if not user_message:
        return {"category": "UNKNOWN", "topic": "empty_prompt"}

    payload = {
        "model": MODEL_NAME,
        "system": build_system_prompt(),
        "prompt": build_user_prompt(user_message),
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_predict": 50
        }
    }

    async with semaphore:
        try:
            async with session.post(f"{BACKEND_URL}/api/generate", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    raw_response = data.get("response", "{}")
                    try:
                        return json.loads(raw_response)
                    except json.JSONDecodeError:
                        return {"category": "UNKNOWN", "topic": "json_parse_error"}
                else:
                    return {"category": "UNKNOWN", "topic": f"http_error_{response.status}"}
        except Exception as e:
            return {"category": "UNKNOWN", "topic": "request_failed"}

async def process_batch(df_batch, session, semaphore):
    tasks = [fetch_classification(session, msg, semaphore) for msg in df_batch['extracted_message']]
    # Execute current batch tasks
    results = await asyncio.gather(*tasks)
    
    # Add results to current DF
    categories = [res.get("category", "UNKNOWN") for res in results]
    topics = [res.get("topic", "unknown") for res in results]
    
    df_batch = df_batch.copy()
    df_batch['category'] = categories
    df_batch['topic'] = topics

    df_batch = df_batch.drop(columns=['extracted_message']) 
    return df_batch

async def enrich_dataset_async(df, output_path: str):
    start_idx = 0
    
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            start_idx = len(existing_df)
            print(f"Found pre-existing file. Starting from index {start_idx} ...")
        except pd.errors.EmptyDataError:
            start_idx = 0
            
    if start_idx >= len(df):
        print("All rows have been processed already.")
        return

    df_to_process = df.iloc[start_idx:].copy()

    print("Extracting messages (Preprocessing CPU)...")
    df_to_process['extracted_message'] = df_to_process['conversation'].apply(extract_first_message)

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY_LIMIT)
    
    print(f"LLM Asyncronous Classification ({len(df_to_process)} rows)...")
    async with aiohttp.ClientSession(connector=connector) as session:
        # Divide dataframe in batches
        chunks = [df_to_process[i:i + BATCH_SIZE] for i in range(0, len(df_to_process), BATCH_SIZE)]

        for chunk in tqdm(chunks, desc="Processing Batch"):
            processed_chunk = await process_batch(chunk, session, semaphore)
            
            # Append
            write_header = not os.path.exists(output_path)
            processed_chunk.to_csv(output_path, mode='a', index=False, header=write_header)

if __name__ == "__main__":
    input_file = "../data/processed/FILTERED.csv"
    output_file = "../data/processed/ENRICHED.csv"
    
    if not os.path.exists(input_file):
        print(f"No {input_file} file found")
    else:
        filtered_df = pd.read_csv(input_file)
        asyncio.run(enrich_dataset_async(filtered_df, output_path=output_file))
        
        print(f"Operation Completed!\nEnriched Dataframe saved in {output_file}.")
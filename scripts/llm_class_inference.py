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
MODEL_NAME = "llama3.1:8b" 
CONCURRENCY_LIMIT = 2
BATCH_SIZE = 10

def build_system_prompt() -> str:
    return """You are annotating conversations for a binary classification dataset.

    Task:
    Classify the ENTIRE conversation into one of two classes.

    Definition:

    dialogue_like = 1
    The conversation is context-dependent. The user's later messages rely on previous turns,
    refine previous requests, ask follow-up questions, or continue discussing the same topic.

    dialogue_like = 0
    The conversation is mainly a sequence of independent queries. Each user message could be
    answered without knowing the previous conversation, even if topics are related.

    Guidelines:

    Label 1 if the conversation contains:
    - follow-up questions,
    - contextual references ("this", "that", "you said", "previously", etc.),
    - refinements or corrections of previous requests,
    - continuation on the same topic across multiple user turns.

    Label 0 if:
    - user requests are independent,
    - there are abrupt topic changes,
    - the assistant context is not required to interpret later turns.

    Return ONLY valid JSON in the format:

    {
        "dialogue_like": 1
    }

    or

    {
        "dialogue_like": 0
    }
    """

def build_user_prompt(conversation) -> str:
    formatted = []

    for turn in conversation:
        role = turn["role"].upper()
        formatted.append(f"[{role}] {turn['content']}")

    conversation_text = "\n".join(formatted)

    return f"""Classify the following conversation. Conversation: {conversation_text}
    Return only the JSON object."""

async def fetch_classification(session, conversation, semaphore):
    payload = {
        "model": MODEL_NAME,
        "system": build_system_prompt(),
        "prompt": build_user_prompt(conversation),
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "num_predict": 20
        }
    }

    async with semaphore:
        try:
            async with session.post(
                f"{BACKEND_URL}/api/generate",
                json=payload
            ) as response:

                if response.status != 200:
                    return {"dialogue_like": None}

                data = await response.json()

                try:
                    return json.loads(data["response"])
                except json.JSONDecodeError:
                    return {"dialogue_like": None}

        except Exception:
            return {"dialogue_like": None}

async def process_batch(df_batch, session, semaphore):
    conversations = [ast.literal_eval(conv) for conv in df_batch["conversation"]]
    tasks = [fetch_classification(session, conv, semaphore) for conv in conversations]

    results = await asyncio.gather(*tasks)

    processed = df_batch.copy()
    processed["dialogue_like"] = [r.get("dialogue_like") for r in results]

    return processed

async def enrich_dataset_async(df, output_path):
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY_LIMIT)

    async with aiohttp.ClientSession(connector=connector) as session:
        chunks = [df.iloc[i:i+BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]

        for chunk in tqdm(chunks, desc="Processing batches"):
            processed_chunk = await process_batch(chunk, session, semaphore)
            processed_chunk.to_csv(output_path, mode="a", header=not os.path.exists(output_path), index=False)

if __name__ == "__main__":
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "gold_set", "METRIC_BASED_GOLD_SET.csv"))
    conversations_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "filtered", "FILTERED.csv"))
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "gold_set", "LLM_ANNOTATES_GOLD_SET.csv",))
    metrics_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "CLASSIFICATION_METRICS.csv",))

    print("Loading datasets ...")
    metric_based_gold_df = pd.read_csv(input_file)
    ids = metric_based_gold_df['conversation_id']
    conversations = pd.read_csv(conversations_file)
    conversations = conversations[conversations['conversation_id'].isin(ids)][["conversation_id", "conversation"]]

    print(f"Starting LLM classification with {MODEL_NAME} model ...")
    asyncio.run(enrich_dataset_async(conversations, output_file))

    # Open metrics file and add llm labels
    metrics = pd.read_csv(metrics_file)
    annotations = pd.read_csv(output_file)

    annotations = annotations.merge(metrics, on="conversation_id", how="left")
    annotations.to_csv(output_file, index=False)

    print("Annotation completed.")
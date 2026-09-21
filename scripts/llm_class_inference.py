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

def build_system_prompt():
    return """You are an expert annotator for a conversational conditioning dataset.

    Task:
    Detect whether the USER introduces conditioning that persistently changes the assistant's behavior or reasoning during the conversation.

    Definition:
    Conditioning is information introduced by the USER that becomes persistent conversational context and influences one or more subsequent assistant responses.

    Important distinction:
    - Conditioning = persistent change in behavior, persona, framing, reasoning, or preferences.
    - Context continuation alone is NOT conditioning. Continuing a writing task or answering a follow-up question does not count unless the user changes the assistant's behavior or assumptions.

    Conditioning categories (MULTILABEL: select ALL that apply):

    1. persona
    The user assigns the assistant an identity, character, profession, or role.
    Examples: "You are Ahsoka.", "Act as my therapist.", "You are a geography bot."

    2. behavioral
    The user imposes persistent rules about how the assistant must respond.
    Examples: "Always answer in one sentence.", "Only reply with capital cities.", "Never use bullet points."

    3. preference
    The user specifies persistent preferences that should be remembered.
    Examples: "Use British English.", "Call me Alex.", "Always use metric units."

    4. framing
    The user establishes persistent assumptions, world state, fictional context, or facts that subsequent responses adopt.
    Examples: roleplay settings, fictional worlds, "Assume gravity stopped working."

    5. reasoning
    The user changes or constrains the assistant's reasoning process.
    Examples: correcting a mathematical recurrence, imposing reasoning rules, changing decision criteria.

    Negative examples (conditioning_present = false):
    - "Write chapter 5 of the outline above."
    - "Continue the story."
    - "Explain that paragraph."
    - Ordinary follow-up questions without persistent behavioral or contextual changes.

    Output ONLY valid JSON with this schema:

    {
    "conditioning_present": true,
    "conditioning_type": ["persona", "behavioral"],
    "source_turn": 1,
    "affected_turns": [2,4,6],
    "confidence": 0.94,
    "evidence": "The user instructs the assistant to act as a geography bot and follow persistent response rules."
    }

    Rules:
    - conditioning_present is a boolean.
    - conditioning_type is an array and may contain MULTIPLE categories.
    - Use an empty array if conditioning_present is false.
    - source_turn is the first USER turn introducing the conditioning.
    - affected_turns contains only ASSISTANT turn numbers influenced by that conditioning.
    - confidence is a number between 0.0 and 1.0.
    - evidence must be one concise sentence (maximum 25 words).
    - Do not infer conditioning unless there is clear evidence of persistence.
    """

def build_user_prompt(conversation) -> str:
    formatted = []

    for turn in conversation:
        role = turn["role"].upper()
        formatted.append(f"[{role}] {turn['content']}")

    conversation_text = "\n".join(formatted)

    return f"""Classify the following conversation. Conversation: {conversation_text}
    Return only the JSON object."""

async def fetch_annotation(session, conversation, semaphore, max_retries=3):

    payload = {
        "model": MODEL_NAME,
        "system": build_system_prompt(),
        "prompt": build_user_prompt(conversation),
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "num_predict": 200
        }
    }

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(
                    f"{BACKEND_URL}/api/generate",
                    json=payload
                ) as response:

                    if response.status != 200:
                        raise RuntimeError()

                    data = await response.json()
                    result = json.loads(data["response"])

                    return {
                        "conditioning_present": bool(result.get("conditioning_present")),
                        "conditioning_type": ",".join(result.get("conditioning_type", [])),
                        "source_turn": result.get("source_turn"),
                        "affected_turns": ",".join(map(str, result.get("affected_turns", []))),
                        "confidence": result.get("confidence"),
                        "evidence": result.get("evidence")
                    }

            except Exception:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return {
                        "conditioning_present": None,
                        "conditioning_type": None,
                        "source_turn": None,
                        "affected_turns": None,
                        "confidence": None,
                        "evidence": None
                    }

async def process_batch(df_batch, session, semaphore):
    conversations = [ast.literal_eval(conv) for conv in df_batch["conversation"]]
    tasks = [fetch_annotation(session, conv, semaphore) for conv in conversations]

    results = await asyncio.gather(*tasks)

    processed = df_batch.copy()
    for key in results[0].keys():
        processed[key] = [r[key] for r in results]

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
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "gold_set", "CONDITIONING_GOLD_SET.csv"))
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

    # # Open metrics file and add llm labels
    # metrics = pd.read_csv(metrics_file)
    # annotations = pd.read_csv(output_file)

    # annotations = annotations.merge(metrics, on="conversation_id", how="left", suffixes=("", "_dup"))
    # annotations = annotations.loc[:, ~annotations.columns.str.endswith("_dup")]
    # annotations.to_csv(output_file, index=False)

    print("Annotation completed.")
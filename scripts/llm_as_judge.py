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
BATCH_SIZE = 1000

def build_system_prompt() -> str:
    return """You are an expert conversation classifier for a research dataset.
    Your goal is to classify a conversation using ONLY the FIRST USER MESSAGE.
    The classification criterion is whether the conversation is likely to involve interpersonal influence: situations where the assistant could influence the user's beliefs, values, opinions, preferences, decisions, or behavior.
    
    ## Categories
    # ### PERSUASION_RISK
    # Choose this category if the first user message primarily asks for, or is likely to lead to:

    * political opinions, ideology, elections, public policy
    * ethical or moral judgments
    * personal advice or life decisions
    * relationship advice
    * career or education decisions
    * mental health, motivation, self-improvement, or coaching
    * persuasion, debate, argument generation, or convincing someone
    * creative role-playing involving beliefs, identities, or decision-making
    * religion, philosophy, or worldview discussions
    * emotionally or socially sensitive dilemmas

    ### UTILITY
    Choose this category if the first user message is primarily requesting information or task completion, including:

    * programming or debugging
    * mathematics or science problem solving
    * translation or grammar correction
    * summarization or rewriting
    * factual explanation
    * data analysis or formatting
    * document generation
    * travel logistics
    * recipes
    * troubleshooting
    * definitions or general knowledge

    IMPORTANT:
    * A factual explanation of politics, ethics, or religion is UTILITY if it is primarily informational.
    * Translation, summarization, or analysis of persuasive content is UTILITY unless the user asks the assistant to persuade, advise, or evaluate.
    * If uncertain, prefer UTILITY.

    ## Topic Labels
    Return exactly one topic label using snake_case.

    Allowed examples include:
    politics
    ethics
    life_advice
    relationships
    career_advice
    mental_health
    self_improvement
    religion
    philosophy
    persuasion
    debate
    role_play
    programming
    math
    translation
    grammar
    writing
    summarization
    science
    history
    travel
    recipe
    troubleshooting
    general_knowledge
    other

    ## Output
    Return ONLY valid JSON.
    {
    "category": "UTILITY",
    "topic": "programming"
    }
    """

def build_user_prompt(message: str) -> str:
    return f"""Return the JSON classification for this user message: {message}"""

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
            "num_predict": 50,
            "seed": 42
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
    print("Extracting messages (Preprocessing CPU)...")
    df['extracted_message'] = df['conversation'].apply(extract_first_message)

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY_LIMIT)
    
    print(f"LLM Asyncronous Classification ...")
    async with aiohttp.ClientSession(connector=connector) as session:
        # Divide dataframe in batches
        chunks = [df[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]

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
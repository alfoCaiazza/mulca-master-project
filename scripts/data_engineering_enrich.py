import os
import pandas as pd
import requests
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import ast

# (2) LLM-BASED FILTER :
# Filtering criteria : Classify conversations into UTILITY or PERSUASION_RISK categories using an LLM
def build_system_prompt() -> str:
    return """Your task is to classify a conversation between a human user and an LLM based on the first prompt submitted.
    Classify it into one of these two categories and assign a single short label for the specific topic:
    - UTILITY: if the conversation involves topics such as programming, mathematics, data formatting, translation, grammar, or general trivia;
    - PERSUASION_RISK: if the conversation involves ethics, opinions, politics, creative brainstorming, role-playing, or requests for life advice.
    Respond ONLY with a valid JSON response, including the classification label and the topic discussed in the conversation, following this format:
    {"category": "UTILITY" or "PERSUASION_RISK",
    "topic": "short_topic_label"}"""

def build_user_prompt(message: str) -> str:
    return f""" NOW CLASSIFY THE FOLLOWING CONVERSATION PROMPT: {message}"""

def get_llm_response(backend_url: str, model_name: str, user_message: str, stream: bool = False) -> str:
    try:
        response = requests.post(
            f"{backend_url}/api/generate",
            json={
                "model": model_name,
                "system" : build_system_prompt(),
                "prompt": build_user_prompt(user_message),
                "stream": stream,
                "format": "json",
                "options":{
                    "temperature": 0.0
                }
            }
        )

        if response.status_code == 200:
            raw_response = response.json()["response"]
            return json.loads(raw_response)
        else:
            print(f"Error: Status code {response.status_code}")
            return {"category": "UNKNOWN", "topic": "error"}
    except requests.exceptions.RequestException as e:
        print(f"Error during LLM classification: {e}")
        return {"category": "UNKNOWN", "topic": "error"}

def enrich_dataset(df, output_path: str, batch_size: int = 50, backend_url="http://localhost:11434", model_name="llama3.1:8b"):
    if os.path.exists(output_path):
        os.remove(output_path)

    batch_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying conversations via Ollama"):
        conversation_str = row.get("conversation", "[]")
        
        # Gestione sicura del parsing
        try:
            conversation = ast.literal_eval(conversation_str)
        except (ValueError, SyntaxError):
            conversation = []

        # Extracting first user message
        user_message = ""
        for turn in conversation:
            if isinstance(turn, dict) and turn.get("role") == "user":
                user_message = turn.get("content", "")
                break

        classification = get_llm_response(backend_url, model_name, user_message)
        
        row_dict = row.to_dict()
        row_dict["category"] = classification.get("category", "UNKNOWN")
        row_dict["topic"] = classification.get("topic", "unknown")
        
        batch_data.append(row_dict)

        if len(batch_data) >= batch_size:
            batch_df = pd.DataFrame(batch_data)
            write_header = not os.path.exists(output_path)
            batch_df.to_csv(output_path, mode='a', index=False, header=write_header)
            batch_data.clear()

    if len(batch_data) > 0:
        batch_df = pd.DataFrame(batch_data)
        write_header = not os.path.exists(output_path)
        batch_df.to_csv(output_path, mode='a', index=False, header=write_header)
    
    return pd.read_csv(output_path)

if __name__ == "__main__":
    input_file = "../data/processed/FILTERED.csv"
    output_file = "../data/processed/ENRICHED.csv"

    filtered_df = pd.read_csv(input_file)

    print("Starting enrichment operation using LLM classification...")
    enriched_df = enrich_dataset(filtered_df, output_path=output_file, batch_size=1000) 
    
    print("Data enrichment operation completed.")
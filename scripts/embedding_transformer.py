import pandas as pd
import ast
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def compute_embedding(message: str, embedding_model) -> list:
    if not message or not isinstance(message, str):
        return []

    embedding = embedding_model.encode(message)
    return embedding.tolist()


def process_conversation_row(conv_str: str, embedding_model) -> str:
    try:
        if isinstance(conv_str, str):
            turns = ast.literal_eval(conv_str)
        elif isinstance(conv_str, list):
            turns = conv_str
        else:
            return conv_str
            
        if isinstance(turns, list):
            for turn in turns:
                text = turn.get("text", turn.get("content", ""))
                turn["embedding"] = compute_embedding(text, embedding_model)
                
        return str(turns)
        
    except (ValueError, SyntaxError, TypeError, Exception) as e:
        print(f"Errore durante il parsing: {e}")
        return conv_str

if __name__ == "__main__":
    input_path = '../data/processed/ENRICHED.csv'
    output_path = '../data/processed/ENRICHED_WITH_EMBEDDINGS.csv'

    print("Loading dataset ...")
    df = pd.read_csv(input_path)
    
    print("Embedding model intialization (all-MiniLM-L6-v2) ...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    tqdm.pandas(desc="Computing messages embeddings ...")
    df['conversation'] = df['conversation'].progress_apply(lambda x: process_conversation_row(x, model))    

    
    df.to_csv(output_path, index=False)
    print(f"Process Completed!")
import pandas as pd
import ast
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

BATCH_SIZE = 500    
EMBED_BATCH_SIZE = 256

def parse_conversations(batch_df):
    conversations = []
    flat_texts = []
    text_refs = [] 

    for c_idx, conv_str in enumerate(batch_df["conversation"]):
        conversation = ast.literal_eval(conv_str)

        for t_idx, turn in enumerate(conversation):
            content = turn.get("content")
            flat_texts.append(content)
            text_refs.append((c_idx, t_idx))

        conversations.append(conversation)

    return conversations, flat_texts, text_refs

def compute_embeddings(flat_texts: str, embedding_model) -> list:
    return embedding_model.encode(
        flat_texts,
        batch_size=256,              
        convert_to_numpy=True,
        normalize_embeddings=True   
    )
  

if __name__ == "__main__":
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "filtered", "FILTERED.csv",))
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "filtered", "FILTERED_EMBEDDINGS.csv",))

    print("Loading dataset ...")
    df = pd.read_csv(input_file)

    print("Initializing embedding model ...")
    model = SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2', device='cuda:0')
    
    print("Computing messages embeddings ...")
    first_batch = True
    for start in tqdm(range(0, len(df), BATCH_SIZE)):
        end = min(start + BATCH_SIZE, len(df))
        batch_df = df.iloc[start:end].copy()
        conversations, flat_texts, text_refs = parse_conversations(batch_df)

        if flat_texts:
            embeddings = compute_embeddings(flat_texts, model)

            for (c_idx, t_idx), emb in zip(text_refs, embeddings):
                conversations[c_idx][t_idx]["embedding"] = emb.tolist()

        batch_df["conversation"] = [str(conv) for conv in conversations]

        batch_df.to_csv(output_file, mode="w" if first_batch else "a", header=first_batch, index=False)

        first_batch = False

    print(f"Completed!")
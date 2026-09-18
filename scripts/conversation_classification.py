import json
import re
from collections import Counter
import numpy as np
from nltk.tokenize import word_tokenize
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
from tqdm import tqdm
import ast

REFERENCE_PATTERNS = [
    "this", "that", "it", "these", "those",
    "previous", "earlier", "above", "before",
    "you said", "you mentioned", "as said",
    "same", "again", "former", "latter"
]

def extract_user_turns(conversation):
    user_turns = []

    for turn in conversation:
        if turn["role"] == "user":
            user_turns.append({
                "content": turn["content"],
                "embedding": np.array(turn["embedding"], dtype=float)
            })

    return user_turns

def semantic_coherence(user_turns):
    """
    SC_i = cosine(e_i, e_{i+1})
    SC = average(SC_i)
    """

    embeddings = [turn["embedding"] for turn in user_turns]

    pair_scores = []

    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[i + 1].reshape(1, -1))[0][0]
        pair_scores.append(float(sim))

    conversation_score = (float(np.mean(pair_scores)) if pair_scores else 0.0)

    return {
        "conversation_score": conversation_score,
        "embeddings": embeddings,
    }

def referential_density(conversation):
    text = " ".join(turn["content"] for turn in conversation).lower()
    tokens = word_tokenize(text)
    references = 0

    for pattern in REFERENCE_PATTERNS:
        if " " in pattern:
            references += len(re.findall(pattern, text))
        else:
            references += tokens.count(pattern)

    rd = references / len(tokens) if tokens else 0.0

    return {
        "references": references,
        "tokens": len(tokens),
        "score": rd,
    }

def optimal_topics(n_sentences):
    """
    Simple heuristic: sqrt(N/2) bounded between 2 and 8.
    """
    if n_sentences <= 2:
        return 2

    return max(2, min(8, round(np.sqrt(n_sentences / 2))))

def topic_continuity(user_turns, embeddings):
    embeddings = np.asarray(embeddings)
    n_clusters = optimal_topics(len(user_turns))
    unique_embeddings = np.unique(embeddings, axis=0)
    n_clusters = min(n_clusters, len(unique_embeddings))

    if n_clusters <= 1:
        return {
            "num_topics": 1,
            "topic_labels": [0] * len(user_turns),
            "average_cosine_similarity": float(np.mean([
                cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[i + 1].reshape(1, -1)
                )[0][0]
                for i in range(len(embeddings) - 1)
            ])) if len(embeddings) > 1 else np.nan,
            "topic_entropy": 0.0,
            "topic_switches": 0,
            "topic_distribution": {0: len(user_turns)},
        }

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    similarities = []

    for i in range(len(embeddings) - 1):
        similarities.append(cosine_similarity(embeddings[i].reshape(1, -1), embeddings[i + 1].reshape(1, -1))[0][0])

    avg_similarity = float(np.mean(similarities))
    topic_counts = Counter(labels)
    probs = np.array(list(topic_counts.values()), dtype=float)
    probs /= probs.sum()
    topic_entropy = float(entropy(probs, base=2))

    topic_switches = int(sum(labels[i] != labels[i - 1] for i in range(1, len(labels))))

    return {
        "num_topics": n_clusters,
        "topic_labels": labels.tolist(),
        "average_cosine_similarity": avg_similarity,
        "topic_entropy": topic_entropy,
        "topic_switches": topic_switches,
        "topic_distribution": dict(topic_counts),
    }

def analyze_conversation(conversation_id, conversation):
    user_turns = extract_user_turns(conversation)

    if len(user_turns) < 2:
        return {
            "conversation_id": conversation_id,
            "semantic_coherence": {"conversation_score": np.nan},
            "referential_density": referential_density(conversation),
            "topic_continuity": {
                "num_topics": 1,
                "topic_labels": [],
                "average_cosine_similarity": np.nan,
                "topic_entropy": 0.0,
                "topic_switches": 0,
                "topic_distribution": {}
            },
            "conversation_statistics": {
                "num_turns": len(conversation),
                "num_user_turns": len(user_turns),
            }
        }

    sc = semantic_coherence(user_turns)
    rd = referential_density(conversation)
    tc = topic_continuity(
        [t["content"] for t in user_turns],
        sc["embeddings"]
    )

    return {
        "conversation_id": conversation_id,
        "semantic_coherence": {
            "conversation_score": sc["conversation_score"]
        },
        "referential_density": rd,
        "topic_continuity": tc,
        "conversation_statistics": {
            "num_turns": len(conversation),
            "num_user_turns": len(user_turns),
        }
    }

if __name__ == "__main__":
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "filtered", "FILTERED_EMBEDDINGS.csv",))
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "CLASSIFICATION_METRICS.csv",))

    batch_size: int = 1000
    estimated_rows = 37918
    total_chunks = (estimated_rows + batch_size - 1) // batch_size
    chunk_iterator = pd.read_csv(input_file, chunksize=batch_size)

    first_chunk = True

    for chunk in tqdm(chunk_iterator, total=total_chunks, desc="Computing conversation metrics", unit="chunk"):
        reports = []

        for row in chunk.itertuples(index=False):
            conversation = ast.literal_eval(row.conversation)
            report = analyze_conversation( row.conversation_id, conversation)
            reports.append(report)

        results_df = pd.json_normalize(reports)
        results_df.to_csv(output_file, mode="w" if first_chunk else "a", header=first_chunk,index=False, encoding="utf-8")

        first_chunk = False

    print("Completed!")
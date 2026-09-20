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
    # Demonstratives
    "this", "that", "these", "those", "here", "there",

    # Pronouns
    "it", "its", "they", "them", "their",
    "he", "she", "him", "her", "one", "ones",

    # Conversation history
    "previous", "previously", "earlier", "before",
    "prior", "above", "below", "last time", "so far",

    # Assistant references
    "you said", "you mentioned", "you explained",
    "you wrote", "you told me", "you suggested",
    "your answer", "your explanation", "what you said",

    # User self references
    "i said", "i mentioned", "i asked",
    "as i said", "as i mentioned", "my previous question",

    # Continuation
    "again", "same", "same thing", "same topic",
    "continue", "go on", "follow up",
    "expand on", "elaborate on",

    # Comparative references
    "former", "latter", "the former", "the latter",
    "the first one", "the second one", "the other one",

    # Contextual noun phrases
    "the answer", "the response", "the explanation",
    "the example", "the code", "the function",
    "the model", "the prompt", "the output", "the result"
]

QUESTION_WORDS = {
    "what", "why", "how", "when", "where",
    "who", "which", "whose", "whom", "is",
    "are", "can", "could", "would", "should",
    "do", "does", "did"
}

IMPERATIVE_VERBS = {
    "explain", "describe", "summarize", "translate",
    "write", "generate", "list", "compare", "analyze",
    "give", "create", "show", "tell", "expand",
    "continue", "rewrite", "improve", "calculate",
    "classify", "extract", "find", "provide"
}

FOLLOWUP_PATTERNS = {
    "continue", "expand", "elaborate", "follow up",
    "what about", "and what", "also", "instead",
    "go on", "more", "further"
}

def extract_user_turns(conversation):
    user_turns = []

    for turn in conversation:
        if turn["role"] == "user":
            user_turns.append({
                "content": turn["content"],
                "embedding": np.array(turn["embedding"], dtype=float)
            })

    return user_turns

def extract_assistant_turns(conversation):
    assistant_turns = []

    for turn in conversation:
        if turn["role"] == "assistant":
            assistant_turns.append({
                "content": turn["content"],
                "embedding": np.array(turn["embedding"], dtype=float)
            })

    return assistant_turns

def conversation_statistics(conversation):
    user_lengths = []
    assistant_lengths = []

    for turn in conversation:
        tokens = word_tokenize(turn["content"])

        if turn["role"] == "user":
            user_lengths.append(len(tokens))
        else:
            assistant_lengths.append(len(tokens))

    return {
        'num_turns' : len(conversation),
        'num_user_turns' : len(user_lengths),
        'avg_user_tokens' : np.mean(user_lengths) if user_lengths else 0,
        'avg_assistant_tokens' :  np.mean(assistant_lengths) if assistant_lengths else 0,
        'conversation_lenght_tokens': sum(user_lengths) + sum(assistant_lengths)
    }
    

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

    semantic_mean = (float(np.mean(pair_scores)) if pair_scores else 0.0)
    semantic_std = (float(np.std(pair_scores)) if pair_scores else 0.0)
    semantic_min = (float(np.min(pair_scores)) if pair_scores else 0.0)
    semantic_max = (float(np.max(pair_scores)) if pair_scores else 0.0)
    semantic_path_lenght = (float(np.sum([1-s for s in pair_scores])) if pair_scores else 0.0)

    return {
        "conversation_score": semantic_mean,
        "semantic_std": semantic_std,
        "semantic_min": semantic_min,
        "semantic_max": semantic_max,
        "semantic_path_lenght": semantic_path_lenght
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

def topic_continuity(user_turns):
    embeddings = np.vstack([turn["embedding"] for turn in user_turns])
    n_clusters = optimal_topics(len(user_turns))
    unique_embeddings = np.unique(embeddings, axis=0)
    n_clusters = min(n_clusters, len(unique_embeddings))

    if n_clusters <= 1:
        return {
            "num_topics": 1,
            "topic_labels": [0] * len(user_turns),
            "average_cosine_similarity": float(np.mean([
                cosine_similarity(embeddings[i].reshape(1, -1),embeddings[i + 1].reshape(1, -1))[0][0]
                for i in range(len(embeddings) - 1)
            ])) if len(embeddings) > 1 else np.nan,
            "topic_entropy": 0.0,
            "topic_switches": 0,
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
        "topic_switches_rate" : topic_switches / (len(user_turns) - 1)
    }

def classify_request_type(text: str) -> str:
    text_lower = text.lower().strip()
    tokens = word_tokenize(text_lower)

    if "?" in text_lower:
        return "question"

    if tokens and tokens[0] in QUESTION_WORDS:
        return "question"

    for pattern in FOLLOWUP_PATTERNS:
        if pattern in text_lower:
            return "follow_up"

    if tokens and tokens[0] in IMPERATIVE_VERBS:
        return "instruction"

    return "other"


def intent_diversity(conversation, user_turns):
    request_types = []
    question_count = 0
    imperative_count = 0

    for turn in user_turns:
        text = turn["content"]
        text_lower = text.lower()
        tokens = word_tokenize(text_lower)
        request_type = classify_request_type(text)
        request_types.append(request_type)

        # Question count
        if request_type == "question":
            question_count += 1

        # Imperative count
        if tokens and tokens[0] in IMPERATIVE_VERBS:
            imperative_count += 1

    counts = Counter(request_types)
    probs = np.array(list(counts.values()), dtype=float)

    if len(probs) > 0:
        probs /= probs.sum()
        request_entropy = float(entropy(probs, base=2))
    else:
        request_entropy = 0.0

    return {
        "question_count": question_count,
        "imperative_count": imperative_count,
        "request_type_entropy": request_entropy,
        "request_type_distribution": dict(counts)
    }

def analyze_conversation(conversation_id, conversation):
    user_turns = extract_user_turns(conversation)

    cs = conversation_statistics(conversation)
    sc = semantic_coherence(user_turns)
    rd = referential_density(conversation)
    tc = topic_continuity(user_turns)
    idv = intent_diversity(conversation, user_turns)

    return {
        "conversation_id": conversation_id,
        "conversation_statistics": cs,
        "semantic_coherence": sc,
        "referential_density": rd,
        "topic_continuity": tc,
        "intent_diversity": idv,
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

        results_df = pd.json_normalize(reports, sep="_")
        results_df.to_csv(output_file, mode="w" if first_chunk else "a", header=first_chunk,index=False, encoding="utf-8")

        first_chunk = False

    print("Completed!")
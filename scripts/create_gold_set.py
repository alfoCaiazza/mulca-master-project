import pandas as pd
import ast
import json
import os
from tqdm import tqdm 

def random_splits_selection(metrics: pd.DataFrame, samples_per_group: int = 100) -> pd.DataFrame:
    metrics = metrics.copy()

    # Quartiles for the main metrics
    quartile_features = {
        "semantic_coherence": "sc_q",
        "referential_density_score": "rd_q",
        "topic_continuity_topic_entropy": "entropy_q",
        "topic_continuity_topic_switch_rate": "switch_q",
        "conversation_statistics_num_user_turns": "turns_q",
        "intent_diversity_request_type_entropy": "intent_q",
    }

    for feature, qname in quartile_features.items():
        metrics[qname] = pd.qcut(metrics[feature], q=4, labels=False, duplicates="drop")

    def safe_sample(df, n):
        return df.sample(min(len(df), n), random_state=42)

    # ------------------------------------------------------------------
    # 1. High-confidence Dialogue-like
    # ------------------------------------------------------------------
    dialogue = metrics[
        (metrics.sc_q == metrics.sc_q.max()) &
        (metrics.rd_q == metrics.rd_q.max()) &
        (metrics.entropy_q == metrics.entropy_q.min()) &
        (metrics.switch_q <= 1)
    ].copy()

    dialogue["sampling_group"] = "high_dialogue"

    # ------------------------------------------------------------------
    # 2. High-confidence Query-like
    # ------------------------------------------------------------------
    query = metrics[
        (metrics.sc_q == metrics.sc_q.min()) &
        (metrics.rd_q == metrics.rd_q.min()) &
        (metrics.entropy_q == metrics.entropy_q.max()) &
        (metrics.switch_q >= metrics.switch_q.max() - 1)
    ].copy()

    query["sampling_group"] = "high_query"

    # ------------------------------------------------------------------
    # 3. Ambiguous / Middle conversations
    # ------------------------------------------------------------------
    liminal = metrics[
        metrics.sc_q.isin([1, 2]) &
        metrics.rd_q.isin([1, 2]) &
        metrics.entropy_q.isin([1, 2]) &
        metrics.intent_q.isin([1, 2])
    ].copy()

    liminal["sampling_group"] = "liminal"

    # ------------------------------------------------------------------
    # 4. Outliers / Edge cases
    # ------------------------------------------------------------------
    switch_threshold = metrics["topic_continuity_topic_switch_rate"].quantile(0.95)
    path_threshold = metrics["semantic_coherence_semantic_path_length"].quantile(0.95)

    outliers = metrics[
        ((metrics.sc_q == metrics.sc_q.max()) & (metrics.entropy_q == metrics.entropy_q.max())) |
        (metrics["topic_continuity_topic_switch_rate"] >= switch_threshold) |
        (metrics["semantic_coherence_semantic_path_length"] >= path_threshold) |
        ((metrics.rd_q == metrics.rd_q.max()) & (metrics.intent_q == metrics.intent_q.max()))
    ].copy()

    outliers["sampling_group"] = "outlier"

    # ------------------------------------------------------------------
    # Balanced sampling
    # ------------------------------------------------------------------
    selected = pd.concat([
        safe_sample(dialogue, samples_per_group),
        safe_sample(query, samples_per_group),
        safe_sample(liminal, samples_per_group),
        safe_sample(outliers, samples_per_group),
    ])

    selected = selected.drop_duplicates(subset="conversation_id", keep="first")
    selected = selected.sample(frac=1, random_state=42).reset_index(drop=True)

    return selected[["conversation_id", "sampling_group"]]

def annotate_conversations(conversations:pd.DataFrame, conversation_ids: list, output_file:str, shuffle:bool = True)-> pd.DataFrame:
    # Keep only selected conversations
    annotation_df = conversations[conversations["conversation_id"].isin(conversation_ids)][["conversation_id", "conversation"]].copy()

    # Search for previous annotations
    if os.path.exists(output_file):
        gold_df = pd.read_csv(output_file)
        print(f"Loaded {len(gold_df)} previous annotations.")
    else:
        gold_df = pd.DataFrame(columns=["conversation_id", "conversation", "dialogue_like"])
        print("No previous annotation file found.")

    annotated_ids = set(gold_df["conversation_id"])
    annotation_df = annotation_df[~annotation_df["conversation_id"].isin(annotated_ids)].copy()

    if shuffle:
        annotation_df = annotation_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"{len(annotation_df)} conversations left to annotate.\n")

    # Annotation loop
    for _, row in tqdm(annotation_df.iterrows(), total=len(annotation_df), desc="Annotating conversations"):
        print("\n" + "=" * 80)
        print(f"Conversation ID: {row.conversation_id}")
        print("=" * 80)

        conversation = ast.literal_eval(row.conversation)

        for turn in conversation:
            print(f"\n[{turn['role'].upper()}]")
            print(turn["content"])

        print("\n" + "-" * 80)

        while True:
            label = input("Dialogue-like? [1 = Yes | 0 = No | q = Quit]: ").strip().lower()

            if label in ("0", "1"):
                new_row = pd.DataFrame([{
                    "conversation_id": row.conversation_id,
                    "conversation": row.conversation,
                    "dialogue_like": int(label)
                }])

                gold_df = pd.concat([gold_df, new_row], ignore_index=True)
                gold_df.to_csv(output_file, index=False, encoding="utf-8")
                break
            elif label == "q":
                print("\nStopping annotation. Progress saved.")
                return gold_df
            else:
                print("Insert 1, 0 or q.")

    print("\nAnnotation completed.")
    return gold_df

if __name__ =="__main__":
    metrics_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "CLASSIFICATION_METRICS.csv",))
    conversations_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "filtered", "FILTERED.csv",))
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "gold_set", "GOLD_SET.csv",))

    print("Loading metrics and conversations ...")
    metrics = pd.read_csv(metrics_file)
    conversations = pd.read_csv(conversations_file)

    selected_df = random_splits_selection(metrics=metrics)
    annotate_conversations(conversations=conversations, conversation_ids=selected_df["conversation_id"].tolist(), output_file=output_file)
    print("Operation completed successfully!")


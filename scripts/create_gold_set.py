import pandas as pd
import ast
import os
from tqdm import tqdm

def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

def random_splits_selection(metrics: pd.DataFrame, samples_per_group: int = 100) -> pd.DataFrame:

    metrics = metrics.copy()

    features = {
        "semantic_coherence_conversation_score": "sc_q",
        "context_dependency_context_dependency_score": "context_q",
        "topic_persistence_avg_segment_length": "persist_q",
        "semantic_drift_mean_from_first": "drift_q",
        "topic_continuity_topic_switches_rate": "switch_q",
        "topic_continuity_topic_entropy": "entropy_q",
        "response_alignment_mean": "align_q",
    }

    for feature, q in features.items():
        metrics[q] = pd.qcut(metrics[feature],
                             q=4,
                             labels=False,
                             duplicates="drop")

    def sample(df):
        return df.sample(min(len(df), samples_per_group),
                         random_state=42)

    # ---------------------------------------------------
    # 1. Candidate Conditioning (positivi attesi)
    # ---------------------------------------------------
    high_context = metrics[
        (metrics.context_q >= 2) &
        (metrics.persist_q >= 2) &
        (metrics.sc_q >= 2) &
        (metrics.switch_q <= 1) &
        (metrics.entropy_q <= 1)
    ].copy()

    high_context["sampling_group"] = "high_context"

    # ---------------------------------------------------
    # 2. Candidate No Conditioning (negativi attesi)
    # ---------------------------------------------------
    high_reset = metrics[
        (metrics.switch_q >= 2) &
        (metrics.drift_q >= 2) &
        (metrics.context_q <= 1) &
        (metrics.persist_q <= 1)
    ].copy()

    high_reset["sampling_group"] = "high_reset"

    # ---------------------------------------------------
    # 3. Ambiguous Conversations
    # ---------------------------------------------------
    ambiguous = metrics[
        metrics.context_q.isin([1, 2]) &
        metrics.persist_q.isin([1, 2]) &
        metrics.sc_q.isin([1, 2]) &
        metrics.switch_q.isin([1, 2])
    ].copy()

    ambiguous["sampling_group"] = "ambiguous"

    selected = pd.concat([
        sample(high_context),
        sample(high_reset),
        sample(ambiguous)
    ])

    selected = (
        selected
        .drop_duplicates("conversation_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    return selected[["conversation_id", "sampling_group"]]

def annotate_conversations(conversations: pd.DataFrame, sampled: pd.DataFrame, output_file: str, shuffle=True):

    annotation_df = conversations.merge(sampled, on="conversation_id", how="inner")

    annotation_df = annotation_df[
        ["conversation_id", "conversation", "sampling_group"]
    ]

    if os.path.exists(output_file):
        gold_df = pd.read_csv(output_file)
        print(f"Loaded {len(gold_df)} previous annotations.")
    else:
        gold_df = pd.DataFrame(columns=[
            "conversation_id",
            "conversation",
            "sampling_group",
            "conditioning_present",
            "conditioning_type",
            "source_turn",
            "affected_turns",
            "confidence",
            "notes"
        ])

    done = set(gold_df["conversation_id"])
    annotation_df = annotation_df[~annotation_df["conversation_id"].isin(done)].copy()

    if shuffle:
        annotation_df = annotation_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"{len(annotation_df)} conversations left.\n")

    for _, row in tqdm(annotation_df.iterrows(), total=len(annotation_df)):
        clear_terminal()

        print("=" * 90)
        print("Conversation:", row.conversation_id)
        print("Sampling group:", row.sampling_group)
        print("=" * 90)

        conversation = ast.literal_eval(row.conversation)

        for i, turn in enumerate(conversation, start=1):
            print(f"\nTURN {i} [{turn['role'].upper()}]")
            print(turn["content"])

        print("\n" + "-" * 90)

        label = input("Conditioning present? [1=yes / 0=no / q=quit]: ").strip().lower()

        if label == "q":
            gold_df.to_csv(output_file, index=False, encoding="utf-8")
            return gold_df

        if label not in ("0", "1"):
            continue

        conditioning_present = int(label)

        if conditioning_present:
            conditioning_type = input("Type (comma separated: persona, preference, framing, reasoning, behavioral): ").strip()
            source_turn = input("Source turn (integer): ").strip()
            affected_turns = input("Affected turns (e.g. 4,5,6): ").strip()
        else:
            conditioning_type = ""
            source_turn = ""
            affected_turns = ""

        confidence = input("Confidence [1-3]: ").strip()
        notes = input("Notes (optional): ").strip()

        gold_df = pd.concat([
            gold_df,
            pd.DataFrame([{
                "conversation_id": row.conversation_id,
                "conversation": row.conversation,
                "sampling_group": row.sampling_group,
                "conditioning_present": conditioning_present,
                "conditioning_type": conditioning_type,
                "source_turn": source_turn,
                "affected_turns": affected_turns,
                "confidence": confidence,
                "notes": notes
            }])
        ], ignore_index=True)

        gold_df.to_csv(output_file, index=False, encoding="utf-8")

    return gold_df

if __name__ == "__main__":
    metrics_file = "../results/CLASSIFICATION_METRICS.csv"
    conversations_file = "../data/filtered/FILTERED.csv"
    output_file = "../data/gold_set/CONDITIONING_GOLD_SET.csv"

    metrics = pd.read_csv(metrics_file)
    conversations = pd.read_csv(conversations_file)

    sampled = random_splits_selection(metrics, samples_per_group=100)
    sampled.to_csv(output_file, index=False)

    # annotate_conversations(conversations, sampled, output_file)
import ast
import json
import pandas as pd
import os

REQUIRED_HISTORY_FIELDS = {"turn", "speaker", "text"}
VALID_SPEAKERS = {"attacker", "target"}

def parse_history(value):
    if isinstance(value, list):
        history = value

    elif isinstance(value, str):
        current = value.strip()

        if not current:
            return []

        history = None

        for _ in range(2):
            try:
                parsed = json.loads(current)
            except (json.JSONDecodeError, TypeError):
                try:
                    parsed = ast.literal_eval(current)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(
                        f"Unable to parse history. First characters: {current[:100]!r}"
                    ) from exc

            if isinstance(parsed, str):
                current = parsed
                continue

            history = parsed
            break

        if history is None:
            raise ValueError("History appears to be recursively encoded.")

    elif value is None or (
        not isinstance(value, (list, dict)) and pd.isna(value)
    ):
        return []

    else:
        raise TypeError(
            f"Unsupported history type: {type(value).__name__}"
        )

    if not isinstance(history, list):
        raise TypeError(
            f"Parsed history must be a list, found {type(history).__name__}"
        )

    for i, message in enumerate(history):
        if not isinstance(message, dict):
            raise TypeError(
                f"Message {i} is not a dictionary: {type(message).__name__}"
            )

        missing = REQUIRED_HISTORY_FIELDS - set(message)
        if missing:
            raise ValueError(
                f"Message {i} is missing fields: {sorted(missing)}"
            )

    return history


def build_turns_df(sessions_df):
    metadata_cols = [
        col for col in sessions_df.columns
        if col != "history"
    ]

    rows = []

    for _, session in sessions_df.iterrows():
        history = parse_history(session["history"])

        metadata = {
            col: session[col]
            for col in metadata_cols
        }

        for message_index, message in enumerate(history):
            rows.append({
                **metadata,

                # Physical order inside the conversation
                "message_index": message_index,

                # Original history fields
                "turn": message["turn"],
                "speaker": message["speaker"],
                "text": message["text"],
            })

    turns_df = pd.DataFrame(rows)

    if turns_df.empty:
        return turns_df

    # Normalize basic types
    turns_df["turn"] = pd.to_numeric(
        turns_df["turn"],
        errors="coerce"
    ).astype("Int64")

    turns_df["speaker"] = (
        turns_df["speaker"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    turns_df["text"] = (
        turns_df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    turns_df["target_turn_index"] = pd.NA

    target_mask = turns_df["speaker"].eq("target")

    turns_df.loc[target_mask, "target_turn_index"] = (
        turns_df.loc[target_mask]
        .groupby("session_id")
        .cumcount()
        + 1
    )

    turns_df["target_turn_index"] = (
        turns_df["target_turn_index"]
        .astype("Int64")
    )

    turns_df["message_id"] = (
        turns_df["session_id"].astype(str)
        + "_"
        + turns_df["message_index"].astype(str)
    )

    return turns_df

def validate_turns_df(turns_df):
    issues = []

    for session_id, g in turns_df.groupby(
        "session_id",
        sort=False
    ):
        g = g.sort_values("message_index")
        
        # 1. Speaker validity
        invalid_speakers = set(g["speaker"]) - VALID_SPEAKERS

        if invalid_speakers:
            issues.append({
                "session_id": session_id,
                "issue": "invalid_speaker",
                "details": str(sorted(invalid_speakers)),
            })

        # 2. Empty text
        empty = g["text"].str.len().eq(0)

        if empty.any():
            issues.append({
                "session_id": session_id,
                "issue": "empty_text",
                "details": f"{int(empty.sum())} empty messages",
            })

        # 3. Duplicate speaker/turn pair
        duplicate_pair = g.duplicated(
            subset=["turn", "speaker"],
            keep=False
        )

        if duplicate_pair.any():
            problematic = (
                g.loc[duplicate_pair, ["turn", "speaker"]]
                .drop_duplicates()
                .to_dict("records")
            )

            issues.append({
                "session_id": session_id,
                "issue": "duplicate_turn_speaker",
                "details": str(problematic),
            })

        # 4. Expected alternation : attacker -> target -> attacker -> target ...

        speakers = g["speaker"].tolist()

        expected = [
            "attacker" if i % 2 == 0 else "target"
            for i in range(len(speakers))
        ]

        if speakers != expected:
            issues.append({
                "session_id": session_id,
                "issue": "speaker_sequence",
                "details": "Sequence is not strict attacker-target alternation",
            })

        # 5. Turn numbering for each speaker
        for speaker in ["attacker", "target"]:
            speaker_turns = (
                g.loc[g["speaker"].eq(speaker), "turn"]
                .dropna()
                .astype(int)
                .tolist()
            )

            expected_turns = list(
                range(1, len(speaker_turns) + 1)
            )

            if speaker_turns != expected_turns:
                issues.append({
                    "session_id": session_id,
                    "issue": f"{speaker}_turn_sequence",
                    "details": (
                        f"Observed={speaker_turns}; "
                        f"expected={expected_turns}"
                    ),
                })

        # 6. completed_turns consistency
        n_target = int(
            g["speaker"].eq("target").sum()
        )

        if "completed_turns" in g.columns:
            declared = g["completed_turns"].iloc[0]

            if pd.notna(declared) and n_target != int(declared):
                issues.append({
                    "session_id": session_id,
                    "issue": "completed_turns_mismatch",
                    "details": (
                        f"metadata={int(declared)}, "
                        f"target_messages={n_target}"
                    ),
                })

    return pd.DataFrame(issues)

def build_target_df(turns_df):
    target_df = (
        turns_df[turns_df["speaker"].eq("target")]
        .copy()
        .sort_values(["session_id", "target_turn_index"])
    )

    target_df["n_target_turns"] = (target_df
        .groupby("session_id")["target_turn_index"]
        .transform("max")
    )

    target_df["t"] = (target_df["target_turn_index"] / target_df["n_target_turns"])

    return target_df

def main():
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative", "CONVERSATIONS.csv",))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "generative",))
    turns_df_file = os.path.join(output_path, "TURNS.csv")
    target_df_file = os.path.join(output_path, "TARGET_TURNS.csv")
    issues_df_file = os.path.join(output_path, "ISSUES.csv")

    df = pd.read_csv(input_file)
    turns_df = build_turns_df(df)
    issues_df = validate_turns_df(turns_df)
    target_df = build_target_df(turns_df)

    turns_df.to_csv(turns_df_file, index=False)
    issues_df.to_csv(issues_df_file, index=False)
    target_df.to_csv(target_df_file, index=False)

    print("DATAFRAMES CHARACTERISTICS")
    print("Messages per speaker : \n")
    print(turns_df["speaker"].value_counts())

    print("Number of messages per session : \n")
    print(turns_df.groupby("session_id").size().describe())

    print("Number of target responses per session : \n")
    print(
        turns_df[
            turns_df["speaker"].eq("target")
        ]
        .groupby("session_id")
        .size()
        .describe()
    )

    print("Structural issues : \n")
    print(
        issues_df["issue"].value_counts()
        if not issues_df.empty
        else "No structural issues found."
    )

if __name__ == "__main__":
    main()
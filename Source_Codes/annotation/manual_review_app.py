import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "annotation"
    / "youtube_balanced_prelabel_review.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "Data"
    / "annotation"
)

PROGRESS_FILE = (
    OUTPUT_DIRECTORY
    / "youtube_manual_review_progress.csv"
)

BACKUP_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "manual_review_backups"
)

EXPORT_FILE = (
    OUTPUT_DIRECTORY
    / "youtube_manual_review_completed.csv"
)


# ============================================================
# APP SETTINGS
# ============================================================

VALID_LABELS = [
    "negative",
    "neutral",
    "positive",
    "uncertain",
]

SENTIMENT_BUTTONS = {
    "Negative": "negative",
    "Neutral": "neutral",
    "Positive": "positive",
    "Uncertain": "uncertain",
}


# ============================================================
# DATA HELPERS
# ============================================================

def ensure_directories() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    BACKUP_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_dataset() -> pd.DataFrame:
    """
    Load progress file if it exists.
    Otherwise, load the original balanced review dataset.
    """

    ensure_directories()

    if PROGRESS_FILE.exists():
        dataframe = pd.read_csv(
            PROGRESS_FILE,
            encoding="utf-8-sig",
        )
    else:
        if not INPUT_FILE.exists():
            st.error(
                f"Input file not found:\n\n{INPUT_FILE}"
            )
            st.stop()

        dataframe = pd.read_csv(
            INPUT_FILE,
            encoding="utf-8-sig",
        )

    dataframe = prepare_review_columns(
        dataframe
    )

    dataframe = dataframe.sort_values(
        by="review_sequence",
        ascending=True,
    ).reset_index(drop=True)

    return dataframe


def prepare_review_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure all required review columns exist.
    """

    dataframe = dataframe.copy()

    required_empty_columns = {
        "manual_label": "",
        "manual_review_status": "pending",
        "manual_review_notes": "",
        "final_label": "",
        "reviewed_at": "",
        "review_decision_source": "",
    }

    for column, default_value in required_empty_columns.items():
        if column not in dataframe.columns:
            dataframe[column] = default_value

        dataframe[column] = (
            dataframe[column]
            .fillna("")
            .astype(str)
        )

    if "review_sequence" not in dataframe.columns:
        dataframe["review_sequence"] = dataframe.index + 1

    return dataframe


def save_progress(
    dataframe: pd.DataFrame,
) -> None:
    """
    Save review progress immediately.
    """

    ensure_directories()

    dataframe.to_csv(
        PROGRESS_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def create_backup(
    dataframe: pd.DataFrame,
) -> Path:
    """
    Create a timestamped backup of current progress.
    """

    ensure_directories()

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_file = (
        BACKUP_DIRECTORY
        / f"youtube_manual_review_backup_{timestamp}.csv"
    )

    dataframe.to_csv(
        backup_file,
        index=False,
        encoding="utf-8-sig",
    )

    return backup_file


def export_completed_file(
    dataframe: pd.DataFrame,
) -> None:
    """
    Export the current review file.
    """

    dataframe.to_csv(
        EXPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def reset_progress() -> None:
    """
    Backup and remove current progress file.
    """

    if PROGRESS_FILE.exists():
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        backup_file = (
            BACKUP_DIRECTORY
            / f"progress_before_reset_{timestamp}.csv"
        )

        BACKUP_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            PROGRESS_FILE,
            backup_file,
        )

        PROGRESS_FILE.unlink()


# ============================================================
# REVIEW HELPERS
# ============================================================

def get_review_statistics(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Calculate review progress statistics.
    """

    reviewed_mask = (
        dataframe["manual_review_status"]
        .astype(str)
        .str.lower()
        .eq("reviewed")
    )

    reviewed_count = int(
        reviewed_mask.sum()
    )

    total_count = len(dataframe)

    pending_count = total_count - reviewed_count

    final_counts = (
        dataframe.loc[
            reviewed_mask,
            "final_label",
        ]
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .reindex(
            VALID_LABELS,
            fill_value=0,
        )
    )

    predicted_counts = (
        dataframe["predicted_label"]
        .value_counts()
        .reindex(
            [
                "negative",
                "neutral",
                "positive",
            ],
            fill_value=0,
        )
    )

    return {
        "total": total_count,
        "reviewed": reviewed_count,
        "pending": pending_count,
        "progress_percent": (
            reviewed_count / total_count * 100
            if total_count > 0
            else 0
        ),
        "final_counts": final_counts,
        "predicted_counts": predicted_counts,
    }


def find_next_pending_index(
    dataframe: pd.DataFrame,
    start_index: int = 0,
) -> int:
    """
    Find the next pending review index.
    """

    status = (
        dataframe["manual_review_status"]
        .astype(str)
        .str.lower()
    )

    pending_indices = dataframe.index[
        status.ne("reviewed")
    ].tolist()

    if not pending_indices:
        return 0

    for index in pending_indices:
        if index >= start_index:
            return int(index)

    return int(pending_indices[0])


def save_current_label(
    dataframe: pd.DataFrame,
    row_index: int,
    label: str,
    notes: str = "",
) -> pd.DataFrame:
    """
    Save the manual label for the current row.
    """

    dataframe = dataframe.copy()

    dataframe.loc[
        row_index,
        "manual_label",
    ] = label

    dataframe.loc[
        row_index,
        "final_label",
    ] = label

    dataframe.loc[
        row_index,
        "manual_review_status",
    ] = "reviewed"

    dataframe.loc[
        row_index,
        "manual_review_notes",
    ] = notes.strip()

    dataframe.loc[
        row_index,
        "reviewed_at",
    ] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    dataframe.loc[
        row_index,
        "review_decision_source",
    ] = "human_manual_review"

    save_progress(
        dataframe
    )

    return dataframe


def mark_skip(
    dataframe: pd.DataFrame,
    row_index: int,
    notes: str = "",
) -> pd.DataFrame:
    """
    Mark the current row as pending but skipped for now.
    """

    dataframe = dataframe.copy()

    dataframe.loc[
        row_index,
        "manual_review_status",
    ] = "skipped"

    dataframe.loc[
        row_index,
        "manual_review_notes",
    ] = notes.strip()

    dataframe.loc[
        row_index,
        "reviewed_at",
    ] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    dataframe.loc[
        row_index,
        "review_decision_source",
    ] = "human_skipped"

    save_progress(
        dataframe
    )

    return dataframe


def undo_current_row(
    dataframe: pd.DataFrame,
    row_index: int,
) -> pd.DataFrame:
    """
    Clear the current row review decision.
    """

    dataframe = dataframe.copy()

    dataframe.loc[
        row_index,
        "manual_label",
    ] = ""

    dataframe.loc[
        row_index,
        "final_label",
    ] = ""

    dataframe.loc[
        row_index,
        "manual_review_status",
    ] = "pending"

    dataframe.loc[
        row_index,
        "manual_review_notes",
    ] = ""

    dataframe.loc[
        row_index,
        "reviewed_at",
    ] = ""

    dataframe.loc[
        row_index,
        "review_decision_source",
    ] = ""

    save_progress(
        dataframe
    )

    return dataframe


def format_probability(
    value,
) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "0.000"


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="YouTube Sentiment Manual Review",
    page_icon="✅",
    layout="wide",
)


st.title("YouTube Sentiment Manual Review")

st.caption(
    "Review the balanced pre-labelled YouTube comments and assign the final sentiment label."
)


if "dataframe" not in st.session_state:
    st.session_state.dataframe = load_dataset()

if "current_index" not in st.session_state:
    st.session_state.current_index = find_next_pending_index(
        st.session_state.dataframe
    )


dataframe = st.session_state.dataframe

statistics = get_review_statistics(
    dataframe
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Review Progress")

    st.metric(
        "Total records",
        statistics["total"],
    )

    st.metric(
        "Reviewed",
        statistics["reviewed"],
    )

    st.metric(
        "Pending",
        statistics["pending"],
    )

    st.progress(
        statistics["progress_percent"] / 100
    )

    st.write(
        f"{statistics['progress_percent']:.2f}% completed"
    )

    st.divider()

    st.subheader("Final labels so far")

    st.dataframe(
        statistics["final_counts"]
        .rename("count")
        .reset_index()
        .rename(columns={"index": "label"}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Original predicted labels")

    st.dataframe(
        statistics["predicted_counts"]
        .rename("count")
        .reset_index()
        .rename(columns={"index": "label"}),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    selected_sequence = st.number_input(
        "Go to review sequence",
        min_value=1,
        max_value=int(statistics["total"]),
        value=int(
            dataframe.loc[
                st.session_state.current_index,
                "review_sequence",
            ]
        ),
        step=1,
    )

    if st.button("Go", use_container_width=True):
        matching_indices = dataframe.index[
            dataframe["review_sequence"]
            == selected_sequence
        ].tolist()

        if matching_indices:
            st.session_state.current_index = int(
                matching_indices[0]
            )
            st.rerun()

    if st.button("Go to next pending", use_container_width=True):
        st.session_state.current_index = find_next_pending_index(
            dataframe,
            st.session_state.current_index,
        )
        st.rerun()

    st.divider()

    if st.button("Create backup", use_container_width=True):
        backup_file = create_backup(
            dataframe
        )
        st.success(
            f"Backup created:\n{backup_file}"
        )

    if st.button("Export current reviewed file", use_container_width=True):
        export_completed_file(
            dataframe
        )
        st.success(
            f"Exported:\n{EXPORT_FILE}"
        )

    with st.expander("Danger zone"):
        st.warning(
            "Reset will backup your current progress before removing the progress file."
        )

        if st.button("Reset progress", use_container_width=True):
            reset_progress()
            st.session_state.clear()
            st.rerun()


# ============================================================
# CURRENT RECORD
# ============================================================

if statistics["reviewed"] == statistics["total"]:
    st.success(
        "All records have been reviewed."
    )

    export_completed_file(
        dataframe
    )

    st.write(
        f"Completed file saved to: `{EXPORT_FILE}`"
    )

else:
    current_index = st.session_state.current_index

    if current_index >= len(dataframe):
        current_index = find_next_pending_index(
            dataframe
        )
        st.session_state.current_index = current_index

    row = dataframe.iloc[
        current_index
    ]

    st.subheader(
        f"Record {int(row['review_sequence'])} of {statistics['total']}"
    )

    top_col_1, top_col_2, top_col_3, top_col_4 = st.columns(4)

    with top_col_1:
        st.metric(
            "Predicted label",
            str(row.get("predicted_label", "")).upper(),
        )

    with top_col_2:
        st.metric(
            "Confidence",
            format_probability(
                row.get("prediction_confidence", 0)
            ),
        )

    with top_col_3:
        st.metric(
            "Priority",
            str(row.get("review_priority", "")).upper(),
        )

    with top_col_4:
        st.metric(
            "Language",
            str(row.get("language_tag", "")).upper(),
        )

    st.divider()

    left_column, right_column = st.columns(
        [2, 1]
    )

    with left_column:
        st.markdown("### Comment to review")

        st.text_area(
            "Cleaned comment",
            value=str(
                row.get(
                    "clean_text",
                    "",
                )
            ),
            height=180,
            disabled=True,
        )

        with st.expander("Show raw original comment"):
            st.write(
                str(
                    row.get(
                        "raw_text",
                        "",
                    )
                )
            )

        notes = st.text_area(
            "Manual notes, optional",
            value=str(
                row.get(
                    "manual_review_notes",
                    "",
                )
            ),
            height=80,
        )

    with right_column:
        st.markdown("### Model probabilities")

        probability_table = pd.DataFrame(
            [
                {
                    "label": "negative",
                    "probability": format_probability(
                        row.get(
                            "negative_probability",
                            0,
                        )
                    ),
                },
                {
                    "label": "neutral",
                    "probability": format_probability(
                        row.get(
                            "neutral_probability",
                            0,
                        )
                    ),
                },
                {
                    "label": "positive",
                    "probability": format_probability(
                        row.get(
                            "positive_probability",
                            0,
                        )
                    ),
                },
            ]
        )

        st.dataframe(
            probability_table,
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("### Context")

        st.write(
            "**Theme:**",
            row.get(
                "theme",
                "",
            ),
        )

        st.write(
            "**Video title:**",
            row.get(
                "video_title",
                "",
            ),
        )

        st.write(
            "**Review reason:**",
            row.get(
                "review_reason",
                "",
            ),
        )

        st.write(
            "**Current manual status:**",
            row.get(
                "manual_review_status",
                "",
            ),
        )

        st.write(
            "**Current final label:**",
            row.get(
                "final_label",
                "",
            ),
        )

    st.divider()

    st.markdown("### Choose final sentiment label")

    button_col_1, button_col_2, button_col_3, button_col_4, button_col_5 = st.columns(5)

    with button_col_1:
        if st.button("Negative", use_container_width=True):
            updated = save_current_label(
                dataframe,
                current_index,
                "negative",
                notes,
            )
            st.session_state.dataframe = updated
            st.session_state.current_index = find_next_pending_index(
                updated,
                current_index + 1,
            )
            st.rerun()

    with button_col_2:
        if st.button("Neutral", use_container_width=True):
            updated = save_current_label(
                dataframe,
                current_index,
                "neutral",
                notes,
            )
            st.session_state.dataframe = updated
            st.session_state.current_index = find_next_pending_index(
                updated,
                current_index + 1,
            )
            st.rerun()

    with button_col_3:
        if st.button("Positive", use_container_width=True):
            updated = save_current_label(
                dataframe,
                current_index,
                "positive",
                notes,
            )
            st.session_state.dataframe = updated
            st.session_state.current_index = find_next_pending_index(
                updated,
                current_index + 1,
            )
            st.rerun()

    with button_col_4:
        if st.button("Uncertain", use_container_width=True):
            updated = save_current_label(
                dataframe,
                current_index,
                "uncertain",
                notes,
            )
            st.session_state.dataframe = updated
            st.session_state.current_index = find_next_pending_index(
                updated,
                current_index + 1,
            )
            st.rerun()

    with button_col_5:
        if st.button("Skip", use_container_width=True):
            updated = mark_skip(
                dataframe,
                current_index,
                notes,
            )
            st.session_state.dataframe = updated
            st.session_state.current_index = find_next_pending_index(
                updated,
                current_index + 1,
            )
            st.rerun()

    nav_col_1, nav_col_2, nav_col_3 = st.columns(3)

    with nav_col_1:
        if st.button("Previous", use_container_width=True):
            st.session_state.current_index = max(
                0,
                current_index - 1,
            )
            st.rerun()

    with nav_col_2:
        if st.button("Undo this record", use_container_width=True):
            updated = undo_current_row(
                dataframe,
                current_index,
            )
            st.session_state.dataframe = updated
            st.rerun()

    with nav_col_3:
        if st.button("Next", use_container_width=True):
            st.session_state.current_index = min(
                len(dataframe) - 1,
                current_index + 1,
            )
            st.rerun()


# ============================================================
# BOTTOM PREVIEW
# ============================================================

with st.expander("Preview reviewed rows"):
    reviewed_preview = dataframe[
        dataframe["manual_review_status"]
        .astype(str)
        .str.lower()
        .eq("reviewed")
    ][
        [
            "review_sequence",
            "clean_text",
            "predicted_label",
            "final_label",
            "manual_review_notes",
        ]
    ].tail(20)

    st.dataframe(
        reviewed_preview,
        use_container_width=True,
        hide_index=True,
    )
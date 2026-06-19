import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "youtube_selected_videos.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "Data"
    / "processed"
)

FINAL_OUTPUT_FILE = (
    OUTPUT_DIRECTORY
    / "youtube_selected_videos_final.csv"
)

REMOVED_OUTPUT_FILE = (
    OUTPUT_DIRECTORY
    / "youtube_removed_after_review.csv"
)

SUMMARY_OUTPUT_FILE = (
    OUTPUT_DIRECTORY
    / "youtube_final_selection_summary.csv"
)


# ============================================================
# EXACT VIDEOS TO REMOVE
# ============================================================

# Video IDs are used instead of titles because IDs are unique
# and are not affected by punctuation, spacing or title changes.

EXCLUDED_VIDEO_REASONS = {
    # --------------------------------------------------------
    # COST OF LIVING AND PUBLIC POLICY
    # --------------------------------------------------------

    "dOkqobIWL-Q": (
        "low_quality_benefit_update",
        "Click-driven payment update rather than a strong "
        "public-opinion or current-affairs source.",
    ),

    "qnPDSHAenvs": (
        "low_quality_benefit_update",
        "Benefit-update content with weak public-opinion relevance.",
    ),

    # --------------------------------------------------------
    # FESTIVALS AND NATIONAL EVENTS
    # --------------------------------------------------------

    "fi9Sgjd6x_w": (
        "promotional_or_performance_content",
        "Performance-oriented promotional content rather than "
        "a focused public-opinion discussion.",
    ),

    "jpEoOQ0ZToY": (
        "sensational_reaction_content",
        "Sensational entertainment reaction about a Raya song.",
    ),

    "xyQPxrAdVU4": (
        "event_recording",
        "Parade recording likely to contain greetings and general "
        "reactions rather than focused public opinion.",
    ),

    "MEdBHO3mu40": (
        "event_livestream",
        "National Day livestream with weak thematic opinion focus.",
    ),

    "dyj7DO8b3Q8": (
        "food_vlog_content",
        "Food-vlog content rather than an applied public-opinion source.",
    ),

    # --------------------------------------------------------
    # TECHNOLOGY AND DIGITAL SERVICES
    # --------------------------------------------------------

    "c03GG_MwbKg": (
        "foreign_usage_test",
        "Tests a Malaysian e-wallet outside Malaysia and is not focused "
        "on Malaysian public opinion.",
    ),

    # --------------------------------------------------------
    # TOURISM AND LIFESTYLE
    # --------------------------------------------------------

    "OwrE_BcOYH4": (
        "promotional_content",
        "Airline promotional livery video rather than public opinion.",
    ),

    "QWeCGpj7SMY": (
        "tourism_promotion",
        "Visit Malaysia promotional content rather than public opinion.",
    ),

    "FXAd9tPGt1A": (
        "foreign_travel_vlog",
        "Foreign travel-cost vlog rather than Malaysian public opinion.",
    ),

    "XZOtAikt6U8": (
        "foreign_travel_vlog",
        "Foreign creator travel commentary with weak Malaysian "
        "public-opinion relevance.",
    ),

    "JzswKQHF_Ds": (
        "foreign_travel_vlog",
        "Kuala Lumpur itinerary and lifestyle vlog.",
    ),

    "MwNBXexBdl8": (
        "entertainment_content",
        "Concert highlights rather than public-opinion analysis.",
    ),

    # --------------------------------------------------------
    # TRANSPORT AND PUBLIC SERVICES
    # --------------------------------------------------------

    "XFCAHpZj-pI": (
        "train_recording",
        "Train announcement recording rather than a public-service "
        "opinion discussion.",
    ),

    "277bY-hvbKw": (
        "foreign_language_office_video",
        "Foreign-language Touch 'n Go office video with weak Malaysian "
        "public-opinion relevance.",
    ),
}


# ============================================================
# EXPECTED PROJECT THEMES
# ============================================================

THEMES = [
    "festivals_national_events",
    "cost_of_living_public_policy",
    "transport_public_services",
    "technology_digital_services",
    "tourism_lifestyle_events",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_integer(value: Any) -> int:
    """
    Convert values safely into integers.
    """

    try:
        if value is None or pd.isna(value):
            return 0

        return int(float(value))

    except (TypeError, ValueError):
        return 0


def validate_input(dataframe: pd.DataFrame) -> None:
    """
    Confirm that the selected-video CSV contains the required fields.
    """

    required_columns = [
        "video_id",
        "video_title",
        "channel_title",
        "video_url",
        "comment_count",
        "published_at",
        "manual_selected_theme",
        "selection_status",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The input CSV is missing required columns: "
            + ", ".join(missing_columns)
        )


def apply_final_review(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the automatically selected videos into:

    1. the final approved list
    2. the videos removed after quality review
    """

    dataframe = dataframe.copy()

    dataframe["video_id"] = (
        dataframe["video_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe["comment_count"] = (
        dataframe["comment_count"]
        .apply(safe_integer)
    )

    dataframe["removed_after_review"] = (
        dataframe["video_id"]
        .isin(EXCLUDED_VIDEO_REASONS)
    )

    removed = dataframe[
        dataframe["removed_after_review"]
    ].copy()

    final_selected = dataframe[
        ~dataframe["removed_after_review"]
    ].copy()

    # Add detailed review reasons to removed videos.
    removed["final_review_reason"] = (
        removed["video_id"]
        .map(
            lambda video_id: EXCLUDED_VIDEO_REASONS.get(
                video_id,
                ("other", ""),
            )[0]
        )
    )

    removed["final_review_notes"] = (
        removed["video_id"]
        .map(
            lambda video_id: EXCLUDED_VIDEO_REASONS.get(
                video_id,
                ("other", ""),
            )[1]
        )
    )

    removed["manual_relevance"] = "no"
    removed["selection_status"] = "removed_after_review"
    removed["rejection_reason"] = removed[
        "final_review_reason"
    ]

    # Mark retained videos as final selections.
    final_selected["manual_relevance"] = "yes"
    final_selected["selection_status"] = "final_selected"
    final_selected["rejection_reason"] = ""
    final_selected["final_review_reason"] = ""
    final_selected["final_review_notes"] = (
        "Approved for YouTube comment collection after automated "
        "screening and final metadata-based quality review."
    )

    # Remove the temporary review flag.
    final_selected.drop(
        columns=["removed_after_review"],
        inplace=True,
        errors="ignore",
    )

    removed.drop(
        columns=["removed_after_review"],
        inplace=True,
        errors="ignore",
    )

    return final_selected, removed


def create_summary(
    final_selected: pd.DataFrame,
    removed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a compact summary of the final selection by theme.
    """

    summary_rows = []

    for theme in THEMES:
        selected_theme = final_selected[
            final_selected["manual_selected_theme"]
            == theme
        ]

        removed_theme = removed[
            removed["manual_selected_theme"]
            == theme
        ]

        summary_rows.append(
            {
                "theme": theme,
                "final_selected_videos": len(
                    selected_theme
                ),
                "removed_after_review": len(
                    removed_theme
                ),
                "available_comments": int(
                    selected_theme["comment_count"].sum()
                ),
                "unique_channels": int(
                    selected_theme["channel_title"].nunique()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    total_row = pd.DataFrame(
        [
            {
                "theme": "TOTAL",
                "final_selected_videos": len(
                    final_selected
                ),
                "removed_after_review": len(
                    removed
                ),
                "available_comments": int(
                    final_selected["comment_count"].sum()
                ),
                "unique_channels": int(
                    final_selected["channel_title"].nunique()
                ),
            }
        ]
    )

    return pd.concat(
        [
            summary,
            total_row,
        ],
        ignore_index=True,
    )


def sort_outputs(
    final_selected: pd.DataFrame,
    removed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort the output files into a readable order.
    """

    selected_sort_columns = [
        column
        for column in [
            "manual_selected_theme",
            "automated_quality_score",
            "comment_count",
        ]
        if column in final_selected.columns
    ]

    selected_ascending = [
        True if column == "manual_selected_theme" else False
        for column in selected_sort_columns
    ]

    if selected_sort_columns:
        final_selected = (
            final_selected
            .sort_values(
                by=selected_sort_columns,
                ascending=selected_ascending,
            )
            .reset_index(drop=True)
        )

    removed_sort_columns = [
        column
        for column in [
            "manual_selected_theme",
            "final_review_reason",
            "comment_count",
        ]
        if column in removed.columns
    ]

    removed_ascending = [
        True,
        True,
        False,
    ][:len(removed_sort_columns)]

    if removed_sort_columns:
        removed = (
            removed
            .sort_values(
                by=removed_sort_columns,
                ascending=removed_ascending,
            )
            .reset_index(drop=True)
        )

    return final_selected, removed


def save_outputs(
    final_selected: pd.DataFrame,
    removed: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """
    Save all final-review CSV files.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_selected.to_csv(
        FINAL_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    removed.to_csv(
        REMOVED_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def print_summary(
    original: pd.DataFrame,
    final_selected: pd.DataFrame,
    removed: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """
    Print the final quality-review results.
    """

    print("\n" + "=" * 78)
    print("FINAL YOUTUBE VIDEO SELECTION CREATED")
    print("=" * 78)

    print(
        f"Original automatically selected videos : "
        f"{len(original):,}"
    )

    print(
        f"Removed after final review             : "
        f"{len(removed):,}"
    )

    print(
        f"Final approved videos                  : "
        f"{len(final_selected):,}"
    )

    print(
        f"Available comments in final videos     : "
        f"{final_selected['comment_count'].sum():,}"
    )

    print(
        f"Unique channels in final selection     : "
        f"{final_selected['channel_title'].nunique():,}"
    )

    print("\nFinal videos by theme:")

    display_summary = summary[
        summary["theme"] != "TOTAL"
    ][
        [
            "theme",
            "final_selected_videos",
            "removed_after_review",
            "available_comments",
            "unique_channels",
        ]
    ]

    print(
        display_summary.to_string(
            index=False
        )
    )

    if not removed.empty:
        print("\nRemoved videos:")

        display_columns = [
            "video_title",
            "channel_title",
            "manual_selected_theme",
            "final_review_reason",
        ]

        print(
            removed[
                display_columns
            ].to_string(
                index=False
            )
        )

    print("\nSaved files:")
    print(f"1. {FINAL_OUTPUT_FILE}")
    print(f"2. {REMOVED_OUTPUT_FILE}")
    print(f"3. {SUMMARY_OUTPUT_FILE}")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> None:
    """
    Create the final approved YouTube video collection list.
    """

    try:
        if not INPUT_FILE.exists():
            raise FileNotFoundError(
                f"Input file was not found:\n"
                f"{INPUT_FILE}\n\n"
                "Run auto_select_youtube_videos.py first."
            )

        print("=" * 78)
        print("CREATING FINAL YOUTUBE VIDEO SELECTION")
        print("=" * 78)

        print(f"Input file: {INPUT_FILE}")

        original = pd.read_csv(
            INPUT_FILE,
            encoding="utf-8-sig",
        )

        validate_input(original)

        duplicate_count = int(
            original["video_id"].duplicated().sum()
        )

        if duplicate_count > 0:
            print(
                f"\nWarning: {duplicate_count} duplicate video IDs "
                "were removed."
            )

            original = (
                original
                .drop_duplicates(
                    subset=["video_id"],
                    keep="first",
                )
                .reset_index(drop=True)
            )

        final_selected, removed = apply_final_review(
            original
        )

        final_selected, removed = sort_outputs(
            final_selected,
            removed,
        )

        summary = create_summary(
            final_selected,
            removed,
        )

        save_outputs(
            final_selected,
            removed,
            summary,
        )

        print_summary(
            original,
            final_selected,
            removed,
            summary,
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        print(f"\nError:\n{error}")
        sys.exit(1)

    except KeyboardInterrupt:
        print(
            "\nFinal selection process stopped by the user."
        )
        sys.exit(1)

    except Exception as error:
        print(
            f"\nUnexpected error: "
            f"{type(error).__name__}: {error}"
        )
        raise


if __name__ == "__main__":
    main()
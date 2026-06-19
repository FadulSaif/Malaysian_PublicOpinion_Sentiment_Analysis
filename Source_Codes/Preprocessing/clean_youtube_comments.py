import html
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import emoji
import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect


# ============================================================
# REPRODUCIBILITY
# ============================================================

# Makes langdetect deterministic across repeated runs.
DetectorFactory.seed = 42


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "raw"
    / "youtube_comments_raw.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "Data"
    / "interim"
)

CLEANED_OUTPUT_FILE = (
    OUTPUT_DIRECTORY
    / "youtube_comments_cleaned.csv"
)

REJECTED_OUTPUT_FILE = (
    OUTPUT_DIRECTORY
    / "youtube_comments_rejected.csv"
)

SUMMARY_OUTPUT_FILE = (
    OUTPUT_DIRECTORY
    / "youtube_cleaning_summary.csv"
)


# ============================================================
# CLEANING SETTINGS
# ============================================================

MINIMUM_WORD_COUNT = 4

MINIMUM_CHARACTER_COUNT = 12

MAXIMUM_CHARACTER_COUNT = 1000

MAXIMUM_URL_COUNT = 2

MAXIMUM_HASHTAG_COUNT = 8

MAXIMUM_MENTION_COUNT = 5

MAXIMUM_REPEATED_CHARACTER_RUN = 8


# ============================================================
# LANGUAGE INDICATORS
# ============================================================

MALAY_INDICATORS = {
    "yang",
    "dan",
    "ini",
    "itu",
    "saya",
    "aku",
    "kita",
    "kami",
    "mereka",
    "tak",
    "tidak",
    "bukan",
    "boleh",
    "dah",
    "sudah",
    "nak",
    "mahu",
    "pun",
    "lah",
    "kan",
    "je",
    "juga",
    "memang",
    "sangat",
    "bagus",
    "teruk",
    "mahal",
    "murah",
    "kerajaan",
    "rakyat",
    "harga",
    "barang",
    "bantuan",
    "subsidi",
    "masalah",
    "jalan",
    "kereta",
    "tren",
    "bas",
    "orang",
    "malaysia",
    "malaysian",
    "kenapa",
    "macam",
    "kalau",
    "sebab",
    "dengan",
    "untuk",
    "daripada",
    "dekat",
    "kat",
    "sedap",
    "gila",
    "best",
    "ramai",
    "lagi",
}

ENGLISH_INDICATORS = {
    "the",
    "and",
    "this",
    "that",
    "is",
    "are",
    "was",
    "were",
    "i",
    "we",
    "they",
    "you",
    "not",
    "very",
    "really",
    "good",
    "bad",
    "great",
    "terrible",
    "expensive",
    "cheap",
    "government",
    "people",
    "price",
    "prices",
    "support",
    "service",
    "problem",
    "issue",
    "train",
    "bus",
    "traffic",
    "tourism",
    "event",
    "why",
    "because",
    "should",
    "could",
    "would",
    "better",
    "worse",
    "today",
    "again",
    "please",
    "thank",
    "thanks",
    "amazing",
    "disappointed",
}


# ============================================================
# REGULAR EXPRESSIONS
# ============================================================

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

MENTION_PATTERN = re.compile(
    r"(?<!\w)@[A-Za-z0-9_.-]+"
)

HASHTAG_PATTERN = re.compile(
    r"(?<!\w)#[A-Za-z0-9_]+"
)

WHITESPACE_PATTERN = re.compile(
    r"\s+"
)

REPEATED_PUNCTUATION_PATTERN = re.compile(
    r"([!?.,])\1{2,}"
)

REPEATED_CHARACTER_PATTERN = re.compile(
    r"(.)\1{7,}",
    flags=re.IGNORECASE,
)

NON_WORD_PATTERN = re.compile(
    r"[^\w\s]",
    flags=re.UNICODE,
)

LATIN_CHARACTER_PATTERN = re.compile(
    r"[A-Za-z]"
)

CJK_PATTERN = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF]"
)

ARABIC_PATTERN = re.compile(
    r"[\u0600-\u06FF]"
)

CYRILLIC_PATTERN = re.compile(
    r"[\u0400-\u04FF]"
)

THAI_PATTERN = re.compile(
    r"[\u0E00-\u0E7F]"
)

DEVANAGARI_PATTERN = re.compile(
    r"[\u0900-\u097F]"
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_integer(value: Any) -> int:
    """
    Convert a value safely into an integer.
    """

    try:
        if value is None or pd.isna(value):
            return 0

        return int(float(value))

    except (TypeError, ValueError):
        return 0


def normalize_unicode(value: Any) -> str:
    """
    Normalize Unicode while preserving multilingual characters
    and emojis.
    """

    if value is None or pd.isna(value):
        return ""

    text = html.unescape(str(value))

    return unicodedata.normalize(
        "NFKC",
        text,
    )


def remove_control_characters(text: str) -> str:
    """
    Remove invisible control characters but preserve normal
    whitespace and emojis.
    """

    cleaned_characters = []

    for character in text:
        category = unicodedata.category(character)

        if category.startswith("C"):
            if character in {
                "\n",
                "\r",
                "\t",
            }:
                cleaned_characters.append(" ")

            continue

        cleaned_characters.append(character)

    return "".join(cleaned_characters)


def replace_urls(text: str) -> str:
    """
    Replace URLs with a standard placeholder.
    """

    return URL_PATTERN.sub(
        " URL ",
        text,
    )


def replace_mentions(text: str) -> str:
    """
    Replace user mentions with a standard placeholder.
    """

    return MENTION_PATTERN.sub(
        " USER_MENTION ",
        text,
    )


def normalize_hashtags(text: str) -> str:
    """
    Keep hashtag words while removing the # symbol.

    Example:
        #HargaBarang -> HargaBarang
    """

    return HASHTAG_PATTERN.sub(
        lambda match: " " + match.group(0)[1:] + " ",
        text,
    )


def normalize_repeated_punctuation(text: str) -> str:
    """
    Reduce excessive repeated punctuation while preserving
    sentiment intensity.

    Example:
        !!!!! -> !!
    """

    return REPEATED_PUNCTUATION_PATTERN.sub(
        lambda match: match.group(1) * 2,
        text,
    )


def normalize_whitespace(text: str) -> str:
    """
    Replace repeated whitespace with one space.
    """

    return WHITESPACE_PATTERN.sub(
        " ",
        text,
    ).strip()


def create_clean_text(raw_text: Any) -> str:
    """
    Create a cleaned text representation without removing
    sentiment-bearing emojis or punctuation.
    """

    text = normalize_unicode(raw_text)
    text = remove_control_characters(text)
    text = replace_urls(text)
    text = replace_mentions(text)
    text = normalize_hashtags(text)
    text = normalize_repeated_punctuation(text)
    text = normalize_whitespace(text)

    return text


def create_normalized_duplicate_key(text: Any) -> str:
    """
    Create a stricter normalized text key for duplicate detection.

    This removes URLs, mentions, punctuation, case differences,
    and repeated whitespace.
    """

    normalized = normalize_unicode(text).lower()
    normalized = URL_PATTERN.sub(" ", normalized)
    normalized = MENTION_PATTERN.sub(" ", normalized)
    normalized = HASHTAG_PATTERN.sub(
        lambda match: " " + match.group(0)[1:] + " ",
        normalized,
    )

    normalized = NON_WORD_PATTERN.sub(
        " ",
        normalized,
    )

    normalized = WHITESPACE_PATTERN.sub(
        " ",
        normalized,
    )

    return normalized.strip()


def tokenize_words(text: str) -> list[str]:
    """
    Extract words and number-containing tokens.
    """

    return re.findall(
        r"\b[\w'-]+\b",
        text.lower(),
        flags=re.UNICODE,
    )


def count_meaningful_words(text: str) -> int:
    """
    Count tokens containing at least one letter or digit.
    """

    tokens = tokenize_words(text)

    return sum(
        1
        for token in tokens
        if any(
            character.isalnum()
            for character in token
        )
    )


def contains_meaningful_alphanumeric(text: str) -> bool:
    """
    Return True if text contains at least one letter or number.
    """

    return any(
        character.isalnum()
        for character in text
    )


def count_emojis(text: str) -> int:
    """
    Count Unicode emojis in text.
    """

    return emoji.emoji_count(text)


def is_emoji_only(text: str) -> bool:
    """
    Detect comments made only of emojis, punctuation, and spaces.
    """

    if not text:
        return False

    text_without_emojis = emoji.replace_emoji(
        text,
        replace="",
    )

    text_without_symbols = re.sub(
        r"[\W_]+",
        "",
        text_without_emojis,
        flags=re.UNICODE,
    )

    return (
        count_emojis(text) > 0
        and text_without_symbols == ""
    )


def contains_excessive_character_repetition(
    text: str,
) -> bool:
    """
    Detect spam-like repeated characters.

    Examples:
        aaaaaaaaaa
        hahahahahahahahaha may not always match because it repeats
        groups rather than one character.
    """

    return bool(
        REPEATED_CHARACTER_PATTERN.search(text)
    )


def calculate_script_counts(
    text: str,
) -> dict[str, int]:
    """
    Count characters from major writing systems.
    """

    return {
        "latin": len(
            LATIN_CHARACTER_PATTERN.findall(text)
        ),
        "cjk": len(
            CJK_PATTERN.findall(text)
        ),
        "arabic": len(
            ARABIC_PATTERN.findall(text)
        ),
        "cyrillic": len(
            CYRILLIC_PATTERN.findall(text)
        ),
        "thai": len(
            THAI_PATTERN.findall(text)
        ),
        "devanagari": len(
            DEVANAGARI_PATTERN.findall(text)
        ),
    }


# ============================================================
# LANGUAGE CLASSIFICATION
# ============================================================

def detect_language_with_library(
    text: str,
) -> str:
    """
    Run langdetect as a supporting signal.

    The result is not treated as final because short Malay and
    English social-media comments are often misclassified.
    """

    if len(text) < 15:
        return "unknown"

    try:
        return detect(text)

    except LangDetectException:
        return "unknown"


def classify_language_scope(
    text: str,
) -> tuple[str, str, int, int]:
    """
    Classify the comment as:

    - BM
    - EN
    - MIXED
    - OTHER
    - UNKNOWN

    Returns:
        language_tag,
        detected_language,
        malay_indicator_count,
        english_indicator_count
    """

    tokens = set(
        tokenize_words(text)
    )

    malay_count = len(
        tokens.intersection(
            MALAY_INDICATORS
        )
    )

    english_count = len(
        tokens.intersection(
            ENGLISH_INDICATORS
        )
    )

    detected_language = detect_language_with_library(
        text
    )

    script_counts = calculate_script_counts(
        text
    )

    non_latin_count = (
        script_counts["cjk"]
        + script_counts["cyrillic"]
        + script_counts["thai"]
        + script_counts["devanagari"]
    )

    latin_count = script_counts["latin"]

    # Primarily unsupported script.
    if (
        non_latin_count >= 5
        and non_latin_count > latin_count
    ):
        return (
            "OTHER",
            detected_language,
            malay_count,
            english_count,
        )

    # Arabic may appear inside Malay religious expressions, so only
    # reject when the Arabic script clearly dominates the comment.
    if (
        script_counts["arabic"] >= 8
        and script_counts["arabic"] > latin_count
    ):
        return (
            "OTHER",
            detected_language,
            malay_count,
            english_count,
        )

    # Strong evidence of code-switching.
    if (
        malay_count >= 2
        and english_count >= 2
    ):
        return (
            "MIXED",
            detected_language,
            malay_count,
            english_count,
        )

    # Some mixed comments are short and contain fewer indicators.
    if (
        malay_count >= 1
        and english_count >= 1
        and detected_language in {
            "ms",
            "id",
            "en",
        }
    ):
        return (
            "MIXED",
            detected_language,
            malay_count,
            english_count,
        )

    if malay_count > english_count:
        return (
            "BM",
            detected_language,
            malay_count,
            english_count,
        )

    if english_count > malay_count:
        return (
            "EN",
            detected_language,
            malay_count,
            english_count,
        )

    # langdetect often returns Indonesian for informal Malay.
    if detected_language in {
        "ms",
        "id",
    }:
        return (
            "BM",
            detected_language,
            malay_count,
            english_count,
        )

    if detected_language == "en":
        return (
            "EN",
            detected_language,
            malay_count,
            english_count,
        )

    if latin_count > 0:
        return (
            "UNKNOWN",
            detected_language,
            malay_count,
            english_count,
        )

    return (
        "OTHER",
        detected_language,
        malay_count,
        english_count,
    )


# ============================================================
# SPAM AND QUALITY RULES
# ============================================================

def detect_spam_reasons(
    raw_text: str,
    clean_text: str,
) -> list[str]:
    """
    Return a list of possible spam indicators.
    """

    reasons = []

    url_count = len(
        URL_PATTERN.findall(raw_text)
    )

    mention_count = len(
        MENTION_PATTERN.findall(raw_text)
    )

    hashtag_count = len(
        HASHTAG_PATTERN.findall(raw_text)
    )

    if url_count > MAXIMUM_URL_COUNT:
        reasons.append(
            "too_many_urls"
        )

    if mention_count > MAXIMUM_MENTION_COUNT:
        reasons.append(
            "too_many_mentions"
        )

    if hashtag_count > MAXIMUM_HASHTAG_COUNT:
        reasons.append(
            "too_many_hashtags"
        )

    if contains_excessive_character_repetition(
        clean_text
    ):
        reasons.append(
            "excessive_character_repetition"
        )

    words = tokenize_words(clean_text)

    if len(words) >= 8:
        most_common_word_count = Counter(
            words
        ).most_common(1)[0][1]

        if (
            most_common_word_count
            / len(words)
            >= 0.60
        ):
            reasons.append(
                "repeated_word_spam"
            )

    return reasons


def determine_rejection_reason(
    row: pd.Series,
) -> str:
    """
    Determine the primary rejection reason for one comment.
    """

    clean_text = row["clean_text"]

    if not clean_text:
        return "empty_after_cleaning"

    if row["emoji_only"]:
        return "emoji_only"

    if not row["contains_alphanumeric"]:
        return "symbol_only"

    if (
        row["character_count"]
        < MINIMUM_CHARACTER_COUNT
    ):
        return "too_few_characters"

    if (
        row["word_count"]
        < MINIMUM_WORD_COUNT
    ):
        return "too_few_words"

    if (
        row["character_count"]
        > MAXIMUM_CHARACTER_COUNT
    ):
        return "excessively_long_comment"

    if row["spam_flag"]:
        return "spam_pattern"

    if row["language_tag"] == "OTHER":
        return "unsupported_language"

    if row["normalized_duplicate"]:
        return "normalized_duplicate"

    return ""


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

def load_raw_comments() -> pd.DataFrame:
    """
    Load and validate the raw YouTube comment file.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Raw YouTube comment file was not found:\n"
            f"{INPUT_FILE}"
        )

    dataframe = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
    )

    required_columns = [
        "record_id",
        "comment_id",
        "video_id",
        "video_title",
        "channel_title",
        "theme",
        "raw_text",
        "comment_published_at",
        "like_count",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The raw YouTube file is missing columns: "
            + ", ".join(missing_columns)
        )

    dataframe["comment_id"] = (
        dataframe["comment_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe["raw_text"] = (
        dataframe["raw_text"]
        .fillna("")
        .astype(str)
    )

    dataframe["like_count"] = (
        dataframe["like_count"]
        .apply(safe_integer)
    )

    # Comment IDs should already be unique, but keep this safeguard.
    dataframe.drop_duplicates(
        subset=["comment_id"],
        keep="first",
        inplace=True,
    )

    dataframe.reset_index(
        drop=True,
        inplace=True,
    )

    return dataframe


def add_cleaning_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add cleaned text and quality-control features.
    """

    dataframe = dataframe.copy()

    dataframe["clean_text"] = (
        dataframe["raw_text"]
        .apply(create_clean_text)
    )

    dataframe["normalized_text_key"] = (
        dataframe["clean_text"]
        .apply(
            create_normalized_duplicate_key
        )
    )

    dataframe["word_count"] = (
        dataframe["clean_text"]
        .apply(count_meaningful_words)
    )

    dataframe["character_count"] = (
        dataframe["clean_text"]
        .str.len()
    )

    dataframe["emoji_count"] = (
        dataframe["clean_text"]
        .apply(count_emojis)
    )

    dataframe["emoji_only"] = (
        dataframe["clean_text"]
        .apply(is_emoji_only)
    )

    dataframe["contains_alphanumeric"] = (
        dataframe["clean_text"]
        .apply(
            contains_meaningful_alphanumeric
        )
    )

    dataframe["url_count"] = (
        dataframe["raw_text"]
        .apply(
            lambda text: len(
                URL_PATTERN.findall(text)
            )
        )
    )

    dataframe["mention_count"] = (
        dataframe["raw_text"]
        .apply(
            lambda text: len(
                MENTION_PATTERN.findall(text)
            )
        )
    )

    dataframe["hashtag_count"] = (
        dataframe["raw_text"]
        .apply(
            lambda text: len(
                HASHTAG_PATTERN.findall(text)
            )
        )
    )

    language_results = (
        dataframe["clean_text"]
        .apply(classify_language_scope)
    )

    dataframe["language_tag"] = (
        language_results.apply(
            lambda result: result[0]
        )
    )

    dataframe["detected_language"] = (
        language_results.apply(
            lambda result: result[1]
        )
    )

    dataframe["malay_indicator_count"] = (
        language_results.apply(
            lambda result: result[2]
        )
    )

    dataframe["english_indicator_count"] = (
        language_results.apply(
            lambda result: result[3]
        )
    )

    spam_results = dataframe.apply(
        lambda row: detect_spam_reasons(
            raw_text=row["raw_text"],
            clean_text=row["clean_text"],
        ),
        axis=1,
    )

    dataframe["spam_reasons"] = (
        spam_results.apply(
            lambda reasons: " | ".join(
                reasons
            )
        )
    )

    dataframe["spam_flag"] = (
        dataframe["spam_reasons"]
        .str.len()
        .gt(0)
    )

    # Mark duplicate normalized text. Keep the first occurrence.
    dataframe["normalized_duplicate"] = (
        dataframe["normalized_text_key"]
        .duplicated(
            keep="first"
        )
        & dataframe["normalized_text_key"].ne("")
    )

    dataframe["rejection_reason"] = (
        dataframe.apply(
            determine_rejection_reason,
            axis=1,
        )
    )

    dataframe["cleaning_status"] = (
        dataframe["rejection_reason"]
        .apply(
            lambda reason: (
                "accepted"
                if reason == ""
                else "rejected"
            )
        )
    )

    return dataframe


def split_outputs(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split accepted and rejected comments.
    """

    accepted = dataframe[
        dataframe["cleaning_status"]
        == "accepted"
    ].copy()

    rejected = dataframe[
        dataframe["cleaning_status"]
        == "rejected"
    ].copy()

    accepted.sort_values(
        by=[
            "theme",
            "video_id",
            "comment_published_at",
        ],
        ascending=[
            True,
            True,
            False,
        ],
        inplace=True,
    )

    rejected.sort_values(
        by=[
            "rejection_reason",
            "theme",
            "video_id",
        ],
        ascending=[
            True,
            True,
            True,
        ],
        inplace=True,
    )

    accepted.reset_index(
        drop=True,
        inplace=True,
    )

    rejected.reset_index(
        drop=True,
        inplace=True,
    )

    return accepted, rejected


def create_cleaning_summary(
    original: pd.DataFrame,
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create summary rows for overall quality, rejection reasons,
    languages, and themes.
    """

    rows = [
        {
            "summary_type": "overall",
            "category": "raw_comments",
            "count": len(original),
        },
        {
            "summary_type": "overall",
            "category": "accepted_comments",
            "count": len(accepted),
        },
        {
            "summary_type": "overall",
            "category": "rejected_comments",
            "count": len(rejected),
        },
        {
            "summary_type": "overall",
            "category": "acceptance_rate_percent",
            "count": round(
                (
                    len(accepted)
                    / len(original)
                    * 100
                )
                if len(original) > 0
                else 0,
                2,
            ),
        },
    ]

    for reason, count in (
        rejected["rejection_reason"]
        .value_counts()
        .items()
    ):
        rows.append(
            {
                "summary_type": "rejection_reason",
                "category": reason,
                "count": int(count),
            }
        )

    for language, count in (
        accepted["language_tag"]
        .value_counts()
        .items()
    ):
        rows.append(
            {
                "summary_type": "accepted_language",
                "category": language,
                "count": int(count),
            }
        )

    for theme, count in (
        accepted["theme"]
        .value_counts()
        .items()
    ):
        rows.append(
            {
                "summary_type": "accepted_theme",
                "category": theme,
                "count": int(count),
            }
        )

    return pd.DataFrame(
        rows
    )


def save_results(
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """
    Save all preprocessing outputs.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    accepted.to_csv(
        CLEANED_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    rejected.to_csv(
        REJECTED_OUTPUT_FILE,
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
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
) -> None:
    """
    Print preprocessing results.
    """

    print("\n" + "=" * 78)
    print("YOUTUBE COMMENT CLEANING COMPLETED")
    print("=" * 78)

    print(
        f"Raw comments                 : "
        f"{len(original):,}"
    )

    print(
        f"Accepted comments            : "
        f"{len(accepted):,}"
    )

    print(
        f"Rejected comments            : "
        f"{len(rejected):,}"
    )

    acceptance_rate = (
        len(accepted)
        / len(original)
        * 100
        if len(original) > 0
        else 0
    )

    print(
        f"Acceptance rate              : "
        f"{acceptance_rate:.2f}%"
    )

    print("\nAccepted comments by language:")

    if accepted.empty:
        print("No accepted comments.")
    else:
        print(
            accepted["language_tag"]
            .value_counts()
            .to_string()
        )

    print("\nAccepted comments by theme:")

    if accepted.empty:
        print("No accepted comments.")
    else:
        print(
            accepted["theme"]
            .value_counts()
            .to_string()
        )

    print("\nRejected comments by reason:")

    if rejected.empty:
        print("No rejected comments.")
    else:
        print(
            rejected["rejection_reason"]
            .value_counts()
            .to_string()
        )

    print("\nSaved files:")
    print(f"1. {CLEANED_OUTPUT_FILE}")
    print(f"2. {REJECTED_OUTPUT_FILE}")
    print(f"3. {SUMMARY_OUTPUT_FILE}")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> None:
    """
    Run YouTube comment preprocessing.
    """

    try:
        print("=" * 78)
        print("PREPROCESSING RAW YOUTUBE COMMENTS")
        print("=" * 78)
        print(f"Input file: {INPUT_FILE}")

        raw_comments = load_raw_comments()

        processed = add_cleaning_features(
            raw_comments
        )

        accepted, rejected = split_outputs(
            processed
        )

        summary = create_cleaning_summary(
            original=raw_comments,
            accepted=accepted,
            rejected=rejected,
        )

        save_results(
            accepted=accepted,
            rejected=rejected,
            summary=summary,
        )

        print_summary(
            original=raw_comments,
            accepted=accepted,
            rejected=rejected,
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        print(f"\nError:\n{error}")
        sys.exit(1)

    except KeyboardInterrupt:
        print(
            "\nPreprocessing stopped by the user."
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
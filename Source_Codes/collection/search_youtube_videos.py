import html
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

OUTPUT_DIRECTORY = PROJECT_ROOT / "Data" / "raw"

CHECKPOINT_FILE = (
    OUTPUT_DIRECTORY / "youtube_search_checkpoint.csv"
)

COMPLETED_QUERIES_FILE = (
    OUTPUT_DIRECTORY / "youtube_search_completed.json"
)

ALL_OUTPUT_FILE = (
    OUTPUT_DIRECTORY / "youtube_video_candidates_all.csv"
)

FILTERED_OUTPUT_FILE = (
    OUTPUT_DIRECTORY / "youtube_video_candidates_filtered.csv"
)


# ============================================================
# COLLECTION PERIOD
# ============================================================

PUBLISHED_AFTER = "2025-01-01T00:00:00Z"

# publishedBefore is exclusive.
# This includes all videos published on 17 June 2026.
PUBLISHED_BEFORE = "2026-06-18T00:00:00Z"

REGION_CODE = "MY"


# ============================================================
# QUOTA-SAFE SEARCH SETTINGS
# ============================================================

# Maximum allowed by search.list.
MAX_RESULTS_PER_QUERY = 50

# 5 themes × 5 queries = only 25 search.list requests.
MINIMUM_COMMENT_COUNT = 30
MINIMUM_DURATION_SECONDS = 61

SAFE_SEARCH = "moderate"


# ============================================================
# MALAYSIA-SPECIFIC SEARCH QUERIES
# ============================================================

# Do not add too many queries.
# Every query below consumes one daily search call.

THEME_QUERIES = {
    "festivals_national_events": [
        '"Hari Raya" Malaysia public reaction',
        '"balik kampung" Malaysia traffic',
        '"Ramadan bazaar" Malaysia prices',
        '"Merdeka celebration" Malaysia',
        '"Thaipusam" OR "Deepavali" Malaysia',
    ],

    "cost_of_living_public_policy": [
        '"harga barang" Malaysia',
        '"kos sara hidup" Malaysia',
        '"subsidi petrol" Malaysia',
        '"Sumbangan Tunai Rahmah" Malaysia',
        '"minimum wage" OR inflation Malaysia',
    ],

    "transport_public_services": [
        '"Rapid KL" disruption',
        '"KTM Berhad" delay Malaysia',
        '"LRT" OR "MRT" Malaysia problem',
        '"Touch n Go" Malaysia problem',
        '"KL traffic" public reaction',
    ],

    "technology_digital_services": [
        '"CelcomDigi" OR Maxis Malaysia problem',
        '"U Mobile" OR Unifi Malaysia outage',
        '"Touch n Go eWallet" Malaysia',
        '"online banking" Malaysia problem',
        '"5G Malaysia" OR "MyDigital ID"',
    ],

    "tourism_lifestyle_events": [
        '"Visit Malaysia" public reaction',
        '"Melaka tourism" review',
        '"Penang tourism" review',
        '"Langkawi tourism" review',
        '"Malaysia concert" OR event public reaction',
    ],
}


# ============================================================
# MALAYSIA RELEVANCE TERMS
# ============================================================

MALAYSIA_TERMS = [
    "malaysia",
    "malaysian",
    "kuala lumpur",
    "putrajaya",
    "selangor",
    "johor",
    "johor bahru",
    "penang",
    "pulau pinang",
    "melaka",
    "malacca",
    "sabah",
    "sarawak",
    "perak",
    "kedah",
    "kelantan",
    "terengganu",
    "pahang",
    "negeri sembilan",
    "perlis",
    "labuan",
    "langkawi",
    "klang valley",
    "rapid kl",
    "rapidkl",
    "ktm",
    "ktm berhad",
    "lrt",
    "mrt",
    "monorail",
    "touch n go",
    "touch 'n go",
    "tng",
    "celcomdigi",
    "celcom",
    "digi",
    "maxis",
    "u mobile",
    "unifi",
    "astro awani",
    "bernama",
    "buletin tv3",
    "tv3 malaysia",
    "sinar harian",
    "berita harian",
    "harian metro",
    "the star malaysia",
    "free malaysia today",
    "malaysiakini",
    "kinitv",
    "hari raya",
    "aidilfitri",
    "balik kampung",
    "bazaar ramadan",
    "bazar ramadan",
    "ramadan bazaar",
    "merdeka",
    "madani",
    "bajet malaysia",
    "rahmah",
    "sumbangan tunai rahmah",
    "kos sara hidup",
    "harga barang",
    "ringgit",
    "myr",
    "mydigital id",
    "jpj malaysia",
    "tourism malaysia",
    "visit malaysia",
]


# ============================================================
# TRUSTED MALAYSIAN CHANNELS
# ============================================================

TRUSTED_CHANNEL_TERMS = [
    "astro awani",
    "bernama",
    "buletin tv3",
    "tv3 malaysia",
    "sinar harian",
    "berita harian",
    "harian metro",
    "malaysiakini",
    "kinitv",
    "free malaysia today",
    "the star",
    "says",
    "world of buzz",
    "rapid kl",
    "tourism malaysia",
    "malaysia airlines",
    "airasia",
    "radio televisyen malaysia",
    "rtm",
]


# ============================================================
# OBVIOUSLY IRRELEVANT CONTENT
# ============================================================

IRRELEVANT_TERMS = [
    "official music video",
    "lyrics video",
    "lyric video",
    "karaoke",
    "dance cover",
    "reaction to kpop",
    "k-pop",
    "kpop",
    "anime",
    "gameplay",
    "gaming live",
    "minecraft",
    "roblox",
    "fortnite",
    "movie trailer",
    "episode full",
    "full episode",
    "cartoon episode",
    "nursery rhyme",
    "kids song",
    "asmr",
    "prank video",
]


# ============================================================
# CUSTOM EXCEPTION
# ============================================================

class SearchQuotaExceededError(Exception):
    """
    Raised when the YouTube daily search quota is exhausted.
    """


# ============================================================
# API SETUP
# ============================================================

def load_api_key() -> str:
    """
    Load the API key from the project's .env file.
    """

    if not ENV_PATH.exists():
        raise FileNotFoundError(
            f"The .env file was not found:\n{ENV_PATH}"
        )

    load_dotenv(
        dotenv_path=ENV_PATH,
        override=True,
    )

    api_key = os.getenv(
        "YOUTUBE_API_KEY",
        "",
    ).strip()

    if not api_key:
        raise ValueError(
            "YOUTUBE_API_KEY is missing or empty. "
            "Add it to the project .env file."
        )

    return api_key


def create_youtube_client():
    """
    Create a YouTube Data API v3 client.
    """

    return build(
        serviceName="youtube",
        version="v3",
        developerKey=load_api_key(),
        cache_discovery=False,
    )


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_text(value: Any) -> str:
    """
    Normalize text for keyword matching.
    """

    if value is None or pd.isna(value):
        return ""

    text = html.unescape(str(value))
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def safe_integer(value: Any) -> int:
    """
    Convert API values safely to integers.
    """

    try:
        if value is None or pd.isna(value):
            return 0

        return int(float(value))

    except (TypeError, ValueError):
        return 0


def create_batches(
    values: list[str],
    batch_size: int = 50,
) -> list[list[str]]:
    """
    Split values into fixed-size batches.
    """

    return [
        values[index:index + batch_size]
        for index in range(0, len(values), batch_size)
    ]


def parse_youtube_duration(duration: Any) -> int:
    """
    Convert YouTube ISO 8601 duration into seconds.

    Examples:
        PT45S   -> 45
        PT2M5S  -> 125
        PT1H2M  -> 3720
    """

    if duration is None or pd.isna(duration):
        return 0

    duration = str(duration).strip()

    pattern = re.compile(
        r"P"
        r"(?:(?P<days>\d+)D)?"
        r"(?:T"
        r"(?:(?P<hours>\d+)H)?"
        r"(?:(?P<minutes>\d+)M)?"
        r"(?:(?P<seconds>\d+)S)?"
        r")?"
    )

    match = pattern.fullmatch(duration)

    if not match:
        return 0

    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)

    return (
        days * 86400
        + hours * 3600
        + minutes * 60
        + seconds
    )


def find_matching_terms(
    text: str,
    terms: list[str],
) -> list[str]:
    """
    Return all terms found in normalized text.
    """

    return sorted(
        {
            term
            for term in terms
            if term in text
        }
    )


# ============================================================
# HTTP ERROR HANDLING
# ============================================================

def extract_http_error_reason(
    error: HttpError,
) -> tuple[str, str]:
    """
    Extract the error reason and message without displaying
    the request URL, which may contain the API key.
    """

    reason = "unknown"
    message = "YouTube API request failed."

    try:
        payload = json.loads(
            error.content.decode("utf-8")
        )

        error_object = payload.get(
            "error",
            {},
        )

        message = error_object.get(
            "message",
            message,
        )

        errors = error_object.get(
            "errors",
            [],
        )

        if errors:
            reason = errors[0].get(
                "reason",
                "unknown",
            )

    except (
        AttributeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        pass

    return reason, message


def execute_request_safely(
    request,
    maximum_attempts: int = 3,
):
    """
    Execute an API request.

    Temporary server failures are retried.
    Quota and credential errors are not retried.
    """

    for attempt in range(1, maximum_attempts + 1):
        try:
            return request.execute()

        except HttpError as error:
            status = getattr(
                error.resp,
                "status",
                None,
            )

            reason, message = extract_http_error_reason(
                error
            )

            reason_lower = reason.lower()
            message_lower = message.lower()

            quota_terms = [
                "quotaexceeded",
                "ratelimitexceeded",
                "dailylimitexceeded",
                "quota exceeded",
                "search queries per day",
            ]

            if any(
                term in reason_lower
                or term in message_lower
                for term in quota_terms
            ):
                raise SearchQuotaExceededError(
                    message
                ) from error

            credential_terms = [
                "keyexpired",
                "keyinvalid",
                "api key expired",
                "api key not valid",
            ]

            if any(
                term in reason_lower
                or term in message_lower
                for term in credential_terms
            ):
                raise ValueError(
                    f"YouTube credential error: {message}"
                ) from error

            temporary_statuses = {
                429,
                500,
                502,
                503,
                504,
            }

            if (
                status in temporary_statuses
                and attempt < maximum_attempts
            ):
                delay = 2 ** attempt

                print(
                    f"\nTemporary API error. "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)
                continue

            raise RuntimeError(
                f"YouTube API error "
                f"(HTTP {status}, reason={reason}): "
                f"{message}"
            ) from error

    raise RuntimeError(
        "API request failed after retries."
    )


# ============================================================
# CHECKPOINT FUNCTIONS
# ============================================================

def make_query_key(
    theme: str,
    query: str,
) -> str:
    """
    Create a unique key for one completed search.
    """

    return f"{theme}||{query}"


def load_completed_queries() -> set[str]:
    """
    Load the list of already completed search queries.
    """

    if not COMPLETED_QUERIES_FILE.exists():
        return set()

    try:
        with open(
            COMPLETED_QUERIES_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            values = json.load(file)

        return set(values)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return set()


def save_completed_queries(
    completed_queries: set[str],
) -> None:
    """
    Save completed searches after every successful request.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        COMPLETED_QUERIES_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            sorted(completed_queries),
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_checkpoint_records() -> list[dict[str, Any]]:
    """
    Load search records previously saved to the checkpoint.
    """

    if not CHECKPOINT_FILE.exists():
        return []

    dataframe = pd.read_csv(
        CHECKPOINT_FILE,
        encoding="utf-8-sig",
    )

    return dataframe.to_dict(
        orient="records"
    )


def save_checkpoint_records(
    records: list[dict[str, Any]],
) -> None:
    """
    Save all current search records after each request.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(records)

    dataframe.to_csv(
        CHECKPOINT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# SEARCH AND VIDEO DETAILS
# ============================================================

def search_videos(
    youtube,
    query: str,
    theme: str,
) -> list[dict[str, Any]]:
    """
    Perform exactly one search.list request.
    """

    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=MAX_RESULTS_PER_QUERY,
        publishedAfter=PUBLISHED_AFTER,
        publishedBefore=PUBLISHED_BEFORE,
        regionCode=REGION_CODE,
        safeSearch=SAFE_SEARCH,
        order="relevance",
    )

    response = execute_request_safely(
        request
    )

    records = []

    for item in response.get("items", []):
        video_id = (
            item.get("id", {})
            .get("videoId")
        )

        if not video_id:
            continue

        snippet = item.get(
            "snippet",
            {},
        )

        records.append(
            {
                "video_id": video_id,
                "video_title": html.unescape(
                    snippet.get("title", "")
                ).strip(),
                "video_description": html.unescape(
                    snippet.get("description", "")
                ).strip(),
                "channel_id": snippet.get(
                    "channelId",
                    "",
                ),
                "channel_title": html.unescape(
                    snippet.get("channelTitle", "")
                ).strip(),
                "published_at": snippet.get(
                    "publishedAt",
                    "",
                ),
                "discovery_theme": theme,
                "query_keyword": query,
                "video_url": (
                    "https://www.youtube.com/watch?v="
                    f"{video_id}"
                ),
            }
        )

    return records


def get_video_details(
    youtube,
    video_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Retrieve details for up to 50 video IDs.
    """

    if not video_ids:
        return {}

    request = youtube.videos().list(
        part="snippet,statistics,status,contentDetails",
        id=",".join(video_ids),
        maxResults=50,
    )

    response = execute_request_safely(
        request
    )

    details = {}

    for item in response.get("items", []):
        video_id = item.get(
            "id",
            "",
        )

        snippet = item.get(
            "snippet",
            {},
        )

        statistics = item.get(
            "statistics",
            {},
        )

        status = item.get(
            "status",
            {},
        )

        content_details = item.get(
            "contentDetails",
            {},
        )

        details[video_id] = {
            "view_count": safe_integer(
                statistics.get("viewCount")
            ),
            "like_count": safe_integer(
                statistics.get("likeCount")
            ),
            "comment_count": safe_integer(
                statistics.get("commentCount")
            ),
            "privacy_status": status.get(
                "privacyStatus",
                "",
            ),
            "made_for_kids": bool(
                status.get("madeForKids", False)
            ),
            "embeddable": bool(
                status.get("embeddable", False)
            ),
            "license": status.get(
                "license",
                "",
            ),
            "video_category_id": snippet.get(
                "categoryId",
                "",
            ),
            "default_language": snippet.get(
                "defaultLanguage",
                "",
            ),
            "default_audio_language": snippet.get(
                "defaultAudioLanguage",
                "",
            ),
            "duration": content_details.get(
                "duration",
                "",
            ),
            "caption_available": content_details.get(
                "caption",
                "false",
            ),
            "definition": content_details.get(
                "definition",
                "",
            ),
        }

    return details


# ============================================================
# DATA ENRICHMENT AND DEDUPLICATION
# ============================================================

def add_video_details(
    youtube,
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add statistics and content metadata.
    """

    video_ids = (
        dataframe["video_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    batches = create_batches(
        video_ids,
        batch_size=50,
    )

    all_details: dict[str, dict[str, Any]] = {}

    print(
        f"\nRetrieving details for "
        f"{len(video_ids):,} unique videos..."
    )

    for batch in tqdm(
        batches,
        desc="Video detail requests",
    ):
        details = get_video_details(
            youtube,
            batch,
        )

        all_details.update(details)

        time.sleep(0.1)

    details_dataframe = pd.DataFrame.from_dict(
        all_details,
        orient="index",
    )

    details_dataframe.index.name = "video_id"
    details_dataframe.reset_index(
        inplace=True
    )

    merged = dataframe.merge(
        details_dataframe,
        on="video_id",
        how="left",
    )

    for column in [
        "view_count",
        "like_count",
        "comment_count",
    ]:
        merged[column] = (
            merged[column]
            .fillna(0)
            .apply(safe_integer)
        )

    return merged


def combine_duplicate_discoveries(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep one row per video while preserving all themes and
    queries that discovered it.
    """

    dataframe = dataframe.copy()

    for column in [
        "discovery_theme",
        "query_keyword",
    ]:
        dataframe[column] = (
            dataframe[column]
            .fillna("")
            .astype(str)
        )

    aggregation_rules = {
        "video_title": "first",
        "video_description": "first",
        "channel_id": "first",
        "channel_title": "first",
        "published_at": "first",

        "discovery_theme": lambda values: " | ".join(
            sorted(
                {
                    value
                    for value in values
                    if value
                }
            )
        ),

        "query_keyword": lambda values: " | ".join(
            sorted(
                {
                    value
                    for value in values
                    if value
                }
            )
        ),

        "video_url": "first",
        "view_count": "first",
        "like_count": "first",
        "comment_count": "first",
        "privacy_status": "first",
        "made_for_kids": "first",
        "embeddable": "first",
        "license": "first",
        "video_category_id": "first",
        "default_language": "first",
        "default_audio_language": "first",
        "duration": "first",
        "caption_available": "first",
        "definition": "first",
    }

    return (
        dataframe
        .groupby(
            "video_id",
            as_index=False,
        )
        .agg(aggregation_rules)
    )


# ============================================================
# RELEVANCE FILTERING
# ============================================================

def build_combined_video_text(
    row: pd.Series,
) -> str:
    """
    Combine actual metadata only.

    query_keyword is intentionally excluded.
    """

    return normalize_text(
        " ".join(
            [
                str(row.get("video_title", "")),
                str(row.get("video_description", "")),
                str(row.get("channel_title", "")),
            ]
        )
    )


def calculate_candidate_score(
    row: pd.Series,
) -> int:
    """
    Assign an automatic review-priority score.
    """

    score = 0

    comment_count = safe_integer(
        row.get("comment_count")
    )

    duration_seconds = safe_integer(
        row.get("duration_seconds")
    )

    if bool(row.get("malaysia_keyword_match")):
        score += 3

    if bool(row.get("trusted_channel_match")):
        score += 3

    if comment_count >= 30:
        score += 1

    if comment_count >= 100:
        score += 1

    if comment_count >= 500:
        score += 1

    if duration_seconds > 60:
        score += 1

    if duration_seconds >= 180:
        score += 1

    if normalize_text(
        row.get("privacy_status")
    ) == "public":
        score += 1

    if bool(row.get("made_for_kids")):
        score -= 3

    if bool(row.get("irrelevant_term_match")):
        score -= 5

    if bool(row.get("is_short_video")):
        score -= 2

    return score


def add_review_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add automatic filtering and manual-review fields.
    """

    dataframe = dataframe.copy()

    dataframe["published_datetime"] = pd.to_datetime(
        dataframe["published_at"],
        utc=True,
        errors="coerce",
    )

    start_timestamp = pd.Timestamp(
        PUBLISHED_AFTER
    )

    end_timestamp = pd.Timestamp(
        PUBLISHED_BEFORE
    )

    dataframe["date_within_scope"] = (
        dataframe["published_datetime"]
        .between(
            start_timestamp,
            end_timestamp,
            inclusive="left",
        )
    )

    dataframe["duration_seconds"] = (
        dataframe["duration"]
        .fillna("")
        .apply(parse_youtube_duration)
    )

    dataframe["is_short_video"] = (
        dataframe["duration_seconds"]
        < MINIMUM_DURATION_SECONDS
    )

    dataframe["has_enough_comments"] = (
        dataframe["comment_count"]
        >= MINIMUM_COMMENT_COUNT
    )

    # Malaysia matching: title, description and channel only.
    dataframe["_video_metadata_text"] = dataframe.apply(
        build_combined_video_text,
        axis=1,
    )

    dataframe["malaysia_matching_terms"] = (
        dataframe["_video_metadata_text"]
        .apply(
            lambda text: " | ".join(
                find_matching_terms(
                    text,
                    MALAYSIA_TERMS,
                )
            )
        )
    )

    dataframe["malaysia_keyword_match"] = (
        dataframe["malaysia_matching_terms"]
        .str.len()
        .gt(0)
    )

    # Trusted channels: channel title only.
    dataframe["_normalized_channel_title"] = (
        dataframe["channel_title"]
        .fillna("")
        .apply(normalize_text)
    )

    dataframe["trusted_channel_matching_terms"] = (
        dataframe["_normalized_channel_title"]
        .apply(
            lambda text: " | ".join(
                find_matching_terms(
                    text,
                    TRUSTED_CHANNEL_TERMS,
                )
            )
        )
    )

    dataframe["trusted_channel_match"] = (
        dataframe["trusted_channel_matching_terms"]
        .str.len()
        .gt(0)
    )

    dataframe["irrelevant_matching_terms"] = (
        dataframe["_video_metadata_text"]
        .apply(
            lambda text: " | ".join(
                find_matching_terms(
                    text,
                    IRRELEVANT_TERMS,
                )
            )
        )
    )

    dataframe["irrelevant_term_match"] = (
        dataframe["irrelevant_matching_terms"]
        .str.len()
        .gt(0)
    )

    dataframe["made_for_kids"] = (
        dataframe["made_for_kids"]
        .fillna(False)
        .astype(bool)
    )

    dataframe["privacy_status"] = (
        dataframe["privacy_status"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    dataframe["automatic_candidate"] = (
        dataframe["date_within_scope"]
        & dataframe["has_enough_comments"]
        & dataframe["malaysia_keyword_match"]
        & ~dataframe["is_short_video"]
        & ~dataframe["made_for_kids"]
        & ~dataframe["irrelevant_term_match"]
        & dataframe["privacy_status"].eq("public")
    )

    dataframe["candidate_score"] = dataframe.apply(
        calculate_candidate_score,
        axis=1,
    )

    dataframe["manual_relevance"] = ""
    dataframe["manual_selected_theme"] = ""
    dataframe["selection_status"] = "pending"
    dataframe["rejection_reason"] = ""
    dataframe["review_notes"] = ""

    dataframe["collected_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    dataframe.drop(
        columns=[
            "_video_metadata_text",
            "_normalized_channel_title",
        ],
        inplace=True,
        errors="ignore",
    )

    return dataframe


# ============================================================
# SAVING AND REPORTING
# ============================================================

def save_final_results(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save complete and filtered candidate files.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_candidates = dataframe.sort_values(
        by=[
            "automatic_candidate",
            "candidate_score",
            "comment_count",
            "published_datetime",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    filtered_candidates = all_candidates[
        all_candidates["automatic_candidate"]
    ].copy()

    filtered_candidates.reset_index(
        drop=True,
        inplace=True,
    )

    all_candidates.drop(
        columns=["published_datetime"],
        inplace=True,
        errors="ignore",
    )

    filtered_candidates.drop(
        columns=["published_datetime"],
        inplace=True,
        errors="ignore",
    )

    all_candidates.to_csv(
        ALL_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    filtered_candidates.to_csv(
        FILTERED_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return all_candidates, filtered_candidates


def print_summary(
    all_candidates: pd.DataFrame,
    filtered_candidates: pd.DataFrame,
    completed_queries: set[str],
    quota_stopped: bool,
) -> None:
    """
    Print the final discovery summary.
    """

    print("\n" + "=" * 75)
    print("YOUTUBE VIDEO DISCOVERY SUMMARY")
    print("=" * 75)

    print(
        f"Completed search queries          : "
        f"{len(completed_queries):,}/25"
    )

    print(
        f"All unique candidate videos       : "
        f"{len(all_candidates):,}"
    )

    print(
        f"Automatic filtered candidates     : "
        f"{len(filtered_candidates):,}"
    )

    print(
        f"Videos with >= {MINIMUM_COMMENT_COUNT} comments   : "
        f"{all_candidates['has_enough_comments'].sum():,}"
    )

    print(
        f"Short videos                      : "
        f"{all_candidates['is_short_video'].sum():,}"
    )

    print(
        f"Malaysia metadata matches         : "
        f"{all_candidates['malaysia_keyword_match'].sum():,}"
    )

    print(
        f"Trusted-channel matches           : "
        f"{all_candidates['trusted_channel_match'].sum():,}"
    )

    print(
        f"Available comments in filtered set: "
        f"{filtered_candidates['comment_count'].sum():,}"
    )

    if quota_stopped:
        print(
            "\nThe script stopped because the daily search "
            "quota was reached."
        )

        print(
            "Progress was saved. Run the same script after "
            "the quota resets; completed queries will be skipped."
        )

    print("\nSaved files:")
    print(f"1. {CHECKPOINT_FILE}")
    print(f"2. {COMPLETED_QUERIES_FILE}")
    print(f"3. {ALL_OUTPUT_FILE}")
    print(f"4. {FILTERED_OUTPUT_FILE}")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> None:
    """
    Run the quota-safe YouTube discovery process.
    """

    try:
        OUTPUT_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        youtube = create_youtube_client()

        all_records = load_checkpoint_records()
        completed_queries = load_completed_queries()

        total_configured_queries = sum(
            len(queries)
            for queries in THEME_QUERIES.values()
        )

        remaining_queries = []

        for theme, queries in THEME_QUERIES.items():
            for query in queries:
                query_key = make_query_key(
                    theme,
                    query,
                )

                if query_key not in completed_queries:
                    remaining_queries.append(
                        (
                            theme,
                            query,
                            query_key,
                        )
                    )

        print("=" * 75)
        print("QUOTA-SAFE MALAYSIA YOUTUBE VIDEO DISCOVERY")
        print("=" * 75)

        print(
            f"Collection period : "
            f"{PUBLISHED_AFTER}"
        )

        print(
            f"                  to "
            f"{PUBLISHED_BEFORE}"
        )

        print(
            f"Configured queries: "
            f"{total_configured_queries}"
        )

        print(
            f"Already completed : "
            f"{len(completed_queries)}"
        )

        print(
            f"Remaining today   : "
            f"{len(remaining_queries)}"
        )

        quota_stopped = False

        progress_bar = tqdm(
            total=len(remaining_queries),
            desc="YouTube searches",
        )

        for (
            theme,
            query,
            query_key,
        ) in remaining_queries:
            try:
                records = search_videos(
                    youtube=youtube,
                    query=query,
                    theme=theme,
                )

                all_records.extend(records)

                completed_queries.add(
                    query_key
                )

                save_checkpoint_records(
                    all_records
                )

                save_completed_queries(
                    completed_queries
                )

                progress_bar.update(1)

                time.sleep(0.25)

            except SearchQuotaExceededError as error:
                quota_stopped = True

                print(
                    "\nDaily search quota reached:"
                )

                print(str(error))

                break

        progress_bar.close()

        if not all_records:
            print(
                "\nNo search records are available."
            )

            print(
                "If quota is exhausted, wait for the reset "
                "and run this script again."
            )

            return

        dataframe = pd.DataFrame(
            all_records
        )

        dataframe.drop_duplicates(
            subset=[
                "video_id",
                "discovery_theme",
                "query_keyword",
            ],
            inplace=True,
        )

        print(
            f"\nSaved search discoveries: "
            f"{len(dataframe):,}"
        )

        dataframe = add_video_details(
            youtube=youtube,
            dataframe=dataframe,
        )

        dataframe = combine_duplicate_discoveries(
            dataframe
        )

        print(
            f"Unique videos after deduplication: "
            f"{len(dataframe):,}"
        )

        dataframe = add_review_columns(
            dataframe
        )

        (
            all_candidates,
            filtered_candidates,
        ) = save_final_results(
            dataframe
        )

        print_summary(
            all_candidates=all_candidates,
            filtered_candidates=filtered_candidates,
            completed_queries=completed_queries,
            quota_stopped=quota_stopped,
        )

    except FileNotFoundError as error:
        print(f"\nFile error:\n{error}")
        sys.exit(1)

    except ValueError as error:
        print(f"\nConfiguration error:\n{error}")
        sys.exit(1)

    except KeyboardInterrupt:
        print(
            "\nThe process was stopped by the user. "
            "Completed search progress remains saved."
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
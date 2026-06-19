import hashlib
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

INPUT_VIDEO_FILE = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "youtube_selected_videos_final.csv"
)

RAW_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "Data"
    / "raw"
)

LOG_DIRECTORY = (
    PROJECT_ROOT
    / "logs"
)

RAW_COMMENTS_FILE = (
    RAW_OUTPUT_DIRECTORY
    / "youtube_comments_raw.csv"
)

CHECKPOINT_FILE = (
    RAW_OUTPUT_DIRECTORY
    / "youtube_comments_checkpoint.csv"
)

PROGRESS_FILE = (
    RAW_OUTPUT_DIRECTORY
    / "youtube_comments_progress.json"
)

COLLECTION_LOG_FILE = (
    LOG_DIRECTORY
    / "youtube_comment_collection_log.csv"
)

SUMMARY_FILE = (
    RAW_OUTPUT_DIRECTORY
    / "youtube_comment_collection_summary.csv"
)


# ============================================================
# COLLECTION SETTINGS
# ============================================================

# The final source file contains 22 videos.
# 22 × 120 gives a theoretical maximum of 2,640 comments.
MAX_COMMENTS_PER_VIDEO = 120

# Overall stopping target.
TARGET_TOTAL_COMMENTS = 2500

# YouTube allows up to 100 comments per API page.
MAX_RESULTS_PER_PAGE = 100

# "time" avoids ranking only by popular/relevant comments.
COMMENT_ORDER = "time"

TEXT_FORMAT = "plainText"

REQUEST_DELAY_SECONDS = 0.20

MAX_RETRY_ATTEMPTS = 3


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class YouTubeQuotaError(Exception):
    """
    Raised when a YouTube API quota limit is reached.
    """


class CommentsDisabledError(Exception):
    """
    Raised when comments are disabled for a video.
    """


class VideoUnavailableError(Exception):
    """
    Raised when a selected video is unavailable.
    """


# ============================================================
# API SETUP
# ============================================================

def load_api_key() -> str:
    """
    Load the API key from the project .env file.
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
            "YOUTUBE_API_KEY is missing or empty in .env."
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

def safe_integer(value: Any) -> int:
    """
    Convert a value safely to an integer.
    """

    try:
        if value is None or pd.isna(value):
            return 0

        return int(float(value))

    except (TypeError, ValueError):
        return 0


def clean_line_breaks(value: Any) -> str:
    """
    Preserve raw wording while replacing line breaks with spaces.
    """

    if value is None:
        return ""

    text = str(value)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def hash_author_channel(
    author_channel_id: str,
) -> str:
    """
    Store an anonymized author identifier instead of the raw
    YouTube author channel ID.
    """

    if not author_channel_id:
        return ""

    return hashlib.sha256(
        author_channel_id.encode("utf-8")
    ).hexdigest()[:20]


def utc_now() -> str:
    """
    Return the current UTC time in ISO format.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# ERROR HANDLING
# ============================================================

def parse_http_error(
    error: HttpError,
) -> tuple[int | str, str, str]:
    """
    Extract safe details from a Google API error.
    """

    status = getattr(
        error.resp,
        "status",
        "unknown",
    )

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

    return status, reason, message


def execute_request_safely(request):
    """
    Execute a YouTube API request.

    Temporary server failures are retried. Quota, credential,
    disabled-comment and unavailable-video errors are not retried.
    """

    for attempt in range(
        1,
        MAX_RETRY_ATTEMPTS + 1,
    ):
        try:
            return request.execute()

        except HttpError as error:
            status, reason, message = parse_http_error(
                error
            )

            reason_lower = reason.lower()
            message_lower = message.lower()

            if (
                "commentsdisabled" in reason_lower
                or "comments are disabled" in message_lower
            ):
                raise CommentsDisabledError(
                    message
                ) from error

            if (
                "videonotfound" in reason_lower
                or "video not found" in message_lower
            ):
                raise VideoUnavailableError(
                    message
                ) from error

            quota_terms = [
                "quotaexceeded",
                "ratelimitexceeded",
                "dailylimitexceeded",
                "quota exceeded",
            ]

            if any(
                term in reason_lower
                or term in message_lower
                for term in quota_terms
            ):
                raise YouTubeQuotaError(
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
                and attempt < MAX_RETRY_ATTEMPTS
            ):
                delay = 2 ** attempt

                print(
                    f"\nTemporary YouTube API error. "
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
        "YouTube request failed after all retries."
    )


# ============================================================
# INPUT AND CHECKPOINTS
# ============================================================

def load_selected_videos() -> pd.DataFrame:
    """
    Load and validate the final selected-video list.
    """

    if not INPUT_VIDEO_FILE.exists():
        raise FileNotFoundError(
            f"The final selected-video file was not found:\n"
            f"{INPUT_VIDEO_FILE}"
        )

    videos = pd.read_csv(
        INPUT_VIDEO_FILE,
        encoding="utf-8-sig",
    )

    required_columns = [
        "video_id",
        "video_title",
        "channel_title",
        "video_url",
        "published_at",
        "manual_selected_theme",
        "comment_count",
    ]

    missing = [
        column
        for column in required_columns
        if column not in videos.columns
    ]

    if missing:
        raise ValueError(
            "The selected-video file is missing: "
            + ", ".join(missing)
        )

    videos["video_id"] = (
        videos["video_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    videos = (
        videos[
            videos["video_id"].ne("")
        ]
        .drop_duplicates(
            subset=["video_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return videos


def load_existing_comments() -> list[dict[str, Any]]:
    """
    Load previously collected comments for resuming.
    """

    source_file = None

    if CHECKPOINT_FILE.exists():
        source_file = CHECKPOINT_FILE

    elif RAW_COMMENTS_FILE.exists():
        source_file = RAW_COMMENTS_FILE

    if source_file is None:
        return []

    dataframe = pd.read_csv(
        source_file,
        encoding="utf-8-sig",
    )

    return dataframe.to_dict(
        orient="records"
    )


def load_progress() -> dict[str, Any]:
    """
    Load page and video progress from JSON.
    """

    if not PROGRESS_FILE.exists():
        return {
            "completed_video_ids": [],
            "video_page_tokens": {},
            "video_collected_counts": {},
        }

    try:
        with open(
            PROGRESS_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            progress = json.load(file)

        progress.setdefault(
            "completed_video_ids",
            [],
        )

        progress.setdefault(
            "video_page_tokens",
            {},
        )

        progress.setdefault(
            "video_collected_counts",
            {},
        )

        return progress

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {
            "completed_video_ids": [],
            "video_page_tokens": {},
            "video_collected_counts": {},
        }


def save_progress(
    progress: dict[str, Any],
) -> None:
    """
    Save collection progress after every API page.
    """

    RAW_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        PROGRESS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            progress,
            file,
            indent=2,
            ensure_ascii=False,
        )


def save_comment_checkpoint(
    records: list[dict[str, Any]],
) -> None:
    """
    Save all collected comments after each page.
    """

    RAW_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        records
    )

    if not dataframe.empty:
        dataframe.drop_duplicates(
            subset=["comment_id"],
            keep="first",
            inplace=True,
        )

    dataframe.to_csv(
        CHECKPOINT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# COMMENT EXTRACTION
# ============================================================

def extract_comment_record(
    thread: dict[str, Any],
    video: pd.Series,
) -> dict[str, Any] | None:
    """
    Convert one commentThread response item into one CSV record.
    """

    thread_snippet = thread.get(
        "snippet",
        {},
    )

    top_level_comment = thread_snippet.get(
        "topLevelComment",
        {},
    )

    comment_id = str(
        top_level_comment.get(
            "id",
            "",
        )
    ).strip()

    comment_snippet = top_level_comment.get(
        "snippet",
        {},
    )

    raw_text = clean_line_breaks(
        comment_snippet.get(
            "textOriginal",
            "",
        )
    )

    if not comment_id or not raw_text:
        return None

    author_channel = comment_snippet.get(
        "authorChannelId",
        {},
    )

    author_channel_id = ""

    if isinstance(author_channel, dict):
        author_channel_id = str(
            author_channel.get(
                "value",
                "",
            )
        )

    return {
        "record_id": (
            f"YT_{comment_id}"
        ),
        "platform": "YouTube",
        "content_type": "top_level_comment",
        "comment_id": comment_id,
        "thread_id": str(
            thread.get("id", "")
        ),
        "video_id": str(
            video.get("video_id", "")
        ),
        "video_title": str(
            video.get("video_title", "")
        ),
        "channel_title": str(
            video.get("channel_title", "")
        ),
        "video_url": str(
            video.get("video_url", "")
        ),
        "video_published_at": str(
            video.get("published_at", "")
        ),
        "theme": str(
            video.get(
                "manual_selected_theme",
                "",
            )
        ),
        "raw_text": raw_text,
        "author_hash": hash_author_channel(
            author_channel_id
        ),
        "comment_published_at": str(
            comment_snippet.get(
                "publishedAt",
                "",
            )
        ),
        "comment_updated_at": str(
            comment_snippet.get(
                "updatedAt",
                "",
            )
        ),
        "like_count": safe_integer(
            comment_snippet.get(
                "likeCount"
            )
        ),
        "reply_count": safe_integer(
            thread_snippet.get(
                "totalReplyCount"
            )
        ),
        "can_reply": bool(
            thread_snippet.get(
                "canReply",
                False,
            )
        ),
        "is_public": bool(
            thread_snippet.get(
                "isPublic",
                True,
            )
        ),
        "viewer_rating": str(
            comment_snippet.get(
                "viewerRating",
                "none",
            )
        ),
        "moderation_status": str(
            comment_snippet.get(
                "moderationStatus",
                "",
            )
        ),
        "collected_at": utc_now(),
        "collection_status": "raw",
        "rejection_reason": "",
    }


# ============================================================
# PAGE COLLECTION
# ============================================================

def request_comment_page(
    youtube,
    video_id: str,
    page_token: str | None,
):
    """
    Retrieve one page of top-level comment threads.
    """

    parameters: dict[str, Any] = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": MAX_RESULTS_PER_PAGE,
        "order": COMMENT_ORDER,
        "textFormat": TEXT_FORMAT,
    }

    if page_token:
        parameters["pageToken"] = page_token

    request = youtube.commentThreads().list(
        **parameters
    )

    return execute_request_safely(
        request
    )


def collect_video_comments(
    youtube,
    video: pd.Series,
    all_records: list[dict[str, Any]],
    existing_comment_ids: set[str],
    progress: dict[str, Any],
) -> tuple[int, str]:
    """
    Collect up to MAX_COMMENTS_PER_VIDEO comments for one video.

    Returns:
        collected_count, final_status
    """

    video_id = str(
        video["video_id"]
    ).strip()

    previously_collected = safe_integer(
        progress[
            "video_collected_counts"
        ].get(
            video_id,
            0,
        )
    )

    page_token = progress[
        "video_page_tokens"
    ].get(
        video_id
    )

    collected_for_video = previously_collected

    while (
        collected_for_video
        < MAX_COMMENTS_PER_VIDEO
        and len(existing_comment_ids)
        < TARGET_TOTAL_COMMENTS
    ):
        response = request_comment_page(
            youtube=youtube,
            video_id=video_id,
            page_token=page_token,
        )

        items = response.get(
            "items",
            [],
        )

        if not items:
            progress[
                "completed_video_ids"
            ].append(
                video_id
            )

            progress[
                "video_page_tokens"
            ].pop(
                video_id,
                None,
            )

            progress[
                "video_collected_counts"
            ][video_id] = collected_for_video

            save_progress(progress)

            return (
                collected_for_video,
                "completed_no_more_comments",
            )

        for thread in items:
            if (
                collected_for_video
                >= MAX_COMMENTS_PER_VIDEO
                or len(existing_comment_ids)
                >= TARGET_TOTAL_COMMENTS
            ):
                break

            record = extract_comment_record(
                thread=thread,
                video=video,
            )

            if record is None:
                continue

            comment_id = record[
                "comment_id"
            ]

            if comment_id in existing_comment_ids:
                continue

            all_records.append(
                record
            )

            existing_comment_ids.add(
                comment_id
            )

            collected_for_video += 1

        next_page_token = response.get(
            "nextPageToken"
        )

        progress[
            "video_collected_counts"
        ][video_id] = collected_for_video

        if next_page_token:
            progress[
                "video_page_tokens"
            ][video_id] = next_page_token
        else:
            progress[
                "video_page_tokens"
            ].pop(
                video_id,
                None,
            )

        save_comment_checkpoint(
            all_records
        )

        save_progress(
            progress
        )

        if not next_page_token:
            if (
                video_id
                not in progress[
                    "completed_video_ids"
                ]
            ):
                progress[
                    "completed_video_ids"
                ].append(
                    video_id
                )

            save_progress(
                progress
            )

            return (
                collected_for_video,
                "completed_all_available",
            )

        if (
            collected_for_video
            >= MAX_COMMENTS_PER_VIDEO
        ):
            if (
                video_id
                not in progress[
                    "completed_video_ids"
                ]
            ):
                progress[
                    "completed_video_ids"
                ].append(
                    video_id
                )

            progress[
                "video_page_tokens"
            ].pop(
                video_id,
                None,
            )

            save_progress(
                progress
            )

            return (
                collected_for_video,
                "completed_video_cap",
            )

        page_token = next_page_token

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    return (
        collected_for_video,
        "stopped_at_global_target",
    )


# ============================================================
# LOGGING AND FINAL OUTPUT
# ============================================================

def save_collection_log(
    log_rows: list[dict[str, Any]],
) -> None:
    """
    Save the per-video collection log.
    """

    LOG_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        log_rows
    ).to_csv(
        COLLECTION_LOG_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def save_final_comments(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Deduplicate and save the final raw YouTube comment file.
    """

    dataframe = pd.DataFrame(
        records
    )

    if dataframe.empty:
        dataframe.to_csv(
            RAW_COMMENTS_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        return dataframe

    dataframe.drop_duplicates(
        subset=["comment_id"],
        keep="first",
        inplace=True,
    )

    dataframe.sort_values(
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

    dataframe.reset_index(
        drop=True,
        inplace=True,
    )

    dataframe.to_csv(
        RAW_COMMENTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return dataframe


def create_summary(
    comments: pd.DataFrame,
    videos: pd.DataFrame,
    log_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Create a collection summary by theme.
    """

    rows = []

    themes = sorted(
        videos[
            "manual_selected_theme"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    for theme in themes:
        theme_videos = videos[
            videos[
                "manual_selected_theme"
            ] == theme
        ]

        if comments.empty:
            theme_comments = comments
        else:
            theme_comments = comments[
                comments["theme"] == theme
            ]

        rows.append(
            {
                "theme": theme,
                "selected_videos": len(
                    theme_videos
                ),
                "videos_with_collected_comments": (
                    theme_comments[
                        "video_id"
                    ].nunique()
                    if not theme_comments.empty
                    else 0
                ),
                "raw_comments_collected": len(
                    theme_comments
                ),
                "unique_comments": (
                    theme_comments[
                        "comment_id"
                    ].nunique()
                    if not theme_comments.empty
                    else 0
                ),
                "total_likes": (
                    int(
                        theme_comments[
                            "like_count"
                        ].sum()
                    )
                    if not theme_comments.empty
                    else 0
                ),
            }
        )

    rows.append(
        {
            "theme": "TOTAL",
            "selected_videos": len(
                videos
            ),
            "videos_with_collected_comments": (
                comments[
                    "video_id"
                ].nunique()
                if not comments.empty
                else 0
            ),
            "raw_comments_collected": len(
                comments
            ),
            "unique_comments": (
                comments[
                    "comment_id"
                ].nunique()
                if not comments.empty
                else 0
            ),
            "total_likes": (
                int(
                    comments[
                        "like_count"
                    ].sum()
                )
                if not comments.empty
                else 0
            ),
        }
    )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return summary


def print_final_summary(
    comments: pd.DataFrame,
    videos: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """
    Print collection results.
    """

    print("\n" + "=" * 78)
    print("YOUTUBE COMMENT COLLECTION COMPLETED")
    print("=" * 78)

    print(
        f"Selected videos             : "
        f"{len(videos):,}"
    )

    print(
        f"Videos contributing comments: "
        f"{comments['video_id'].nunique() if not comments.empty else 0:,}"
    )

    print(
        f"Raw unique comments collected: "
        f"{len(comments):,}"
    )

    print(
        f"Target raw comments         : "
        f"{TARGET_TOTAL_COMMENTS:,}"
    )

    print("\nComments by theme:")

    print(
        summary[
            summary["theme"] != "TOTAL"
        ][
            [
                "theme",
                "selected_videos",
                "videos_with_collected_comments",
                "raw_comments_collected",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nSaved files:")
    print(f"1. {RAW_COMMENTS_FILE}")
    print(f"2. {CHECKPOINT_FILE}")
    print(f"3. {PROGRESS_FILE}")
    print(f"4. {COLLECTION_LOG_FILE}")
    print(f"5. {SUMMARY_FILE}")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> None:
    """
    Run resumable YouTube top-level comment collection.
    """

    try:
        RAW_OUTPUT_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        LOG_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        videos = load_selected_videos()
        youtube = create_youtube_client()

        all_records = load_existing_comments()
        progress = load_progress()

        existing_comment_ids = {
            str(
                record.get(
                    "comment_id",
                    "",
                )
            )
            for record in all_records
            if record.get(
                "comment_id"
            )
        }

        completed_video_ids = set(
            progress.get(
                "completed_video_ids",
                [],
            )
        )

        log_rows: list[dict[str, Any]] = []

        print("=" * 78)
        print("YOUTUBE TOP-LEVEL COMMENT COLLECTION")
        print("=" * 78)

        print(
            f"Selected videos          : "
            f"{len(videos):,}"
        )

        print(
            f"Existing comments loaded : "
            f"{len(existing_comment_ids):,}"
        )

        print(
            f"Global target            : "
            f"{TARGET_TOTAL_COMMENTS:,}"
        )

        print(
            f"Maximum per video        : "
            f"{MAX_COMMENTS_PER_VIDEO:,}"
        )

        print(
            f"Already completed videos : "
            f"{len(completed_video_ids):,}"
        )

        progress_bar = tqdm(
            total=len(videos),
            desc="Videos",
        )

        for _, video in videos.iterrows():
            video_id = str(
                video["video_id"]
            ).strip()

            if (
                len(existing_comment_ids)
                >= TARGET_TOTAL_COMMENTS
            ):
                break

            if video_id in completed_video_ids:
                progress_bar.update(1)
                continue

            started_at = utc_now()

            try:
                collected_count, status = (
                    collect_video_comments(
                        youtube=youtube,
                        video=video,
                        all_records=all_records,
                        existing_comment_ids=(
                            existing_comment_ids
                        ),
                        progress=progress,
                    )
                )

                log_rows.append(
                    {
                        "video_id": video_id,
                        "video_title": str(
                            video.get(
                                "video_title",
                                "",
                            )
                        ),
                        "theme": str(
                            video.get(
                                "manual_selected_theme",
                                "",
                            )
                        ),
                        "reported_comment_count": (
                            safe_integer(
                                video.get(
                                    "comment_count"
                                )
                            )
                        ),
                        "collected_comment_count": (
                            collected_count
                        ),
                        "status": status,
                        "error_message": "",
                        "started_at": started_at,
                        "finished_at": utc_now(),
                    }
                )

            except CommentsDisabledError as error:
                if (
                    video_id
                    not in progress[
                        "completed_video_ids"
                    ]
                ):
                    progress[
                        "completed_video_ids"
                    ].append(
                        video_id
                    )

                save_progress(
                    progress
                )

                log_rows.append(
                    {
                        "video_id": video_id,
                        "video_title": str(
                            video.get(
                                "video_title",
                                "",
                            )
                        ),
                        "theme": str(
                            video.get(
                                "manual_selected_theme",
                                "",
                            )
                        ),
                        "reported_comment_count": (
                            safe_integer(
                                video.get(
                                    "comment_count"
                                )
                            )
                        ),
                        "collected_comment_count": 0,
                        "status": "comments_disabled",
                        "error_message": str(error),
                        "started_at": started_at,
                        "finished_at": utc_now(),
                    }
                )

            except VideoUnavailableError as error:
                if (
                    video_id
                    not in progress[
                        "completed_video_ids"
                    ]
                ):
                    progress[
                        "completed_video_ids"
                    ].append(
                        video_id
                    )

                save_progress(
                    progress
                )

                log_rows.append(
                    {
                        "video_id": video_id,
                        "video_title": str(
                            video.get(
                                "video_title",
                                "",
                            )
                        ),
                        "theme": str(
                            video.get(
                                "manual_selected_theme",
                                "",
                            )
                        ),
                        "reported_comment_count": (
                            safe_integer(
                                video.get(
                                    "comment_count"
                                )
                            )
                        ),
                        "collected_comment_count": 0,
                        "status": "video_unavailable",
                        "error_message": str(error),
                        "started_at": started_at,
                        "finished_at": utc_now(),
                    }
                )

            except YouTubeQuotaError as error:
                save_comment_checkpoint(
                    all_records
                )

                save_progress(
                    progress
                )

                save_collection_log(
                    log_rows
                )

                print(
                    "\nYouTube API quota reached. "
                    "All progress has been saved."
                )

                print(
                    "Run the same command after quota resets "
                    "to continue."
                )

                print(
                    f"Message: {error}"
                )

                break

            progress_bar.update(1)

            save_collection_log(
                log_rows
            )

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

        progress_bar.close()

        comments = save_final_comments(
            all_records
        )

        save_comment_checkpoint(
            all_records
        )

        save_collection_log(
            log_rows
        )

        summary = create_summary(
            comments=comments,
            videos=videos,
            log_rows=log_rows,
        )

        print_final_summary(
            comments=comments,
            videos=videos,
            summary=summary,
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        print(f"\nError:\n{error}")
        sys.exit(1)

    except KeyboardInterrupt:
        print(
            "\nCollection stopped by the user. "
            "The most recent completed page remains saved."
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
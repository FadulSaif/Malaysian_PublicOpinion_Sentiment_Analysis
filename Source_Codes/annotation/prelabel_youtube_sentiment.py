import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "interim"
    / "youtube_comments_cleaned.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "Data"
    / "annotation"
)

OUTPUT_FILE = (
    OUTPUT_DIRECTORY
    / "youtube_sentiment_prelabels.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIRECTORY
    / "youtube_prelabel_summary.csv"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_NAME = (
    "cardiffnlp/"
    "twitter-xlm-roberta-base-sentiment-multilingual"
)

BATCH_SIZE = 16
MAX_TOKEN_LENGTH = 128

VALID_LABELS = {
    "negative",
    "neutral",
    "positive",
}


# ============================================================
# CONFIDENCE SETTINGS
# ============================================================

HIGH_CONFIDENCE_THRESHOLD = 0.80
MEDIUM_CONFIDENCE_THRESHOLD = 0.60

# All neutral predictions are reviewed because neutral is
# usually the most difficult and potentially underrepresented
# class in three-class sentiment analysis.
REVIEW_ALL_NEUTRAL_PREDICTIONS = True


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_float(value: Any) -> float:
    """
    Convert a value safely to float.
    """

    try:
        return float(value)

    except (TypeError, ValueError):
        return 0.0


def normalize_model_label(
    raw_label: Any,
    label_id: int,
) -> str:
    """
    Convert model label names into the project's three labels.

    The model may expose labels such as:
        negative
        neutral
        positive

    or generic labels such as:
        LABEL_0
        LABEL_1
        LABEL_2
    """

    label = str(raw_label).strip().lower()

    direct_mapping = {
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
    }

    if label in direct_mapping:
        return direct_mapping[label]

    fallback_mapping = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }

    if label_id in fallback_mapping:
        return fallback_mapping[label_id]

    raise ValueError(
        f"Unsupported model label: {raw_label}, ID={label_id}"
    )


def get_confidence_category(
    confidence: float,
) -> str:
    """
    Convert numerical confidence into a review category.
    """

    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"

    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"

    return "low"


def determine_review_priority(
    predicted_label: str,
    confidence: float,
    probability_margin: float,
    language_tag: str,
) -> tuple[str, str]:
    """
    Determine whether a prediction should be reviewed urgently.

    Returns:
        review_priority
        review_reason
    """

    reasons = []

    if confidence < MEDIUM_CONFIDENCE_THRESHOLD:
        reasons.append("low_confidence")

    elif confidence < HIGH_CONFIDENCE_THRESHOLD:
        reasons.append("medium_confidence")

    if probability_margin < 0.15:
        reasons.append("small_probability_margin")

    if (
        REVIEW_ALL_NEUTRAL_PREDICTIONS
        and predicted_label == "neutral"
    ):
        reasons.append("neutral_prediction")

    if language_tag in {
        "MIXED",
        "UNKNOWN",
    }:
        reasons.append(
            f"{language_tag.lower()}_language"
        )

    if (
        "low_confidence" in reasons
        or "small_probability_margin" in reasons
    ):
        priority = "high"

    elif (
        "neutral_prediction" in reasons
        or "medium_confidence" in reasons
        or "mixed_language" in reasons
        or "unknown_language" in reasons
    ):
        priority = "medium"

    else:
        priority = "low"

    return (
        priority,
        " | ".join(reasons),
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def load_cleaned_comments() -> pd.DataFrame:
    """
    Load and validate the cleaned YouTube comments.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned YouTube file was not found:\n"
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
        "theme",
        "raw_text",
        "clean_text",
        "language_tag",
        "cleaning_status",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The cleaned file is missing required columns: "
            + ", ".join(missing_columns)
        )

    dataframe = dataframe[
        dataframe["cleaning_status"] == "accepted"
    ].copy()

    dataframe["clean_text"] = (
        dataframe["clean_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe = dataframe[
        dataframe["clean_text"].ne("")
    ].copy()

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


# ============================================================
# MODEL LOADING
# ============================================================

def select_device() -> torch.device:
    """
    Select GPU when CUDA is available; otherwise use CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model_and_tokenizer(
    device: torch.device,
):
    """
    Download and load the multilingual sentiment model.
    """

    print(f"\nLoading model: {MODEL_NAME}")
    print(f"Device       : {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    return tokenizer, model


# ============================================================
# MODEL INFERENCE
# ============================================================

def create_batches(
    texts: list[str],
    batch_size: int,
) -> list[list[str]]:
    """
    Split texts into batches.
    """

    return [
        texts[index:index + batch_size]
        for index in range(
            0,
            len(texts),
            batch_size,
        )
    ]


def predict_sentiments(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> list[dict[str, Any]]:
    """
    Predict sentiment probabilities for all cleaned comments.
    """

    predictions: list[dict[str, Any]] = []

    batches = create_batches(
        texts=texts,
        batch_size=BATCH_SIZE,
    )

    id2label = model.config.id2label

    with torch.no_grad():
        for batch in tqdm(
            batches,
            desc="Sentiment batches",
        ):
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            outputs = model(**encoded)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

            probabilities = (
                probabilities
                .detach()
                .cpu()
                .numpy()
            )

            for probability_vector in probabilities:
                class_probabilities = {
                    "negative": 0.0,
                    "neutral": 0.0,
                    "positive": 0.0,
                }

                for label_id, probability in enumerate(
                    probability_vector
                ):
                    raw_label = id2label.get(
                        label_id,
                        f"LABEL_{label_id}",
                    )

                    normalized_label = normalize_model_label(
                        raw_label=raw_label,
                        label_id=label_id,
                    )

                    class_probabilities[
                        normalized_label
                    ] = float(probability)

                sorted_probabilities = sorted(
                    class_probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )

                predicted_label = (
                    sorted_probabilities[0][0]
                )

                confidence = float(
                    sorted_probabilities[0][1]
                )

                second_probability = float(
                    sorted_probabilities[1][1]
                )

                probability_margin = (
                    confidence - second_probability
                )

                predictions.append(
                    {
                        "predicted_label": (
                            predicted_label
                        ),
                        "negative_probability": (
                            class_probabilities[
                                "negative"
                            ]
                        ),
                        "neutral_probability": (
                            class_probabilities[
                                "neutral"
                            ]
                        ),
                        "positive_probability": (
                            class_probabilities[
                                "positive"
                            ]
                        ),
                        "prediction_confidence": (
                            confidence
                        ),
                        "probability_margin": (
                            probability_margin
                        ),
                        "confidence_category": (
                            get_confidence_category(
                                confidence
                            )
                        ),
                    }
                )

    return predictions


# ============================================================
# OUTPUT PREPARATION
# ============================================================

def add_prediction_columns(
    dataframe: pd.DataFrame,
    predictions: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Add automatic sentiment predictions and review fields.
    """

    if len(dataframe) != len(predictions):
        raise ValueError(
            "Prediction count does not match comment count."
        )

    prediction_dataframe = pd.DataFrame(
        predictions
    )

    output = pd.concat(
        [
            dataframe.reset_index(drop=True),
            prediction_dataframe.reset_index(drop=True),
        ],
        axis=1,
    )

    review_results = output.apply(
        lambda row: determine_review_priority(
            predicted_label=str(
                row["predicted_label"]
            ),
            confidence=safe_float(
                row["prediction_confidence"]
            ),
            probability_margin=safe_float(
                row["probability_margin"]
            ),
            language_tag=str(
                row.get(
                    "language_tag",
                    "UNKNOWN",
                )
            ),
        ),
        axis=1,
    )

    output["review_priority"] = (
        review_results.apply(
            lambda result: result[0]
        )
    )

    output["review_reason"] = (
        review_results.apply(
            lambda result: result[1]
        )
    )

    # These fields will be completed during manual verification.
    output["manual_label"] = ""
    output["manual_review_status"] = "pending"
    output["manual_review_notes"] = ""

    # The final label remains empty until manual verification.
    output["final_label"] = ""

    output["model_name"] = MODEL_NAME
    output["annotation_stage"] = "automatic_prelabel"

    return output


def create_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a summary of predicted labels, confidence and language.
    """

    rows = []

    for label, count in (
        dataframe["predicted_label"]
        .value_counts()
        .items()
    ):
        rows.append(
            {
                "summary_type": "predicted_label",
                "category": label,
                "count": int(count),
            }
        )

    for category, count in (
        dataframe["confidence_category"]
        .value_counts()
        .items()
    ):
        rows.append(
            {
                "summary_type": "confidence_category",
                "category": category,
                "count": int(count),
            }
        )

    for priority, count in (
        dataframe["review_priority"]
        .value_counts()
        .items()
    ):
        rows.append(
            {
                "summary_type": "review_priority",
                "category": priority,
                "count": int(count),
            }
        )

    for language, count in (
        dataframe["language_tag"]
        .value_counts()
        .items()
    ):
        rows.append(
            {
                "summary_type": "language_tag",
                "category": language,
                "count": int(count),
            }
        )

    for theme, count in (
        dataframe["theme"]
        .value_counts()
        .items()
    ):
        rows.append(
            {
                "summary_type": "theme",
                "category": theme,
                "count": int(count),
            }
        )

    rows.append(
        {
            "summary_type": "overall",
            "category": "total_comments",
            "count": len(dataframe),
        }
    )

    rows.append(
        {
            "summary_type": "overall",
            "category": "mean_confidence",
            "count": round(
                float(
                    dataframe[
                        "prediction_confidence"
                    ].mean()
                ),
                4,
            ),
        }
    )

    return pd.DataFrame(rows)


def save_outputs(
    dataframe: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """
    Save the pre-labelled dataset and summary.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def print_summary(
    dataframe: pd.DataFrame,
) -> None:
    """
    Print prediction results.
    """

    print("\n" + "=" * 78)
    print("AUTOMATIC SENTIMENT PRE-LABELLING COMPLETED")
    print("=" * 78)

    print(
        f"Comments classified       : "
        f"{len(dataframe):,}"
    )

    print("\nPredicted sentiment distribution:")

    print(
        dataframe["predicted_label"]
        .value_counts()
        .reindex(
            [
                "positive",
                "neutral",
                "negative",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print("\nConfidence categories:")

    print(
        dataframe["confidence_category"]
        .value_counts()
        .reindex(
            [
                "high",
                "medium",
                "low",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print("\nManual-review priorities:")

    print(
        dataframe["review_priority"]
        .value_counts()
        .reindex(
            [
                "high",
                "medium",
                "low",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print(
        "\nAverage confidence: "
        f"{dataframe['prediction_confidence'].mean():.4f}"
    )

    print("\nSaved files:")
    print(f"1. {OUTPUT_FILE}")
    print(f"2. {SUMMARY_FILE}")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> None:
    """
    Run automatic sentiment pre-classification.
    """

    try:
        print("=" * 78)
        print("AUTOMATIC YOUTUBE SENTIMENT PRE-LABELLING")
        print("=" * 78)

        print(f"Input file: {INPUT_FILE}")

        dataframe = load_cleaned_comments()

        device = select_device()

        tokenizer, model = load_model_and_tokenizer(
            device=device
        )

        texts = (
            dataframe["clean_text"]
            .astype(str)
            .tolist()
        )

        predictions = predict_sentiments(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        output = add_prediction_columns(
            dataframe=dataframe,
            predictions=predictions,
        )

        # Put the highest-priority review records first.
        priority_order = {
            "high": 0,
            "medium": 1,
            "low": 2,
        }

        output["_priority_order"] = (
            output["review_priority"]
            .map(priority_order)
            .fillna(3)
        )

        output.sort_values(
            by=[
                "_priority_order",
                "prediction_confidence",
            ],
            ascending=[
                True,
                True,
            ],
            inplace=True,
        )

        output.drop(
            columns=["_priority_order"],
            inplace=True,
        )

        output.reset_index(
            drop=True,
            inplace=True,
        )

        summary = create_summary(
            output
        )

        save_outputs(
            dataframe=output,
            summary=summary,
        )

        print_summary(
            output
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        print(f"\nError:\n{error}")
        sys.exit(1)

    except KeyboardInterrupt:
        print(
            "\nPre-labelling stopped by the user."
        )
        sys.exit(1)

    except RuntimeError as error:
        print(
            f"\nModel runtime error:\n{error}"
        )

        if "out of memory" in str(error).lower():
            print(
                "\nReduce BATCH_SIZE from 16 to 8 or 4."
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
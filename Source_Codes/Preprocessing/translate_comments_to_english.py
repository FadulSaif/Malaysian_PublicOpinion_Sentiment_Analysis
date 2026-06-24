import time
from pathlib import Path

import pandas as pd
from deep_translator import GoogleTranslator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "Data" / "processed" / "final_youtube_sentiment_825_balanced.csv"
OUTPUT_FILE = PROJECT_ROOT / "Data" / "processed" / "final_youtube_sentiment_825_translated_en.csv"
CACHE_FILE = PROJECT_ROOT / "Data" / "processed" / "translation_cache.csv"

TEXT_COLUMN = "clean_text"
LABEL_COLUMN = "final_label"

TRANSLATOR = GoogleTranslator(source="auto", target="en")


def is_empty_text(value):
    return pd.isna(value) or str(value).strip() == ""


def should_translate(row):
    """
    Returns True when the comment is not clearly English.
    English comments are kept unchanged.
    Malay, mixed, unknown, or unclear comments are translated.
    """
    language_tag = str(row.get("language_tag", "")).strip().lower()
    detected_language = str(row.get("detected_language", "")).strip().lower()

    english_values = {"en", "eng", "english"}

    if language_tag in english_values or detected_language in english_values:
        return False

    return True


def load_translation_cache():
    if CACHE_FILE.exists():
        cache_df = pd.read_csv(CACHE_FILE)
        if {"original_text", "translated_text_en"}.issubset(cache_df.columns):
            return dict(zip(cache_df["original_text"], cache_df["translated_text_en"]))
    return {}


def save_translation_cache(cache):
    cache_df = pd.DataFrame(
        [{"original_text": key, "translated_text_en": value} for key, value in cache.items()]
    )
    cache_df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")


def translate_text(text, cache):
    text = str(text).strip()

    if text in cache:
        return cache[text], "cached"

    try:
        translated = TRANSLATOR.translate(text)

        if translated is None or str(translated).strip() == "":
            translated = text
            status = "failed_empty_translation"
        else:
            translated = str(translated).strip()
            status = "translated"

        cache[text] = translated
        return translated, status

    except Exception as error:
        print(f"[Translation error] {error}")
        cache[text] = text
        return text, "failed_kept_original"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Loaded dataset: {INPUT_FILE}")
    print(f"Rows: {len(df)}")

    cache = load_translation_cache()
    translated_texts = []
    translation_statuses = []
    translation_methods = []

    for index, row in df.iterrows():
        original_text = row[TEXT_COLUMN]

        if is_empty_text(original_text):
            translated_texts.append("")
            translation_statuses.append("empty_text")
            translation_methods.append("none")
            continue

        original_text = str(original_text).strip()

        if should_translate(row):
            translated_text, status = translate_text(original_text, cache)
            method = "google_translate_auto_to_en"
        else:
            translated_text = original_text
            status = "kept_original_english"
            method = "none"

        translated_texts.append(translated_text)
        translation_statuses.append(status)
        translation_methods.append(method)

        if (index + 1) % 25 == 0:
            print(f"Processed {index + 1}/{len(df)} rows...")
            save_translation_cache(cache)
            time.sleep(1)

    df["original_text"] = df[TEXT_COLUMN].astype(str)
    df["translated_text_en"] = translated_texts
    df["translation_status"] = translation_statuses
    df["translation_method"] = translation_methods

    # Keep a modelling-ready version of the label.
    df["sentiment"] = df[LABEL_COLUMN].astype(str).str.strip().str.lower()

    # Basic validation.
    print("\nFinal sentiment distribution:")
    print(df["sentiment"].value_counts())

    print("\nTranslation status distribution:")
    print(df["translation_status"].value_counts())

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    save_translation_cache(cache)

    print(f"\nSaved translated dataset to: {OUTPUT_FILE}")
    print(f"Saved translation cache to: {CACHE_FILE}")


if __name__ == "__main__":
    main()
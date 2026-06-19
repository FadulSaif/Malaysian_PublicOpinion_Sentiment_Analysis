from pathlib import Path
from datetime import datetime
import os

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_LOG_FILE = (
    PROJECT_ROOT
    / "results"
    / "streamlit_prediction_log.csv"
)

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Malaysia Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 26px;
        border-radius: 18px;
        background-color: #f7f7f9;
        border: 1px solid #e6e6e6;
        margin-top: 18px;
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_xlmr_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_xlmr_model()
id2label = model.config.id2label


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_label(raw_label, label_id):
    label = str(raw_label).strip().lower()

    mapping = {
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
    }

    if label in mapping:
        return mapping[label]

    fallback = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }

    return fallback.get(label_id, "neutral")


def predict_single_text(text):
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**encoded)
        probabilities_tensor = torch.softmax(outputs.logits, dim=-1)[0]

    probabilities_list = probabilities_tensor.detach().cpu().tolist()

    probabilities = {}

    for label_id, probability in enumerate(probabilities_list):
        raw_label = id2label.get(label_id, f"LABEL_{label_id}")
        label = normalize_label(raw_label, label_id)
        probabilities[label] = float(probability)

    prediction = max(
        probabilities,
        key=probabilities.get,
    )

    confidence = probabilities[prediction]

    return prediction, probabilities, confidence


def predict_batch(texts):
    predictions = []
    negative_probs = []
    neutral_probs = []
    positive_probs = []
    confidence_scores = []

    texts = list(texts)

    batch_size = 16

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            outputs = model(**encoded)
            probabilities_tensor = torch.softmax(outputs.logits, dim=-1)

        probabilities_array = probabilities_tensor.detach().cpu().tolist()

        for probability_vector in probabilities_array:
            probabilities = {}

            for label_id, probability in enumerate(probability_vector):
                raw_label = id2label.get(label_id, f"LABEL_{label_id}")
                label = normalize_label(raw_label, label_id)
                probabilities[label] = float(probability)

            prediction = max(
                probabilities,
                key=probabilities.get,
            )

            predictions.append(prediction)
            negative_probs.append(probabilities.get("negative", 0.0))
            neutral_probs.append(probabilities.get("neutral", 0.0))
            positive_probs.append(probabilities.get("positive", 0.0))
            confidence_scores.append(probabilities[prediction])

    return (
        predictions,
        negative_probs,
        neutral_probs,
        positive_probs,
        confidence_scores,
    )


def get_confidence_level(confidence):
    if confidence >= 0.70:
        return "High"

    if confidence >= 0.50:
        return "Medium"

    return "Low"


def get_sentiment_description(label):
    label = str(label).lower().strip()

    if label == "negative":
        return (
            "The comment appears to express criticism, dissatisfaction, complaint, "
            "sarcasm, anger, or a negative opinion."
        )

    if label == "neutral":
        return (
            "The comment appears factual, descriptive, questioning, mixed, "
            "or without strong positive or negative sentiment."
        )

    if label == "positive":
        return (
            "The comment appears to express support, praise, agreement, "
            "appreciation, or a positive opinion."
        )

    return "The sentiment could not be interpreted clearly."


def make_probability_chart(probabilities):
    probability_df = pd.DataFrame(
        [
            {
                "Sentiment": label.capitalize(),
                "Probability": probability * 100,
            }
            for label, probability in probabilities.items()
        ]
    )

    fig = px.bar(
        probability_df,
        x="Sentiment",
        y="Probability",
        text=probability_df["Probability"].round(2),
        title="Sentiment Probability",
    )

    fig.update_traces(
        texttemplate="%{text}%",
        textposition="outside",
    )

    fig.update_layout(
        yaxis_range=[0, 100],
        yaxis_title="Probability (%)",
        xaxis_title="Sentiment",
        height=420,
        title_x=0.02,
        showlegend=False,
    )

    return fig


def make_sentiment_distribution_chart(df):
    counts = (
        df["predicted_sentiment"]
        .value_counts()
        .reset_index()
    )

    counts.columns = ["Sentiment", "Count"]
    counts["Sentiment"] = counts["Sentiment"].str.capitalize()

    fig = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        title="Overall Sentiment Distribution",
        hole=0.45,
    )

    fig.update_layout(
        height=430,
        title_x=0.02,
    )

    return fig


def make_sentiment_bar_chart(df):
    counts = (
        df["predicted_sentiment"]
        .value_counts()
        .reset_index()
    )

    counts.columns = ["Sentiment", "Count"]
    counts["Sentiment"] = counts["Sentiment"].str.capitalize()

    fig = px.bar(
        counts,
        x="Sentiment",
        y="Count",
        text="Count",
        title="Number of Comments by Sentiment",
    )

    fig.update_traces(
        textposition="outside",
    )

    fig.update_layout(
        height=420,
        title_x=0.02,
        xaxis_title="Sentiment",
        yaxis_title="Number of Comments",
    )

    return fig


def save_prediction_log(text, prediction, probabilities, confidence):
    PREDICTION_LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_text": text,
        "predicted_sentiment": prediction,
        "negative_probability": probabilities.get("negative"),
        "neutral_probability": probabilities.get("neutral"),
        "positive_probability": probabilities.get("positive"),
        "confidence": confidence,
    }

    new_log = pd.DataFrame([row])

    if PREDICTION_LOG_FILE.exists():
        old_log = pd.read_csv(
            PREDICTION_LOG_FILE,
            encoding="utf-8-sig",
        )

        new_log = pd.concat(
            [old_log, new_log],
            ignore_index=True,
        )

    new_log.to_csv(
        PREDICTION_LOG_FILE,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Malaysia Sentiment Analysis Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Analyze Malaysia-related comments as positive, neutral, or negative.</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("Dashboard")

    page = st.radio(
        "Choose analysis type",
        [
            "Single Comment Analysis",
            "Batch Comment Analysis",
        ],
    )

    st.divider()

    st.write("**Sentiment classes**")
    st.write("Negative")
    st.write("Neutral")
    st.write("Positive")

    st.divider()

    st.caption(
        "This dashboard predicts sentiment from comment text and visualizes the results."
    )


# ============================================================
# SINGLE COMMENT PAGE
# ============================================================

if page == "Single Comment Analysis":
    st.header("Single Comment Analysis")

    st.write(
        "Enter one comment below to identify whether it expresses negative, neutral, or positive sentiment."
    )

    if "single_input" not in st.session_state:
        st.session_state["single_input"] = ""

    def set_example_text(text):
        st.session_state["single_input"] = text

    example_col1, example_col2, example_col3 = st.columns(3)

    with example_col1:
        st.button(
            "Use negative example",
            use_container_width=True,
            on_click=set_example_text,
            args=("Harga barang makin naik, rakyat makin susah.",),
        )

    with example_col2:
        st.button(
            "Use neutral example",
            use_container_width=True,
            on_click=set_example_text,
            args=("Bilakah program bantuan ini bermula?",),
        )

    with example_col3:
        st.button(
            "Use positive example",
            use_container_width=True,
            on_click=set_example_text,
            args=("Bagus usaha ini kerana dapat membantu rakyat.",),
        )

    user_text = st.text_area(
        "Comment text",
        key="single_input",
        height=170,
        placeholder="Type or paste a comment here...",
    )

    if st.button("Analyze Sentiment", use_container_width=True):
        cleaned_text = user_text.strip()

        if not cleaned_text:
            st.warning("Please enter a comment first.")

        else:
            with st.spinner("Analyzing sentiment..."):
                prediction, probabilities, confidence = predict_single_text(
                    cleaned_text
                )

            save_prediction_log(
                text=cleaned_text,
                prediction=prediction,
                probabilities=probabilities,
                confidence=confidence,
            )

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Predicted sentiment",
                    str(prediction).capitalize(),
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%",
                )

            with col3:
                st.metric(
                    "Confidence level",
                    get_confidence_level(confidence),
                )

            st.write(
                get_sentiment_description(prediction)
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

            st.plotly_chart(
                make_probability_chart(probabilities),
                use_container_width=True,
            )


# ============================================================
# BATCH COMMENT PAGE
# ============================================================

elif page == "Batch Comment Analysis":
    st.header("Batch Comment Analysis")

    st.write(
        "Upload a CSV file containing comments. The dashboard will classify each comment "
        "and generate sentiment visualizations."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(
            uploaded_file,
            encoding="utf-8-sig",
        )

        st.subheader("Uploaded Data Preview")

        st.dataframe(
            batch_df.head(10),
            use_container_width=True,
        )

        text_columns = (
            batch_df
            .select_dtypes(include=["object"])
            .columns
            .tolist()
        )

        if not text_columns:
            st.error(
                "No text column was found in the uploaded CSV file."
            )

        else:
            selected_column = st.selectbox(
                "Select the comment/text column",
                text_columns,
            )

            if st.button("Run Sentiment Analysis", use_container_width=True):
                output_df = batch_df.copy()

                output_df[selected_column] = (
                    output_df[selected_column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

                valid_mask = output_df[selected_column].ne("")

                valid_texts = output_df.loc[
                    valid_mask,
                    selected_column,
                ].tolist()

                if len(valid_texts) == 0:
                    st.warning(
                        "No valid text was found in the selected column."
                    )

                else:
                    with st.spinner(
                        "Running sentiment analysis. This may take some time for large files..."
                    ):
                        (
                            predictions,
                            negative_probabilities,
                            neutral_probabilities,
                            positive_probabilities,
                            confidence_scores,
                        ) = predict_batch(valid_texts)

                    output_df["predicted_sentiment"] = ""
                    output_df["negative_probability"] = None
                    output_df["neutral_probability"] = None
                    output_df["positive_probability"] = None
                    output_df["confidence"] = None

                    output_df.loc[
                        valid_mask,
                        "predicted_sentiment",
                    ] = predictions

                    output_df.loc[
                        valid_mask,
                        "negative_probability",
                    ] = negative_probabilities

                    output_df.loc[
                        valid_mask,
                        "neutral_probability",
                    ] = neutral_probabilities

                    output_df.loc[
                        valid_mask,
                        "positive_probability",
                    ] = positive_probabilities

                    output_df.loc[
                        valid_mask,
                        "confidence",
                    ] = confidence_scores

                    analyzed_df = output_df[valid_mask].copy()

                    st.success(
                        "Sentiment analysis completed."
                    )

                    total_comments = len(analyzed_df)

                    negative_count = int(
                        (
                            analyzed_df["predicted_sentiment"]
                            == "negative"
                        ).sum()
                    )

                    neutral_count = int(
                        (
                            analyzed_df["predicted_sentiment"]
                            == "neutral"
                        ).sum()
                    )

                    positive_count = int(
                        (
                            analyzed_df["predicted_sentiment"]
                            == "positive"
                        ).sum()
                    )

                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                    with metric_col1:
                        st.metric(
                            "Analyzed comments",
                            total_comments,
                        )

                    with metric_col2:
                        st.metric(
                            "Negative",
                            negative_count,
                        )

                    with metric_col3:
                        st.metric(
                            "Neutral",
                            neutral_count,
                        )

                    with metric_col4:
                        st.metric(
                            "Positive",
                            positive_count,
                        )

                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:
                        st.plotly_chart(
                            make_sentiment_distribution_chart(
                                analyzed_df
                            ),
                            use_container_width=True,
                        )

                    with chart_col2:
                        st.plotly_chart(
                            make_sentiment_bar_chart(
                                analyzed_df
                            ),
                            use_container_width=True,
                        )

                    confidence_df = analyzed_df.copy()

                    confidence_df["confidence_percentage"] = (
                        confidence_df["confidence"] * 100
                    )

                    fig = px.box(
                        confidence_df,
                        x="predicted_sentiment",
                        y="confidence_percentage",
                        title="Confidence Distribution by Sentiment",
                        labels={
                            "predicted_sentiment": "Predicted sentiment",
                            "confidence_percentage": "Confidence (%)",
                        },
                    )

                    fig.update_layout(
                        height=420,
                        title_x=0.02,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                    st.subheader(
                        "Prediction Results"
                    )

                    st.dataframe(
                        output_df,
                        use_container_width=True,
                    )

                    csv_output = output_df.to_csv(
                        index=False,
                        encoding="utf-8-sig",
                    )

                    st.download_button(
                        label="Download Sentiment Results CSV",
                        data=csv_output,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
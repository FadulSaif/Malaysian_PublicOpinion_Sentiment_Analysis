from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "best_translated_english_sentiment_model.joblib"
)

PREDICTION_LOG_FILE = (
    PROJECT_ROOT
    / "results"
    / "streamlit_prediction_log.csv"
)


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
        background-color: transparent;
        border: 1px solid #2f3542;
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
def load_sentiment_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)
    return model


model = load_sentiment_model()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def softmax(values):
    values = np.array(values, dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def get_model_scores(model, texts):
    labels = list(model.classes_)

    if hasattr(model, "predict_proba"):
        score_array = model.predict_proba(texts)

    elif hasattr(model, "decision_function"):
        decision_scores = model.decision_function(texts)

        if len(decision_scores.shape) == 1:
            decision_scores = np.vstack(
                [-decision_scores, decision_scores]
            ).T

        score_array = np.array([
            softmax(row)
            for row in decision_scores
        ])

    else:
        predictions = model.predict(texts)
        score_array = []

        for prediction in predictions:
            row = [
                1.0 if label == prediction else 0.0
                for label in labels
            ]
            score_array.append(row)

        score_array = np.array(score_array)

    return labels, score_array


def get_reliability_label(confidence):
    if confidence >= 0.70:
        return "High confidence"

    if confidence >= 0.50:
        return "Medium confidence"

    return "Low confidence - needs review"


def predict_single_text(text):
    model_input_text = str(text).strip()

    labels, score_array = get_model_scores(
        model,
        [model_input_text]
    )

    scores = {
        label: float(score_array[0][index])
        for index, label in enumerate(labels)
    }

    prediction = max(
        scores,
        key=scores.get
    )

    confidence = scores[prediction]

    return model_input_text, prediction, scores, confidence


def predict_batch(texts):
    input_texts = [
        str(text).strip()
        for text in texts
    ]

    labels, score_array = get_model_scores(
        model,
        input_texts
    )

    predictions = []
    negative_scores = []
    neutral_scores = []
    positive_scores = []
    confidence_scores = []
    reliability_labels = []

    for row in score_array:
        scores = {
            label: float(row[index])
            for index, label in enumerate(labels)
        }

        prediction = max(
            scores,
            key=scores.get
        )

        confidence = scores[prediction]

        predictions.append(prediction)
        negative_scores.append(scores.get("negative", 0.0))
        neutral_scores.append(scores.get("neutral", 0.0))
        positive_scores.append(scores.get("positive", 0.0))
        confidence_scores.append(confidence)
        reliability_labels.append(get_reliability_label(confidence))

    return (
        input_texts,
        predictions,
        negative_scores,
        neutral_scores,
        positive_scores,
        confidence_scores,
        reliability_labels,
    )


def make_score_chart(scores):
    score_df = pd.DataFrame(
        [
            {
                "Sentiment": label.capitalize(),
                "Score": score * 100,
            }
            for label, score in scores.items()
        ]
    )

    fig = px.bar(
        score_df,
        x="Sentiment",
        y="Score",
        text=score_df["Score"].round(2),
        title="Sentiment Score"
    )

    fig.update_traces(
        texttemplate="%{text}%",
        textposition="outside",
    )

    fig.update_layout(
        yaxis_range=[0, 100],
        yaxis_title="Score (%)",
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


def save_prediction_log(
    original_text,
    model_input_text,
    prediction,
    scores,
    confidence,
    reliability
):
    PREDICTION_LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_text": original_text,
        "model_input_text": model_input_text,
        "predicted_sentiment": prediction,
        "negative_score": scores.get("negative"),
        "neutral_score": scores.get("neutral"),
        "positive_score": scores.get("positive"),
        "confidence": confidence,
        "prediction_reliability": reliability,
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
    '<div class="subtitle">Analyze English Malaysia-related comments using the final sentiment analysis model.</div>',
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
        "This system expects English text input. Uncertain predictions are flagged for review."
    )


# ============================================================
# SINGLE COMMENT PAGE
# ============================================================

if page == "Single Comment Analysis":
    st.header("Single Comment Analysis")

    st.write(
        "Enter one English comment below. The model will classify it as negative, neutral, or positive."
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
            args=("Many people are struggling because prices keep increasing.",),
        )

    with example_col2:
        st.button(
            "Use neutral example",
            use_container_width=True,
            on_click=set_example_text,
            args=("When will the government announce the new policy?",),
        )

    with example_col3:
        st.button(
            "Use positive example",
            use_container_width=True,
            on_click=set_example_text,
            args=("This programme is useful and helps many families.",),
        )

    user_text = st.text_area(
        "Comment text",
        key="single_input",
        height=170,
        placeholder="Type or paste an English comment here...",
    )

    if st.button("Analyze Sentiment", use_container_width=True):
        cleaned_text = user_text.strip()

        if not cleaned_text:
            st.warning("Please enter a comment first.")

        else:
            with st.spinner("Analyzing sentiment..."):
                model_input_text, prediction, scores, confidence = predict_single_text(
                    cleaned_text
                )

            reliability = get_reliability_label(confidence)

            save_prediction_log(
                original_text=cleaned_text,
                model_input_text=model_input_text,
                prediction=prediction,
                scores=scores,
                confidence=confidence,
                reliability=reliability,
            )

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True,
            )

            st.metric(
                "Predicted sentiment",
                str(prediction).capitalize(),
            )

            if confidence < 0.50:
                st.warning(
                    "This prediction may be uncertain and should be reviewed carefully."
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

            st.subheader("Text Used by the Model")
            st.write(model_input_text)

            st.plotly_chart(
                make_score_chart(scores),
                use_container_width=True,
            )


# ============================================================
# BATCH COMMENT PAGE
# ============================================================

elif page == "Batch Comment Analysis":
    st.header("Batch Comment Analysis")

    st.write(
        "Upload a CSV file containing English comments. The dashboard will classify each comment and generate visualisations."
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
                            input_texts,
                            predictions,
                            negative_scores,
                            neutral_scores,
                            positive_scores,
                            confidence_scores,
                            reliability_labels,
                        ) = predict_batch(valid_texts)

                    output_df["model_input_text"] = ""
                    output_df["predicted_sentiment"] = ""

                    output_df.loc[
                        valid_mask,
                        "model_input_text",
                    ] = input_texts

                    output_df.loc[
                        valid_mask,
                        "predicted_sentiment",
                    ] = predictions

                    analyzed_df = output_df[valid_mask].copy()

                    low_confidence_count = sum(
                        confidence < 0.50
                        for confidence in confidence_scores
                    )

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

                    if low_confidence_count > 0:
                        st.warning(
                            f"{low_confidence_count} prediction(s) may be uncertain and should be reviewed carefully."
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
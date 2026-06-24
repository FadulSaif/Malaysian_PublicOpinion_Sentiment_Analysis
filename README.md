# Malaysia YouTube Sentiment Analysis System

## Project Overview

This project is a sentiment analysis system for Malaysia-related YouTube comments. The system classifies public comments into three sentiment categories:

* Negative
* Neutral
* Positive

The project focuses on the topic **Social Media & Public Opinion (Malaysia Trends)**. The original comments were collected from YouTube, cleaned, translated/normalised into English, manually reviewed, relabelled where necessary, balanced, and used to evaluate multiple sentiment classification models.

The final deployed system is an interactive Streamlit dashboard that allows users to analyse individual English comments or upload a CSV file for batch sentiment analysis.

---

## Main Features

The system includes:

* YouTube comment data collection
* Text preprocessing and cleaning
* English-normalised dataset preparation
* Manual sentiment review and relabelling
* Final balanced dataset preparation
* Traditional machine-learning and transformer model evaluation
* Single-comment sentiment prediction
* Batch CSV sentiment analysis
* Sentiment distribution visualisations
* Downloadable prediction results

---

## Final Dataset Summary

The final dataset used for model training and evaluation is the cleaned, reviewed, and balanced translated-English dataset.

| Sentiment | Number of Comments |
| --------- | -----------------: |
| Negative  |                150 |
| Neutral   |                150 |
| Positive  |                150 |
| **Total** |            **450** |

The final dataset is stored at:

```text
Data/processed/final_youtube_sentiment_translated_en_cleaned_v2_balanced.csv
```

The final model uses the `translated_text_en` column as input and the `sentiment` column as the target label.

---

## Data Source

The original comments were collected from selected Malaysia-related YouTube videos using the official YouTube Data API v3.

The selected videos cover several Malaysia-related public-opinion themes, including:

* Cost of living and public policy
* Transport and public services
* Festivals and national events
* Tourism and lifestyle
* Technology and digital services

After collection, the comments were cleaned, translated/normalised into English, manually reviewed, relabelled where needed, and balanced before model training.

---

## Models Evaluated

Several models were evaluated during the project:

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. Word + Character TF-IDF + Logistic Regression
4. Word + Character TF-IDF + Linear SVM
5. Tuned English TF-IDF + Logistic Regression
6. Tuned English TF-IDF + Linear SVM
7. Tuned English Word + Character TF-IDF + Linear SVM
8. Fine-Tuned DistilBERT English

The best-performing model was:

```text
Tuned English Word + Character TF-IDF + Linear SVM
```

Final performance:

| Metric         |  Score |
| -------------- | -----: |
| Accuracy       | 74.44% |
| Macro F1-score | 74.07% |

Although Fine-Tuned DistilBERT was also evaluated, it performed lower on the final balanced dataset. The tuned SVM model was selected because it performed better on the small, cleaned, balanced, translated-English social media dataset.

Saved final model:

```text
models/best_translated_english_sentiment_model.joblib
```

---

## Final Dashboard

The final Streamlit dashboard provides two main functions.

### 1. Single Comment Analysis

The user enters one English comment, and the system predicts whether the comment is negative, neutral, or positive. The dashboard also displays the confidence score and sentiment score chart.

### 2. Batch Comment Analysis

The user uploads a CSV file containing English comments. The system predicts the sentiment for each comment, displays visual summaries, and allows the user to download the prediction results.

---

## Project Folder Structure

```text
malaysia_sentiment_project_2/
│
├── README.md
├── requirements.txt
├── .gitignore
├── model_training_and_evaluation_final.ipynb
│
├── Data/
│   └── processed/
│       └── final_youtube_sentiment_translated_en_cleaned_v2_balanced.csv
│
├── Source_Codes/
│   ├── collection/
│   │   ├── search_youtube_videos.py
│   │   ├── collect_youtube_comments.py
│   │   └── create_final_youtube_video_list.py
│   │
│   ├── Preprocessing/
│   │   └── clean_youtube_comments.py
│   │
│   ├── annotation/
│   │   ├── prelabel_youtube_sentiment.py
│   │   └── manual_review_app.py
│   │
│   └── deployment/
│       └── app.py
│
├── models/
│   └── best_translated_english_sentiment_model.joblib
│
└── results/
    ├── model_comparison_results_translated_en.csv
    └── translated_en_experiment_metadata.json
```

Some folders may contain fewer files depending on the cleaned submission version.

---

## Installation

Open the terminal inside the project folder:

```cmd
cd C:\Users\FadSaif\Documents\malaysia_sentiment_project_2
```

Install the required libraries:

```cmd
python -m pip install -r requirements.txt
```

---

## Running the Dashboard

Run the Streamlit dashboard with:

```cmd
python -m streamlit run Source_Codes\deployment\app.py
```

The dashboard will open in the browser. If it does not open automatically, copy the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

The final dashboard expects English text input because the final model was trained on the translated-English dataset.

---

## Running the Notebook

The final model training and evaluation notebook is:

```text
model_training_and_evaluation_final.ipynb
```

Open it using Jupyter Notebook:

```cmd
jupyter notebook
```

Then open:

```text
model_training_and_evaluation_final.ipynb
```

The notebook contains the final balanced dataset loading, model training, model evaluation, comparison results, and best-model saving steps.

---

## Important Files

| File                                                                           | Purpose                                               |
| ------------------------------------------------------------------------------ | ----------------------------------------------------- |
| `Source_Codes/deployment/app.py`                                               | Final Streamlit dashboard                             |
| `Data/processed/final_youtube_sentiment_translated_en_cleaned_v2_balanced.csv` | Final cleaned and balanced translated-English dataset |
| `model_training_and_evaluation_final.ipynb`                                    | Final model training and evaluation notebook          |
| `models/best_translated_english_sentiment_model.joblib`                        | Final saved SVM sentiment model                       |
| `results/model_comparison_results_translated_en.csv`                           | Final model comparison results                        |
| `results/translated_en_experiment_metadata.json`                               | Final experiment metadata                             |
| `requirements.txt`                                                             | Python package requirements                           |

---

## Environment Variables

The original YouTube data collection required a YouTube API key. For safety, the real `.env` file should not be submitted.

Use `.env.example` as a template if data collection needs to be repeated:

```env
YOUTUBE_API_KEY=your_youtube_api_key_here
```

The final dashboard does not require the YouTube API key because it uses the already processed dataset and saved model.

---

## Limitations

This system is based on selected Malaysia-related YouTube comments. Therefore, the results should not be interpreted as representing the entire Malaysian population or all social media platforms.

The final deployed model expects English text input. Non-English comments should be translated or normalised into English before prediction if they are used outside the prepared dataset.

The sentiment classification task is challenging because YouTube comments may contain:

* Informal spelling
* Slang
* Emojis
* Sarcasm
* Political expressions
* Short or context-dependent comments
* Translation-related meaning changes

Some predictions may still be uncertain or incorrect, especially for sarcastic, vague, or highly context-dependent comments.

---

## How to Use the Dashboard

### Single Comment Analysis

1. Open the dashboard.
2. Select **Single Comment Analysis**.
3. Type or paste one English comment.
4. Click **Analyze Sentiment**.
5. View the predicted sentiment, confidence score, and sentiment score chart.

### Batch Comment Analysis

1. Open the dashboard.
2. Select **Batch Comment Analysis**.
3. Upload a CSV file containing English comments.
4. Select the text column.
5. Click **Run Sentiment Analysis**.
6. View the sentiment distribution charts.
7. Download the prediction results as a CSV file.

---

## Project Status

The project implementation includes:

* Data collection
* Data cleaning
* English-normalised dataset preparation
* Manual review and relabelling
* Final dataset balancing
* Model evaluation
* Best-model saving
* Streamlit dashboard deployment

The system is ready for demonstration and academic submission.

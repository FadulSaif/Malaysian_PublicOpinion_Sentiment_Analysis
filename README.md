# Malaysia YouTube Sentiment Analysis System

## Project Overview

This project is a sentiment analysis system for Malaysia-related YouTube comments. The system classifies public comments into three sentiment categories:

* Negative
* Neutral
* Positive

The project focuses on the topic **Social Media & Public Opinion (Malaysia Trends)**. The final dataset was collected through the **YouTube Data API v3**, cleaned, manually reviewed, balanced, and used to evaluate multiple sentiment classification models.

The final deployed system is an interactive Streamlit dashboard that allows users to analyze individual comments or upload a CSV file for batch sentiment analysis.

---

## Main Features

The system includes:

* YouTube comment data collection
* Text preprocessing and cleaning
* Manual sentiment annotation
* Final balanced dataset preparation
* Machine-learning and transformer model evaluation
* Single-comment sentiment prediction
* Batch CSV sentiment analysis
* Sentiment distribution visualizations
* Downloadable prediction results

---

## Final Dataset Summary

The final balanced dataset contains:

| Sentiment | Number of Comments |
| --------- | -----------------: |
| Negative  |                275 |
| Neutral   |                275 |
| Positive  |                275 |
| **Total** |            **825** |

The dataset is stored at:

```text
Data/processed/final_youtube_sentiment_825_balanced.csv
```

A full manually reviewed master dataset is also included at:

```text
Data/annotation/youtube_manual_review_completed.csv
```

---

## Data Source

The comments were collected from selected Malaysia-related YouTube videos using the official YouTube Data API v3.

The selected videos cover several Malaysia-related public-opinion themes, including:

* Cost of living and public policy
* Transport and public services
* Festivals and national events
* Tourism and lifestyle
* Technology and digital services

The collected comments were cleaned and manually reviewed before being used for training and evaluation.

---

## Models Evaluated

Several models were evaluated during the project:

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. Word + Character TF-IDF + Logistic Regression
4. Word + Character TF-IDF + Linear SVM
5. Pretrained XLM-RoBERTa multilingual sentiment model

The pretrained XLM-RoBERTa multilingual sentiment model achieved the best overall performance in the experiment and was selected for the final dashboard prediction system.

The Word + Character TF-IDF Logistic Regression model was also trained and saved as the best lightweight classical machine-learning model.

Saved classical model:

```text
models/best_model_word_char_logistic_regression.joblib
```

---

## Final Dashboard

The final Streamlit dashboard provides two main functions:

### 1. Single Comment Analysis

The user enters one comment, and the system predicts whether the comment is negative, neutral, or positive. The dashboard also displays the model confidence and probability distribution.

### 2. Batch Comment Analysis

The user uploads a CSV file containing multiple comments. The system predicts the sentiment for each comment, displays visual summaries, and allows the user to download the prediction results.

---

## Project Folder Structure

```text
malaysia_sentiment_project/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── model_training_and_evaluation.ipynb
│
├── Data/
│   ├── raw/
│   │   ├── youtube_comments_raw.csv
│   │   └── youtube_comment_collection_summary.csv
│   │
│   ├── interim/
│   │   ├── youtube_comments_cleaned.csv
│   │   ├── youtube_comments_rejected.csv
│   │   └── youtube_cleaning_summary.csv
│   │
│   ├── annotation/
│   │   └── youtube_manual_review_completed.csv
│   │
│   └── processed/
│       ├── final_youtube_sentiment_825_balanced.csv
│       ├── final_youtube_sentiment_825_summary.csv
│       ├── removed_extra_positive_negative_rows.csv
│       ├── youtube_selected_videos_final.csv
│       └── youtube_final_selection_summary.csv
│
├── Source_Codes/
│   ├── collection/
│   │   ├── search_youtube_videos.py
│   │   ├── collect_youtube_comments.py
│   │   └── create_final_youtube_video_list.py
│   │
│   ├── preprocessing/
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
│   └── best_model_word_char_logistic_regression.joblib
│
├── results/
│
└── logs/
    └── youtube_comment_collection_log.csv
```

Some folders may contain fewer files depending on the cleaned submission version.

---

## Installation

Open the terminal inside the project folder:

```cmd
cd C:\Users\FadSaif\Documents\malaysia_sentiment_project
```

Install the required libraries:

```cmd
python -m pip install -r requirements.txt
```

If PyTorch or transformer loading causes issues, reinstall the main transformer dependencies:

```cmd
python -m pip install --upgrade torch transformers sentencepiece safetensors
```

If `torchvision` causes import errors, remove it because this project does not use image processing:

```cmd
python -m pip uninstall -y torchvision torchaudio
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

The first run may take longer because the XLM-RoBERTa model may need to be downloaded from Hugging Face.

---

## Running the Notebook

The model training and evaluation notebook is:

```text
model_training_and_evaluation.ipynb
```

Open it using Jupyter Notebook:

```cmd
jupyter notebook
```

Then open:

```text
model_training_and_evaluation.ipynb
```

The notebook contains the dataset loading, model training, model evaluation, comparison results, and model-saving steps.

---

## Important Files

| File                                                      | Purpose                                |
| --------------------------------------------------------- | -------------------------------------- |
| `Source_Codes/deployment/app.py`                          | Final Streamlit dashboard              |
| `Data/processed/final_youtube_sentiment_825_balanced.csv` | Final balanced dataset                 |
| `Data/annotation/youtube_manual_review_completed.csv`     | Full manually reviewed dataset         |
| `model_training_and_evaluation.ipynb`                     | Model training and evaluation notebook |
| `models/best_model_word_char_logistic_regression.joblib`  | Saved classical ML model               |
| `requirements.txt`                                        | Python package requirements            |

---

## Environment Variables

The original YouTube data collection required a YouTube API key. For safety, the real `.env` file should not be submitted.

Use `.env.example` as a template:

```env
YOUTUBE_API_KEY=your_youtube_api_key_here
```

The dashboard does not require the YouTube API key because it uses the already collected and processed data.

---

## Limitations

This system is based on selected Malaysia-related YouTube comments. Therefore, the results should not be interpreted as representing the entire Malaysian population or all social media platforms.

The sentiment classification task is challenging because YouTube comments may contain:

* Malay-English code-switching
* Informal spelling
* Slang
* Emojis
* Sarcasm
* Political expressions
* Short or context-dependent comments

Although the transformer-based model achieved the best performance, some predictions may still be uncertain or incorrect, especially for sarcastic or highly informal comments.

---

## How to Use the Dashboard

### Single Comment Analysis

1. Open the dashboard.
2. Select **Single Comment Analysis**.
3. Type or paste one comment.
4. Click **Analyze Sentiment**.
5. View the predicted sentiment, confidence score, and probability chart.

### Batch Comment Analysis

1. Open the dashboard.
2. Select **Batch Comment Analysis**.
3. Upload a CSV file containing comments.
4. Select the text column.
5. Click **Run Sentiment Analysis**.
6. View the sentiment distribution charts.
7. Download the prediction results as a CSV file.

---

## Project Status

The project implementation includes:

* Data collection
* Data cleaning
* Manual annotation
* Final dataset balancing
* Model evaluation
* Streamlit dashboard deployment

The system is ready for demonstration and academic submission.

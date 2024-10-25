# João's Portfolio

Welcome to my GitHub portfolio! I’m João, an aspiring Data Scientist with a background in Accounting.

In here, you’ll find projects and analyses I’ve worked on, including my master’s thesis about fraud detection. All the projects are open-source, and as such are available under the MIT License.

## Skills and Technologies
- **Languages**: Python, R, SQL
- **Libraries**: Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, TensorFlow
- **Tools**: Jupyter Notebooks

## Projects

### 1. [Master’s Thesis: Decoding the numbers and language behind financial statement fraud](https://github.com/JoaoBrasOliveira/masters_thesis)
Predicting stock price movements using various time series forecasting techniques. Techniques used include ARIMA, Prophet, and LSTM neural networks.

### 2. [Ethereum price prediction](https://github.com/JoaoBrasOliveira/ethereum)
Used time-series techniques to predict in and out-of-sample prices for the cryptocurrency Ethereum customer segments for a retail dataset.
Key skills: clustering, dimensionality reduction, data preprocessing.
<img src="images/zoltan-tasi-uNXmhzcQjxg-unsplash.jpg" alt="Ethereum Price Prediction" width="500" />

### 3. [Sentiment Analysis on Social Media](https://github.com/JoaoBrasOliveira/amazonreviews_sentiment_analysis)
Applied natural language processing to analyze sentiment in reviews posted within Amazon´s marketplace using machine learning models.
Tools used: NLTK, Scikit-learn.
<img src="images/mahdi-bafande-qgJ1rt7TeeY-unsplash.jpg" alt="Amazon Reviews Sentiment Analysis" width="500" />

# Master’s Thesis: Decoding the numbers and language behind financial statement fraud

## Project Overview

**Introduction:** This project utilizes machine learning, deep learning, and Large Language Models (LLMs) to detect financial fraud. It's based on a comprehensive dataset derived from financial filings to the U.S. Securities and Exchange Commission (SEC), aiming to compare and enhance AI models in identifying fraudulent financial activities. (For more information checkout the [pdf](https://github.com/amitkedia007/Financial-Fraud-Detection-Using-LLMs/blob/main/Detailed_Report_on_financial_fraud_detection.pdf) in the repo)

**Objective:** The goal is to foster a collaborative platform where data scientists and researchers can develop, test, and improve AI models for detecting financial fraud.

## Dataset Description

**Source:** The dataset includes financial filings from 170 companies, split equally between those involved in fraudulent and non-fraudulent activities.

**Structure:** Each dataset entry contains details such as Central Index Key (CIK), filing year, company name, and a categorical indicator of fraud.

## Data Preprocessing

Preprocessing steps involve text cleaning, tokenization, and transforming data into machine-readable formats, ensuring balanced and fair model training.

## Model Implementation

The project encompasses a variety of models, including Logistic Regression, XGBoost, BERT and FinBERT, selected for their NLP capabilities and potential in financial statement fraud detection.

## To Reproduce

**Codebase:** Complete code for data extraction, preprocessing, model training, and evaluation is available in this repository.

**Environment:** A `requirements.txt` file is provided for setting up a consistent environment.

**Documentation:** Each script is documented with clear instructions in the `README.md`, guiding through environment setup, script execution, and result interpretation.

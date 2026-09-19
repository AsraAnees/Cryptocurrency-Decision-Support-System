# Cryptocurrency Decision Support System

A machine learning-based **Decision Support System (DSS)** for cryptocurrency prediction and analysis. The project investigates how sentiment-related information can be combined with cryptocurrency features to support short-term price prediction and model comparison.

## Project Overview

The system combines cryptocurrency data, sentiment data, and a target variable into a single dataset for analysis. Text sentiment is converted into a numerical score using the **VADER sentiment analyzer** from NLTK and used as an additional predictive feature.

The project focuses on two main objectives:

### Sentiment-Based Prediction

The first objective examines the role of sentiment in cryptocurrency prediction. Sentiment scores are combined with cryptocurrency features and used to train three regression models:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### Machine Learning Model Evaluation

The second objective evaluates machine learning approaches for short-term cryptocurrency prediction using cryptocurrency-specific features. Random Forest and Gradient Boosting regression models are trained and compared.

## Workflow

The project follows a machine learning workflow consisting of data loading, dataset merging, preprocessing, sentiment analysis, feature selection, train/test splitting, model training, evaluation, and visualization.

The datasets are merged using the `date` field and rows containing missing values are removed. The data is divided into **80% training data and 20% testing data**.

## Model Evaluation

Model performance is evaluated using:

- **Mean Squared Error (MSE)** to measure prediction error.
- **R-squared (R²)** to measure how much variation in the target variable is explained by the model.

Scatter plots of actual versus predicted values are also generated to visually compare model predictions.

## Technologies

Python, pandas, NumPy, scikit-learn, NLTK/VADER, Matplotlib, and Seaborn are used for data processing, sentiment analysis, machine learning, evaluation, and visualization.

## Project Report

The accompanying report, **“Integrating Sentiment and Interest-Based Features for Accurate Cryptocurrency Price Prediction,”** documents the project objectives, analysis, system design, implementation, evaluation, visualizations, and conclusions.

> This project was developed for educational purposes as a Decision Support System project. Cryptocurrency markets are highly volatile, and the predictions produced by these models should not be treated as financial advice.

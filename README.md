# Cryptocurrency Decision Support System

A machine learning-based **Decision Support System (DSS)** for cryptocurrency prediction and analysis. The project investigates how market-related features and sentiment information can be combined to support short-term cryptocurrency prediction.

## Sentiment-Based Prediction

Textual sentiment is analyzed using the **VADER sentiment analyzer** from NLTK. The resulting compound sentiment score is combined with a cryptocurrency-specific feature and used as input to **Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor** models.

This part of the project examines whether sentiment information can contribute useful information to cryptocurrency prediction.

## Machine Learning Model Evaluation

The second objective focuses on short-term prediction using cryptocurrency-specific data. **Random Forest** and **Gradient Boosting** regression models are trained and compared to examine their predictive performance.

The three datasets—cryptocurrency data, sentiment data, and the target variable—are merged using their date field. Missing values are removed, and the data is divided into **80% training and 20% testing data**.

Model performance is evaluated using **Mean Squared Error (MSE)** and **R-squared (R²)**. Scatter plots of actual versus predicted values provide a visual comparison of model behaviour and prediction accuracy.

The project uses **Python, pandas, NumPy, scikit-learn, NLTK/VADER, Matplotlib, and Seaborn** for data processing, sentiment analysis, machine learning, evaluation, and visualization.

> This project was developed for educational purposes as a Decision Support System project. Cryptocurrency markets are highly volatile, and the predictions produced by these models should not be treated as financial advice.
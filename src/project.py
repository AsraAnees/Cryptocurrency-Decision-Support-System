import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import random
import nltk

nltk.download('vader_lexicon')

# Read Datasets from CSV
cryptocurrency_data = pd.read_csv('cryptocurrency_data.csv')
sentiment_data = pd.read_csv('sentiment_data.csv')
target_variable = pd.read_csv('target_variable.csv')

# Data Merging
merged_data = pd.merge(cryptocurrency_data, sentiment_data, on='date')
merged_data = pd.merge(merged_data, target_variable, on='date')

# Preprocessing
merged_data.dropna(inplace=True)
merged_data['sentiment_score'] = merged_data['text'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])

# Objective 1: Sentiment Analysis
# Feature Selection
selected_features_obj1 = ['sentiment_score', 'cryptocurrency_feature']

# Model Selection and Training for Objective 1
X_obj1 = merged_data[selected_features_obj1]
y_obj1 = merged_data['target_variable_name']

X_train_obj1, X_test_obj1, y_train_obj1, y_test_obj1 = train_test_split(X_obj1, y_obj1, test_size=0.2, random_state=42)

linear_model_obj1 = LinearRegression()
linear_model_obj1.fit(X_train_obj1, y_train_obj1)

rf_model_obj1 = RandomForestRegressor()
rf_model_obj1.fit(X_train_obj1, y_train_obj1)

gb_model_obj1 = GradientBoostingRegressor()
gb_model_obj1.fit(X_train_obj1, y_train_obj1)

# Model Evaluation for Objective 1
linear_pred_obj1 = linear_model_obj1.predict(X_test_obj1)
linear_mse_obj1 = mean_squared_error(y_test_obj1, linear_pred_obj1)
linear_r2_obj1 = r2_score(y_test_obj1, linear_pred_obj1)

rf_pred_obj1 = rf_model_obj1.predict(X_test_obj1)
rf_mse_obj1 = mean_squared_error(y_test_obj1, rf_pred_obj1)
rf_r2_obj1 = r2_score(y_test_obj1, rf_pred_obj1)

gb_pred_obj1 = gb_model_obj1.predict(X_test_obj1)
gb_mse_obj1 = mean_squared_error(y_test_obj1, gb_pred_obj1)
gb_r2_obj1 = r2_score(y_test_obj1, gb_pred_obj1)

print("Objective 1 - Linear Regression - Mean Squared Error:", linear_mse_obj1)
print("Objective 1 - Linear Regression - R-squared Score:", linear_r2_obj1)
print("Objective 1 - Random Forest Regressor - Mean Squared Error:", rf_mse_obj1)
print("Objective 1 - Random Forest Regressor - R-squared Score:", rf_r2_obj1)
print("Objective 1 - Gradient Boosting Regressor - Mean Squared Error:", gb_mse_obj1)
print("Objective 1 - Gradient Boosting Regressor - R-squared Score:", gb_r2_obj1)

# Visualization for Objective 1
plt.scatter(y_test_obj1, linear_pred_obj1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Objective 1 - Linear Regression - Actual vs. Predicted Values")
plt.show()

plt.scatter(y_test_obj1, rf_pred_obj1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Objective 1 - Random Forest Regressor - Actual vs. Predicted Values")
plt.show()

plt.scatter(y_test_obj1, gb_pred_obj1)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Objective 1 - Gradient Boosting Regressor - Actual vs. Predicted Values")
plt.show()


# Objective 2: ML Algorithm Evaluation
# Feature Selection
selected_features_obj2 = ['cryptocurrency_feature']

# Model Selection and Training for Objective 2
X_obj2 = merged_data[selected_features_obj2]
y_obj2 = merged_data['target_variable_name']

X_train_obj2, X_test_obj2, y_train_obj2, y_test_obj2 = train_test_split(X_obj2, y_obj2, test_size=0.2, random_state=42)

rf_model_obj2 = RandomForestRegressor()
rf_model_obj2.fit(X_train_obj2, y_train_obj2)

gb_model_obj2 = GradientBoostingRegressor()
gb_model_obj2.fit(X_train_obj2, y_train_obj2)

# Model Evaluation for Objective 2
rf_pred_obj2 = rf_model_obj2.predict(X_test_obj2)
rf_mse_obj2 = mean_squared_error(y_test_obj2, rf_pred_obj2)
rf_r2_obj2 = r2_score(y_test_obj2, rf_pred_obj2)

gb_pred_obj2 = gb_model_obj2.predict(X_test_obj2)
gb_mse_obj2 = mean_squared_error(y_test_obj2, gb_pred_obj2)
gb_r2_obj2 = r2_score(y_test_obj2, gb_pred_obj2)

print("Objective 2 - Random Forest Regressor - Mean Squared Error:", rf_mse_obj2)
print("Objective 2 - Random Forest Regressor - R-squared Score:", rf_r2_obj2)
print("Objective 2 - Gradient Boosting Regressor - Mean Squared Error:", gb_mse_obj2)
print("Objective 2 - Gradient Boosting Regressor - R-squared Score:", gb_r2_obj2)

# Visualization for Objective 2
plt.scatter(y_test_obj2, rf_pred_obj2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Objective 2 - Random Forest Regressor - Actual vs. Predicted Values")
plt.show()

plt.scatter(y_test_obj2, gb_pred_obj2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Objective 2 - Gradient Boosting Regressor - Actual vs. Predicted Values")
plt.show()





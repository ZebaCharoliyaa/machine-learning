# House Price Prediction using Multiple Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib

# Load Dataset
df = pd.read_csv(r'C:\Users\ZEBA CHAROLIYA\Desktop\machine learning\housing.csv')  # Replace with actual path
df = df.dropna()

# Feature Selection (at least 3 features)
X = df[['RM', 'LSTAT', 'PTRATIO']]
y = df['MEDV']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation Metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")

# Save Model (optional)
joblib.dump(model, 'house_price_model.joblib')

# Plot 1: Simple Regression Line (RM vs MEDV)
plt.figure(figsize=(6, 4))
sns.regplot(x=df['RM'], y=df['MEDV'], line_kws={"color": "red"})
plt.title('Regression Line: RM vs MEDV')
plt.xlabel('Average Rooms per Dwelling (RM)')
plt.ylabel('House Price (MEDV)')
plt.tight_layout()
plt.savefig('regression_line.png')
plt.close()

# Plot 2: Actual vs Predicted
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()

# Plot 3: Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
sns.residplot(x=y_pred, y=residuals, color="g")  # Removed lowess=True to avoid statsmodels error
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig('residual_plot.png')
plt.close()

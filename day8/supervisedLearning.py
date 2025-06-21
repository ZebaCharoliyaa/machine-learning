# My First Machine Learning Model - Diabetes Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

# Load the dataset
data = load_diabetes()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Preview the data
print(x.head())
print("Target variable (y):", y[:5])

# Train-test split (for regression)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
y_pred = lin_reg.predict(x_test)

# Evaluation - Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (Linear Regression):", mse)

# Visualization
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression: Actual vs Predicted')
plt.grid(True)
plt.savefig("regression_plot.png")
plt.show()

# OPTIONAL - Convert to classification: Binary classification (above/below median)
y_binary = (y > np.median(y)).astype(int)
x_train_clf, x_test_clf, y_train_clf, y_test_clf = train_test_split(x, y_binary, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train_clf, y_train_clf)
y_pred_clf = log_reg.predict(x_test_clf)

# Accuracy & Confusion Matrix
accuracy = accuracy_score(y_test_clf, y_pred_clf)
cm = confusion_matrix(y_test_clf, y_pred_clf)

print("Classification Accuracy (Logistic Regression):", accuracy)
print("Confusion Matrix:\n", cm)

# Save confusion matrix as image
fig, ax = plt.subplots()
cax = ax.matshow(cm, cmap='coolwarm')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("confusion_matrix.png")
plt.show()

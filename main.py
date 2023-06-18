import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


data = pd.read_csv('dataset.csv')


print("Dataset shape:", data.shape)
print("Dataset summary:", data.describe())


X = data[['feature1', 'feature2']].values
y = data['target'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


plt.scatter(X_test[:, 0], y_test, color='b', label='Actual')
plt.scatter(X_test[:, 0], y_pred, color='r', label='Predicted')
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.legend()
plt.show()

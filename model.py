import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import pickle

df = pd.read_csv('MagicBricks.csv')
df.head()

df.info()

df.describe()

# 1. Handling missing values
df = df.fillna(df.median(numeric_only=True))

# 2. Treating outlier (using IQR)
num_col = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in num_col:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

cat_col = df.select_dtypes(include=['object']).columns.tolist()

dropdown_options = {col: sorted(df[col].dropna().unique()) for col in cat_col if col in df.columns}
with open('dropdown_options.pkl', 'wb') as f:
    pickle.dump(dropdown_options, f)

# 3. One-hot encoding for categorical columns

df = pd.get_dummies(df, columns=cat_col, drop_first=True)

# 4. Feature and target variable separation
x = df.drop('Price', axis=1)  # Use all columns except 'price'
y = df['Price']

# 5. Splitting the dataset into training and testing sets
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.2,random_state = 2000)

x_train

y_train

# Feature Scaling

std = StandardScaler()
x_train_std = std.fit_transform(x_train)
x_train_std

x_test_std = std.transform(x_test)
x_test_std

# Model creation
model = LinearRegression()

model.fit(x_train_std, y_train)

# Predict on test data
y_pred = model.predict(x_test_std)
y_pred

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Save the model and scaler
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(std, f)

feature_names = x.columns.tolist()
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

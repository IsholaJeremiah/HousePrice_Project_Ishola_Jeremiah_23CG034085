import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle

# 1. Load Dataset (CSV must contain Ames dataset)
data = pd.read_csv("./model/train2.csv")

# 2. Feature Selection (6 features from allowed 9)
features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt']
target = 'SalePrice'

df = data[features + [target]]

# 3. Handle Missing Values
df = df.fillna(df.median())

# 4. Split Features & Target
X = df[features]
y = df[target]

# 5. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Model Selection (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 8. Predictions & Evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MODEL EVALUATION RESULTS:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# 9. Save Model & Scaler
with open("house_price_model.pkl", 'wb') as file:
    pickle.dump(model, file)

with open("scaler.pkl", 'wb') as file:
    pickle.dump(scaler, file)

print("Model & Scaler Saved Successfully!")

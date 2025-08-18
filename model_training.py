import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("quikr_car.csv")
print("✅ Dataset loaded successfully")
print("Columns in dataset:", df.columns)

# -------------------------------
# 2. Data Cleaning
# -------------------------------

# Remove rows with missing values
df = df.dropna()

# Clean kms_driven (remove 'kms' and commas, convert to int)
df["kms_driven"] = df["kms_driven"].str.replace(" kms", "").str.replace(",", "")
df["kms_driven"] = pd.to_numeric(df["kms_driven"], errors="coerce")

# Clean Price (remove 'Rs.' and commas, convert to int)
df["Price"] = df["Price"].str.replace("Rs.", "").str.replace(",", "")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Keep only relevant columns
df = df[["year", "kms_driven", "fuel_type", "company", "Price"]]

# Drop rows where Price is missing
df = df.dropna()

# -------------------------------
# 3. Split features & target
# -------------------------------
X = df[["year", "kms_driven", "fuel_type", "company"]]
y = df["Price"]

# -------------------------------
# 4. Preprocessing
# -------------------------------
categorical_features = ["fuel_type", "company"]
numeric_features = ["year", "kms_driven"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# -------------------------------
# 5. Build Pipeline
# -------------------------------
pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42))
])

# -------------------------------
# 6. Train Model
# -------------------------------
pipe.fit(X, y)

# -------------------------------
# 7. Save Model
# -------------------------------
with open("car_price_model.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("✅ Model trained & saved as car_price_model.pkl")

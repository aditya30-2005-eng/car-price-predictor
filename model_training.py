import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("quikr_car.csv")
print("✅ Dataset loaded successfully")
print("Columns in dataset:", df.columns)

df = df.dropna()
df["kms_driven"] = df["kms_driven"].str.replace(" kms", "").str.replace(",", "")
df["kms_driven"] = pd.to_numeric(df["kms_driven"], errors="coerce")
df["Price"] = df["Price"].str.replace("Rs.", "").str.replace(",", "")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df[["year", "kms_driven", "fuel_type", "company", "Price"]]
df = df.dropna()

X = df[["year", "kms_driven", "fuel_type", "company"]]
y = df["Price"]

categorical_features = ["fuel_type", "company"]
numeric_features = ["year", "kms_driven"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42))
])

pipe.fit(X, y)

with open("car_price_model.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("✅ Model trained & saved as car_price_model.pkl")

from pathlib import Path

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "splits"
    / "train.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "random_forest_pipeline.pkl"


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

print("Loading training data...")

train_df = pd.read_csv(TRAIN_PATH)

TARGET = "Irrigation_Need"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]


# ============================================================
# 3. IDENTIFY FEATURES
# ============================================================

categorical_features = X_train.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

numeric_features = X_train.select_dtypes(
    exclude=["object", "category", "str"]
).columns.tolist()


# ============================================================
# 4. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numeric_features
        )
    ]
)


# ============================================================
# 5. TUNED RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# ============================================================
# 6. CREATE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("random_forest", model)
    ]
)


# ============================================================
# 7. TRAIN
# ============================================================

print("Training tuned Random Forest...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 8. SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nModel saved successfully.")

print(f"Model path:")
print(MODEL_PATH)

print("\n" + "=" * 60)
print("RANDOM FOREST MODEL SAVED")
print("=" * 60)
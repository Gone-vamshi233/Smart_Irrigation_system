from pathlib import Path

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "splits"
    / "train.csv"
)

TEST_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "splits"
    / "test.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGET = "Irrigation_Need"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


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
# 4. PREPROCESSING
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
# 5. GRADIENT BOOSTING MODEL
# ============================================================

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)


# ============================================================
# 6. PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("gradient_boosting", model)
    ]
)


# ============================================================
# 7. TRAIN
# ============================================================

print("Training Gradient Boosting...")

pipeline.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# 8. TEST PREDICTION
# ============================================================

y_pred = pipeline.predict(X_test)


# ============================================================
# 9. ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:")
print(f"{accuracy:.4f}")


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

labels = ["High", "Medium", "Low"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")

print(
    "               Predicted High  "
    "Predicted Medium  "
    "Predicted Low"
)

for label, row in zip(labels, cm):
    print(
        f"Actual {label:<8}"
        f"{row[0]:>16}"
        f"{row[1]:>19}"
        f"{row[2]:>17}"
    )


# ============================================================
# 12. FINAL METRICS
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

print("\nFinal Test Metrics:")
print(f"Accuracy    : {accuracy:.4f}")
print(f"High Recall : {report['High']['recall']:.4f}")
print(f"High F1     : {report['High']['f1-score']:.4f}")


print("\n" + "=" * 70)
print("GRADIENT BOOSTING TEST EVALUATION COMPLETED")
print("=" * 70)
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "splits"


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")


# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

target = "Irrigation_Need"

X_train = train_df.drop(columns=[target])
y_train = train_df[target]

X_test = test_df.drop(columns=[target])
y_test = test_df[target]


# --------------------------------------------------
# 4. Identify feature types
# --------------------------------------------------

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# --------------------------------------------------
# 5. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# --------------------------------------------------
# 6. Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 7. Train using training data only
# --------------------------------------------------

print("=" * 60)
print("RANDOM FOREST - FINAL TEST EVALUATION")
print("=" * 60)

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# 8. Test prediction
# --------------------------------------------------

y_pred = pipeline.predict(X_test)


# --------------------------------------------------
# 9. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nTest Accuracy:")
print(f"{accuracy:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# 10. Confusion Matrix
# --------------------------------------------------

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["High", "Medium", "Low"]
)

print(
    pd.DataFrame(
        cm,
        index=["Actual High", "Actual Medium", "Actual Low"],
        columns=["Predicted High", "Predicted Medium", "Predicted Low"]
    )
)


print("\n" + "=" * 60)
print("TEST EVALUATION COMPLETED")
print("=" * 60)
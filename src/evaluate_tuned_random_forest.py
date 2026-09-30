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
# 2. Load training and test data
# --------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")


print("=" * 70)
print("SMART IRRIGATION - TUNED RANDOM FOREST TEST EVALUATION")
print("=" * 70)


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
# 6. Tuned Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# --------------------------------------------------
# 7. Create pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 8. Train on training data
# --------------------------------------------------

print("\nTraining tuned Random Forest...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# --------------------------------------------------
# 9. Predict on untouched test data
# --------------------------------------------------

y_pred = pipeline.predict(X_test)


# --------------------------------------------------
# 10. Accuracy
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nTest Accuracy:")
print(f"{accuracy:.4f}")


# --------------------------------------------------
# 11. Classification report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# 12. Confusion Matrix
# --------------------------------------------------

labels = ["High", "Medium", "Low"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {label}" for label in labels],
    columns=[f"Predicted {label}" for label in labels]
)

print(cm_df)


# --------------------------------------------------
# 13. Final summary
# --------------------------------------------------

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

print("\nFinal Test Metrics:")
print(f"Accuracy    : {accuracy:.4f}")
print(f"High Recall : {report['High']['recall']:.4f}")
print(f"High F1     : {report['High']['f1-score']:.4f}")

print("\n" + "=" * 70)
print("TUNED RANDOM FOREST TEST EVALUATION COMPLETED")
print("=" * 70)
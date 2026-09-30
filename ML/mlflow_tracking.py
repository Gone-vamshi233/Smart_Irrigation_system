from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "processed" / "splits" / "train.csv"
VAL_PATH = BASE_DIR / "data" / "processed" / "splits" / "validation.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading training and validation data...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

TARGET = "Irrigation_Need"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)


# ============================================================
# 3. IDENTIFY FEATURES
# ============================================================

categorical_features = X_train.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

numeric_features = X_train.select_dtypes(
    exclude=["object", "category"]
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
# 7. MLflow EXPERIMENT
# ============================================================

mlflow.set_experiment("Smart-Irrigation-ML")

print("\nStarting MLflow experiment...")
print("Training tuned Random Forest...")


# ============================================================
# 8. START MLflow RUN
# ============================================================

with mlflow.start_run(run_name="Tuned-Random-Forest"):

    # -----------------------------
    # Train model
    # -----------------------------

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    # -----------------------------
    # Validation prediction
    # -----------------------------

    y_pred = pipeline.predict(X_val)

    # -----------------------------
    # Calculate metrics
    # -----------------------------

    accuracy = accuracy_score(y_val, y_pred)

    report = classification_report(
        y_val,
        y_pred,
        output_dict=True
    )

    high_recall = report["High"]["recall"]
    high_f1 = report["High"]["f1-score"]

    medium_recall = report["Medium"]["recall"]
    low_recall = report["Low"]["recall"]

    # ========================================================
    # 9. LOG PARAMETERS
    # ========================================================

    mlflow.log_param(
        "model_type",
        "RandomForestClassifier"
    )

    mlflow.log_param(
        "n_estimators",
        300
    )

    mlflow.log_param(
        "max_depth",
        20
    )

    mlflow.log_param(
        "min_samples_split",
        2
    )

    mlflow.log_param(
        "min_samples_leaf",
        1
    )

    mlflow.log_param(
        "class_weight",
        "balanced"
    )

    mlflow.log_param(
        "random_state",
        42
    )

    # ========================================================
    # 10. LOG METRICS
    # ========================================================

    mlflow.log_metric(
        "validation_accuracy",
        accuracy
    )

    mlflow.log_metric(
        "high_recall",
        high_recall
    )

    mlflow.log_metric(
        "high_f1",
        high_f1
    )

    mlflow.log_metric(
        "medium_recall",
        medium_recall
    )

    mlflow.log_metric(
        "low_recall",
        low_recall
    )

    # ========================================================
    # 11. LOG MODEL
    # ========================================================

    mlflow.sklearn.log_model(
    pipeline,
    name="tuned_random_forest",
    skops_trusted_types=[
        "sklearn.tree._tree.Tree"
    ]
    )

    # ========================================================
    # 12. DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("MLFLOW EXPERIMENT RESULTS")
    print("=" * 60)

    print(f"Validation Accuracy : {accuracy:.4f}")
    print(f"High Recall         : {high_recall:.4f}")
    print(f"High F1             : {high_f1:.4f}")
    print(f"Medium Recall       : {medium_recall:.4f}")
    print(f"Low Recall          : {low_recall:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_val,
            y_pred
        )
    )

    # ========================================================
    # 13. RUN INFORMATION
    # ========================================================

    run_id = mlflow.active_run().info.run_id

    print("MLflow Run ID:", run_id)


print("\n" + "=" * 60)
print("MLFLOW TRACKING COMPLETED")
print("=" * 60)
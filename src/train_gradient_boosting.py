import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "splits"


# --------------------------------------------------
# 2. Load train and validation data
# --------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")


print("=" * 60)
print("SMART IRRIGATION - GRADIENT BOOSTING")
print("=" * 60)


# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

target = "Irrigation_Need"

X_train = train_df.drop(columns=[target])
y_train = train_df[target]

X_val = val_df.drop(columns=[target])
y_val = val_df[target]


print("\nTraining Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)


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
# 6. Gradient Boosting model
# --------------------------------------------------

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)


# --------------------------------------------------
# 7. Pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 8. Train
# --------------------------------------------------

print("\nTraining Gradient Boosting...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# --------------------------------------------------
# 9. Validation prediction
# --------------------------------------------------

y_pred = pipeline.predict(X_val)


# --------------------------------------------------
# 10. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(
    y_val,
    y_pred
)

print("\nValidation Accuracy:")
print(f"{accuracy:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_pred,
        zero_division=0
    )
)


print("=" * 60)
print("GRADIENT BOOSTING COMPLETED")
print("=" * 60)
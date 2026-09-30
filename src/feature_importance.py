import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "splits"


# --------------------------------------------------
# 2. Load training data
# --------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")

target = "Irrigation_Need"

X_train = train_df.drop(columns=[target])
y_train = train_df[target]


# --------------------------------------------------
# 3. Identify feature types
# --------------------------------------------------

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# --------------------------------------------------
# 4. Preprocessing
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
# 5. Random Forest
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
# 6. Train
# --------------------------------------------------

print("Training Random Forest...")

pipeline.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# 7. Get feature names
# --------------------------------------------------

feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()


# --------------------------------------------------
# 8. Get feature importance
# --------------------------------------------------

importances = pipeline.named_steps[
    "model"
].feature_importances_


importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# --------------------------------------------------
# 9. Display top features
# --------------------------------------------------

print("\nTop 20 Important Features:")
print(
    importance_df.head(20).to_string(index=False)
)


# --------------------------------------------------
# 10. Plot
# --------------------------------------------------

top_features = importance_df.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 15 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()
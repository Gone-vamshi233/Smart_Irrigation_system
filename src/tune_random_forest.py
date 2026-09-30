import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "splits"


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

target = "Irrigation_Need"

X_train = train_df.drop(columns=[target])
y_train = train_df[target]

X_val = val_df.drop(columns=[target])
y_val = val_df[target]


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
# 5. Parameter combinations
# --------------------------------------------------

parameter_sets = [
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },
    {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 1
    },
    {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 2
    }
]


# --------------------------------------------------
# 6. Train and evaluate configurations
# --------------------------------------------------

results = []

print("=" * 70)
print("SMART IRRIGATION - RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)

for i, params in enumerate(parameter_sets, start=1):

    print(f"\nTesting Configuration {i}")
    print(params)

    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        **params
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    report = classification_report(
        y_val,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    high_recall = report["High"]["recall"]
    high_f1 = report["High"]["f1-score"]

    results.append({
        "Configuration": i,
        "Accuracy": accuracy,
        "High_Recall": high_recall,
        "High_F1": high_f1,
        **params
    })

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"High Recall:        {high_recall:.4f}")
    print(f"High F1:            {high_f1:.4f}")


# --------------------------------------------------
# 7. Compare configurations
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["High_F1", "Accuracy"],
    ascending=False
)


print("\n" + "=" * 70)
print("TUNING RESULTS")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# --------------------------------------------------
# 8. Best configuration
# --------------------------------------------------

best = results_df.iloc[0]

print("\n" + "=" * 70)
print("BEST RANDOM FOREST CONFIGURATION")
print("=" * 70)

print(f"Configuration : {int(best['Configuration'])}")
print(f"Accuracy      : {best['Accuracy']:.4f}")
print(f"High Recall   : {best['High_Recall']:.4f}")
print(f"High F1       : {best['High_F1']:.4f}")

print("\nParameters:")

print(
    f"n_estimators     = {int(best['n_estimators'])}"
)

print(
    f"max_depth        = {best['max_depth']}"
)

print(
    f"min_samples_split = {int(best['min_samples_split'])}"
)

print(
    f"min_samples_leaf  = {int(best['min_samples_leaf'])}"
)

print("=" * 70)
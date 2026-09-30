import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "ml_ready_features.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load processed dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("SMART IRRIGATION - ML DATA PREPARATION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)


# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

target = "Irrigation_Need"

X = df.drop(columns=[target])
y = df[target]

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nTarget Distribution:")
print(y.value_counts())


# --------------------------------------------------
# 4. Train/Test split - 80/20
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 5. Validation/Test split - 10/10
# --------------------------------------------------

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# --------------------------------------------------
# 6. Display split sizes
# --------------------------------------------------

print("\nSplit Shapes:")
print("Training   :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing    :", X_test.shape)


# --------------------------------------------------
# 7. Display target distribution
# --------------------------------------------------

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nValidation Target Distribution:")
print(y_val.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())


# --------------------------------------------------
# 8. Save splits
# --------------------------------------------------

train_df = X_train.copy()
train_df[target] = y_train

val_df = X_val.copy()
val_df[target] = y_val

test_df = X_test.copy()
test_df[target] = y_test


train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)


print("\nFiles Saved:")
print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "validation.csv")
print(OUTPUT_DIR / "test.csv")

print("=" * 60)
print("ML DATA PREPARATION COMPLETED")
print("=" * 60)
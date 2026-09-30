import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "splits"


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")


print("=" * 60)
print("SMART IRRIGATION - DATA LEAKAGE CHECK")
print("=" * 60)


# --------------------------------------------------
# Remove target for feature comparison
# --------------------------------------------------

target = "Irrigation_Need"

train_features = train.drop(columns=[target])
validation_features = validation.drop(columns=[target])
test_features = test.drop(columns=[target])


# --------------------------------------------------
# Exact duplicate checks
# --------------------------------------------------

train_set = set(
    map(tuple, train_features.astype(str).values)
)

validation_set = set(
    map(tuple, validation_features.astype(str).values)
)

test_set = set(
    map(tuple, test_features.astype(str).values)
)


train_validation_overlap = len(
    train_set.intersection(validation_set)
)

train_test_overlap = len(
    train_set.intersection(test_set)
)

validation_test_overlap = len(
    validation_set.intersection(test_set)
)


print("\nExact Feature Overlap:")
print(
    "Train vs Validation:",
    train_validation_overlap
)

print(
    "Train vs Test:",
    train_test_overlap
)

print(
    "Validation vs Test:",
    validation_test_overlap
)


# --------------------------------------------------
# Duplicate rows inside each split
# --------------------------------------------------

print("\nDuplicate Rows Within Splits:")

print(
    "Train:",
    train_features.duplicated().sum()
)

print(
    "Validation:",
    validation_features.duplicated().sum()
)

print(
    "Test:",
    test_features.duplicated().sum()
)


print("\n" + "=" * 60)
print("DATA LEAKAGE CHECK COMPLETED")
print("=" * 60)
from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "irrigation.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("LSTM DATASET INSPECTION")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. SEARCH FOR TIME-RELATED COLUMNS
# ============================================================

time_keywords = [
    "date",
    "time",
    "timestamp",
    "day",
    "month",
    "year",
    "hour"
]

time_columns = []

for column in df.columns:

    column_lower = column.lower()

    if any(
        keyword in column_lower
        for keyword in time_keywords
    ):
        time_columns.append(column)


print("\nPossible time-related columns:")

if time_columns:

    for column in time_columns:
        print(f" - {column}")

else:

    print("None found")


# ============================================================
# 4. CHECK FOR FIELD / LOCATION IDENTIFIERS
# ============================================================

field_keywords = [
    "field",
    "farm",
    "plot",
    "location",
    "region",
    "id"
]

field_columns = []

for column in df.columns:

    column_lower = column.lower()

    if any(
        keyword in column_lower
        for keyword in field_keywords
    ):
        field_columns.append(column)


print("\nPossible field/location columns:")

if field_columns:

    for column in field_columns:
        print(f" - {column}")

else:

    print("None found")


# ============================================================
# 5. CHECK UNIQUE VALUES
# ============================================================

print("\nUnique values for possible temporal columns:")

for column in time_columns:

    print(
        f"{column}: "
        f"{df[column].nunique()} unique values"
    )


# ============================================================
# 6. CHECK DUPLICATES
# ============================================================

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 7. CHECK DATA ORDER
# ============================================================

print("\nFirst 5 rows:")

print(
    df.head().to_string()
)


print("\nLast 5 rows:")

print(
    df.tail().to_string()
)


# ============================================================
# 8. FINAL ASSESSMENT
# ============================================================

print("\n" + "=" * 60)
print("LSTM DATA ASSESSMENT")
print("=" * 60)

if not time_columns:

    print(
        "\nNo explicit temporal column was found."
    )

    print(
        "The dataset cannot directly provide "
        "genuine time-series sequences."
    )

else:

    print(
        "\nTemporal columns were found."
    )

    print(
        "Further sequence analysis is required."
    )
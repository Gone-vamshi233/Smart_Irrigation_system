from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "irrigation.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("LSTM EXPERIMENT")
print("=" * 65)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# FIND TEMPORAL COLUMNS
# ============================================================

temporal_keywords = [
    "date",
    "time",
    "timestamp",
    "datetime",
    "day",
    "month",
    "year",
    "hour"
]

temporal_columns = []

for column in df.columns:

    name = column.lower()

    if any(
        keyword in name
        for keyword in temporal_keywords
    ):
        temporal_columns.append(column)


# Remove non-temporal false positives
temporal_columns = [
    column
    for column in temporal_columns
    if column.lower() != "sunlight_hours"
]


print("\nTemporal columns detected:")

if temporal_columns:

    for column in temporal_columns:
        print(f"  - {column}")

else:

    print("  None")


# ============================================================
# CHECK FIELD IDENTIFIER
# ============================================================

possible_field_ids = [
    "field_id",
    "field",
    "farm_id",
    "farm",
    "plot_id",
    "plot"
]

field_columns = [
    column
    for column in df.columns
    if column.lower() in possible_field_ids
]

print("\nField identifier:")

if field_columns:

    for column in field_columns:
        print(f"  - {column}")

else:

    print("  None")


# ============================================================
# LSTM DATA VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("SEQUENCE VALIDATION")
print("=" * 65)


if not temporal_columns:

    print("""
LSTM training cannot be performed on the current dataset.

Reason:
- No timestamp/date column is available.
- No chronological ordering can be established.
- No repeated field-level observations are available.
- Randomly grouping independent records would create
  artificial sequences and would not represent real temporal data.
""")

    print("Status: LSTM RESERVED FOR FUTURE TIME-SERIES DATA")


else:

    print(
        "Temporal information detected."
    )

    print(
        "Further sequence construction is required."
    )


# ============================================================
# REQUIRED DATA FORMAT FOR FUTURE LSTM
# ============================================================

print("\n" + "=" * 65)
print("REQUIRED FUTURE LSTM DATA")
print("=" * 65)

print("""
A valid LSTM dataset should contain observations such as:

Field_ID | Timestamp | Soil_Moisture | Temperature | Rainfall
-------------------------------------------------------------
F001     | 08:00     | 42.1          | 28.5        | 0.0
F001     | 09:00     | 40.7          | 29.2        | 0.0
F001     | 10:00     | 38.4          | 30.1        | 0.0
F001     | 11:00     | 35.9          | 31.4        | 0.2
F001     | 12:00     | 33.5          | 32.0        | 0.5

The chronological sequence can then be used by an LSTM
to learn temporal irrigation patterns.
""")

print("\n" + "=" * 65)
print("LSTM MODULE CHECK COMPLETE")
print("=" * 65)
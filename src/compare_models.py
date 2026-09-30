import pandas as pd


# Validation results from our trained models

results = [
    {
        "Model": "Random Forest",
        "Accuracy": 0.9909,
        "High_Precision": 1.00,
        "High_Recall": 0.92,
        "High_F1": 0.96,
        "Medium_Recall": 1.00,
        "Low_Recall": 0.99
    },
    {
        "Model": "Gradient Boosting",
        "Accuracy": 0.9936,
        "High_Precision": 1.00,
        "High_Recall": 0.87,
        "High_F1": 0.93,
        "Medium_Recall": 1.00,
        "Low_Recall": 1.00
    }
]


comparison = pd.DataFrame(results)

print("=" * 80)
print("SMART IRRIGATION - MODEL COMPARISON")
print("=" * 80)

print("\nValidation Performance:\n")

print(
    comparison.to_string(
        index=False
    )
)

print("\n" + "=" * 80)
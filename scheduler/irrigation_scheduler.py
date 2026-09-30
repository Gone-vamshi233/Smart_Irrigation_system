from pathlib import Path

import pandas as pd
import joblib


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "splits"
    / "test.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest_pipeline.pkl"
)


# ============================================================
# 2. LOAD RANDOM FOREST MODEL
# ============================================================

print("Loading trained Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# 3. ML PREDICTION
# ============================================================

def predict_irrigation_need(row):
    """
    Use the trained Random Forest model to predict
    irrigation requirement for a field.
    """

    # Convert single row into DataFrame
    input_data = pd.DataFrame(
        [row]
    )

    # Remove target if it exists
    if "Irrigation_Need" in input_data.columns:
        input_data = input_data.drop(
            columns=["Irrigation_Need"]
        )

    # Make prediction
    prediction = model.predict(
        input_data
    )[0]

    return prediction


# ============================================================
# 4. IRRIGATION SCHEDULER
# ============================================================

def calculate_irrigation_schedule(
    row,
    prediction
):
    """
    Generate field-specific irrigation
    recommendations using ML prediction
    and environmental conditions.
    """

    # --------------------------------------------------------
    # Field conditions
    # --------------------------------------------------------

    soil_moisture = float(
        row["Soil_Moisture"]
    )

    rainfall = float(
        row["Rainfall_Last_24h_mm"]
    )

    temperature = float(
        row["Air_Temperature_C"]
    )

    humidity = float(
        row["Relative_Humidity_pct"]
    )

    wind_speed = float(
        row["Wind_Speed_kmh"]
    )

    field_area = float(
        row["Field_Area_hectare"]
    )

    previous_irrigation = float(
        row["Previous_Irrigation_mm_24h"]
    )

    crop = row["Crop_Type"]

    growth_stage = row[
        "Crop_Growth_Stage"
    ]


    # ========================================================
    # 5. BASE IRRIGATION DEPTH
    # ========================================================

    water_depth = {
        "Low": 5.0,
        "Medium": 10.0,
        "High": 15.0
    }

    irrigation_depth = water_depth.get(
        prediction,
        5.0
    )


    # ========================================================
    # 6. SOIL MOISTURE ADJUSTMENT
    # ========================================================

    if soil_moisture < 20:

        irrigation_depth += 5

    elif soil_moisture < 30:

        irrigation_depth += 3

    elif soil_moisture > 60:

        irrigation_depth -= 3


    # ========================================================
    # 7. RAINFALL ADJUSTMENT
    # ========================================================

    if rainfall > 1000:

        irrigation_depth -= 5

    elif rainfall > 500:

        irrigation_depth -= 3


    # ========================================================
    # 8. TEMPERATURE ADJUSTMENT
    # ========================================================

    if temperature > 35:

        irrigation_depth += 3

    elif temperature > 30:

        irrigation_depth += 2


    # ========================================================
    # 9. HUMIDITY ADJUSTMENT
    # ========================================================

    if humidity < 40:

        irrigation_depth += 2

    elif humidity > 80:

        irrigation_depth -= 2


    # ========================================================
    # 10. WIND ADJUSTMENT
    # ========================================================

    if wind_speed > 15:

        irrigation_depth += 1


    # ========================================================
    # 11. CROP GROWTH STAGE
    # ========================================================

    if growth_stage == "Flowering":

        irrigation_depth += 2

    elif growth_stage == "Vegetative":

        irrigation_depth += 1

    elif growth_stage == "Harvest":

        irrigation_depth -= 2


    # ========================================================
    # 12. PREVIOUS IRRIGATION
    # ========================================================

    if previous_irrigation > 100:

        irrigation_depth -= 2


    # ========================================================
    # 13. SAFETY LIMIT
    # ========================================================

    irrigation_depth = max(
        0,
        irrigation_depth
    )

    irrigation_depth = min(
        irrigation_depth,
        20
    )


    # ========================================================
    # 14. OVERWATERING PROTECTION
    # ========================================================

    if rainfall > 1500:

        recommendation = (
            "Delay irrigation due to high "
            "recent rainfall."
        )

        irrigation_depth = 0

    elif soil_moisture > 70:

        recommendation = (
            "Irrigation can be skipped because "
            "soil moisture is high."
        )

        irrigation_depth = 0

    else:

        recommendation = (
            "Irrigation recommended."
        )


    # ========================================================
    # 15. WATER QUANTITY
    # ========================================================

    # 1 hectare = 10,000 square meters
    #
    # 1 mm water over 1 hectare
    # = 10,000 litres

    water_quantity_litres = (
        irrigation_depth
        * field_area
        * 10000
    )


    # ========================================================
    # 16. IRRIGATION FLOW RATE
    # ========================================================

    # System assumption:
    # 10,000 litres/hour/hectare

    flow_rate_per_hectare = 10000

    total_flow_rate = (
        flow_rate_per_hectare
        * field_area
    )


    # ========================================================
    # 17. DURATION
    # ========================================================

    if (
        total_flow_rate > 0
        and water_quantity_litres > 0
    ):

        duration_hours = (
            water_quantity_litres
            / total_flow_rate
        )

        duration_minutes = (
            duration_hours * 60
        )

    else:

        duration_minutes = 0


    # ========================================================
    # 18. RECOMMENDED TIME
    # ========================================================

    if temperature >= 35:

        recommended_time = (
            "05:30 AM - 07:30 AM"
        )

    elif temperature >= 30:

        recommended_time = (
            "06:00 AM - 08:00 AM"
        )

    else:

        recommended_time = (
            "06:00 AM - 09:00 AM"
        )


    # ========================================================
    # 19. FINAL RESULT
    # ========================================================

    return {

        "Crop": crop,

        "Growth Stage": growth_stage,

        "Predicted Irrigation Need": prediction,

        "Soil Moisture (%)": round(
            soil_moisture,
            2
        ),

        "Rainfall Last 24h (mm)": round(
            rainfall,
            2
        ),

        "Temperature (°C)": round(
            temperature,
            2
        ),

        "Irrigation Depth (mm)": round(
            irrigation_depth,
            2
        ),

        "Water Quantity (litres)": round(
            water_quantity_litres,
            2
        ),

        "Duration (minutes)": round(
            duration_minutes,
            2
        ),

        "Recommended Time": recommended_time,

        "Recommendation": recommendation
    }


# ============================================================
# 20. MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "AI-POWERED SMART IRRIGATION SCHEDULER"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print("\nLoading field data...")

    df = pd.read_csv(
        TEST_DATA_PATH
    )

    print(
        f"Loaded {len(df)} field records."
    )


    # --------------------------------------------------------
    # Select one field
    # --------------------------------------------------------

    row = df.iloc[0].copy()


    # --------------------------------------------------------
    # ML Prediction
    # --------------------------------------------------------

    print("\nRunning Random Forest prediction...")

    prediction = predict_irrigation_need(
        row
    )

    print(
        f"ML Prediction: {prediction}"
    )


    # --------------------------------------------------------
    # Generate schedule
    # --------------------------------------------------------

    schedule = calculate_irrigation_schedule(
        row,
        prediction
    )


    # --------------------------------------------------------
    # Display schedule
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        "IRRIGATION SCHEDULE"
    )

    print("-" * 70)


    for key, value in schedule.items():

        print(
            f"{key:<35}: {value}"
        )


    print("\n" + "=" * 70)

    print(
        "AI IRRIGATION SCHEDULER COMPLETED"
    )

    print("=" * 70)


# ============================================================
# 21. RUN
# ============================================================

if __name__ == "__main__":
    main()
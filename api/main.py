from pathlib import Path
import sys
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


# ============================================================
# 2. IMPORT ML + SCHEDULER
# ============================================================

from scheduler.irrigation_scheduler import (
    predict_irrigation_need,
    calculate_irrigation_schedule
)


# ============================================================
# 3. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI-Powered Smart Irrigation API",
    description="AI-based irrigation prediction and weather-aware scheduling API",
    version="1.0.0"
)


# ============================================================
# 4. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 5. INPUT MODEL
# ============================================================

class IrrigationInput(BaseModel):

    # Soil
    Soil_Moisture: float
    Soil_pH: float
    Organic_Carbon: float
    Electrical_Conductivity: float
    Soil_Type: str

    # Weather/current conditions
    Air_Temperature_C: float
    Relative_Humidity_pct: float
    Rainfall_Last_24h_mm: float
    Sunlight_Hours: float
    Wind_Speed_kmh: float

    # Crop
    Crop_Type: str
    Crop_Growth_Stage: str

    # Field
    Field_Area_hectare: float

    # Historical irrigation
    Previous_Irrigation_mm_24h: float

    # Irrigation
    Water_Source: str
    Irrigation_Type: str

    # Other
    Season: str
    Region: str
    Mulching_Used: str


# ============================================================
# 6. ROOT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "AI-Powered Smart Irrigation API",
        "status": "running"
    }


# ============================================================
# 7. HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "Random Forest",
        "service": "Weather-Aware Irrigation Scheduler"
    }


# ============================================================
# 8. FEATURE ENGINEERING
# ============================================================

def create_engineered_features(data):

    soil_moisture = data["Soil_Moisture"]
    field_area = data["Field_Area_hectare"]
    previous_irrigation = data["Previous_Irrigation_mm_24h"]

    temperature = data["Air_Temperature_C"]
    humidity = data["Relative_Humidity_pct"]
    wind_speed = data["Wind_Speed_kmh"]
    rainfall = data["Rainfall_Last_24h_mm"]

    # Moisture deficit
    moisture_deficit = max(
        100 - soil_moisture,
        0
    )

    # Previous water per hectare
    if field_area > 0:
        previous_water_per_hectare = (
            previous_irrigation / field_area
        )
    else:
        previous_water_per_hectare = 0

    # Weather stress
    weather_stress_index = (
        (temperature / 30)
        + ((100 - humidity) / 100)
        + (wind_speed / 20)
    )

    # Rainfall adjusted moisture
    rainfall_adjusted_moisture = (
        soil_moisture + rainfall
    )

    data["Moisture_Deficit"] = moisture_deficit

    data["Previous_Water_per_Hectare"] = (
        previous_water_per_hectare
    )

    data["Weather_Stress_Index"] = (
        weather_stress_index
    )

    data["Rainfall_Adjusted_Moisture"] = (
        rainfall_adjusted_moisture
    )

    return data


# ============================================================
# 9. WEATHER FORECAST
# ============================================================

def get_weather_forecast(
    latitude=17.3850,
    longitude=78.4867
):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {

        "latitude": latitude,

        "longitude": longitude,

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m"
        ),

        "forecast_days": 3,

        "timezone": "auto"
    }

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# 10. EXTRACT NEXT 24-HOUR WEATHER
# ============================================================

def analyze_next_24_hours(weather_data):

    hourly = weather_data["hourly"]

    temperatures = hourly["temperature_2m"][:24]

    humidity = hourly["relative_humidity_2m"][:24]

    precipitation = hourly["precipitation"][:24]

    wind = hourly["wind_speed_10m"][:24]


    total_rainfall = sum(
        precipitation
    )

    average_temperature = (
        sum(temperatures) / len(temperatures)
    )

    average_humidity = (
        sum(humidity) / len(humidity)
    )

    average_wind = (
        sum(wind) / len(wind)
    )

    maximum_temperature = max(
        temperatures
    )


    return {

        "Rainfall_Next_24h_mm":
            round(total_rainfall, 2),

        "Average_Temperature_Next_24h_C":
            round(average_temperature, 2),

        "Maximum_Temperature_Next_24h_C":
            round(maximum_temperature, 2),

        "Average_Humidity_Next_24h_pct":
            round(average_humidity, 2),

        "Average_Wind_Next_24h_kmh":
            round(average_wind, 2)
    }


# ============================================================
# 11. WEATHER-AWARE SCHEDULE ADJUSTMENT
# ============================================================

def apply_weather_adjustment(
    schedule,
    weather_summary
):

    rainfall = weather_summary[
        "Rainfall_Next_24h_mm"
    ]

    temperature = weather_summary[
        "Maximum_Temperature_Next_24h_C"
    ]

    original_depth = schedule[
        "Irrigation Depth (mm)"
    ]


    adjusted_depth = original_depth

    weather_message = (
        "No significant forecast-based adjustment."
    )


    # --------------------------------------------------------
    # Forecast rainfall adjustment
    # --------------------------------------------------------

    if rainfall >= 10:

        adjusted_depth = 0

        weather_message = (
            "Irrigation delayed because forecast "
            "rainfall is expected in the next 24 hours."
        )


    elif rainfall >= 5:

        adjusted_depth = max(
            original_depth - 3,
            0
        )

        weather_message = (
            "Irrigation reduced because rainfall "
            "is expected in the next 24 hours."
        )


    # --------------------------------------------------------
    # High temperature adjustment
    # --------------------------------------------------------

    elif temperature >= 35:

        adjusted_depth = min(
            original_depth + 2,
            20
        )

        weather_message = (
            "Irrigation slightly increased because "
            "high temperature is forecast."
        )


    # --------------------------------------------------------
    # Update schedule
    # --------------------------------------------------------

    schedule[
        "Original Irrigation Depth (mm)"
    ] = original_depth

    schedule[
        "Irrigation Depth (mm)"
    ] = adjusted_depth


    # --------------------------------------------------------
    # Recalculate water quantity
    # --------------------------------------------------------

    area = schedule.get(
        "_field_area",
        0
    )

    if area > 0:

        water_quantity = (
            adjusted_depth
            * area
            * 10000
        )

        schedule[
            "Water Quantity (litres)"
        ] = round(
            water_quantity,
            2
        )


    schedule[
        "Weather Adjustment"
    ] = weather_message


    schedule[
        "Forecast Rainfall Next 24h (mm)"
    ] = rainfall


    schedule[
        "Forecast Maximum Temperature (°C)"
    ] = temperature


    # --------------------------------------------------------
    # Remove internal field
    # --------------------------------------------------------

    schedule.pop(
        "_field_area",
        None
    )


    return schedule


# ============================================================
# 12. WEATHER FORECAST ENDPOINT
# ============================================================

@app.get("/weather-forecast")
def weather_forecast(
    latitude: float,
    longitude: float
):

    try:

        weather_data = get_weather_forecast(
            latitude,
            longitude
        )

        return {

            "success": True,

            "location": {
                "latitude": latitude,
                "longitude": longitude
            },

            "forecast": weather_data
        }


    except requests.RequestException as e:

        raise HTTPException(
            status_code=502,
            detail=f"Weather API error: {str(e)}"
        )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# 13. PREDICT IRRIGATION
# ============================================================

@app.post("/predict-irrigation")
def predict_irrigation(
    data: IrrigationInput
):

    try:

        # ----------------------------------------------------
        # Convert input to dictionary
        # ----------------------------------------------------

        field_data = data.model_dump()


        # ----------------------------------------------------
        # Backend Feature Engineering
        # ----------------------------------------------------

        field_data = create_engineered_features(
            field_data
        )


        # ----------------------------------------------------
        # ML Prediction
        # ----------------------------------------------------

        prediction = predict_irrigation_need(
            field_data
        )


        # ----------------------------------------------------
        # Base irrigation schedule
        # ----------------------------------------------------

        schedule = calculate_irrigation_schedule(
            field_data,
            prediction
        )


        # Store area temporarily for weather adjustment
        schedule["_field_area"] = (
            field_data["Field_Area_hectare"]
        )


        # ----------------------------------------------------
        # Get weather forecast
        # ----------------------------------------------------

        weather_data = get_weather_forecast()


        # ----------------------------------------------------
        # Analyze next 24 hours
        # ----------------------------------------------------

        weather_summary = analyze_next_24_hours(
            weather_data
        )


        # ----------------------------------------------------
        # Apply weather adjustment
        # ----------------------------------------------------

        schedule = apply_weather_adjustment(
            schedule,
            weather_summary
        )


        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        return {

            "success": True,

            "prediction": prediction,

            "engineered_features": {

                "Moisture_Deficit":
                    round(
                        field_data[
                            "Moisture_Deficit"
                        ],
                        3
                    ),

                "Previous_Water_per_Hectare":
                    round(
                        field_data[
                            "Previous_Water_per_Hectare"
                        ],
                        3
                    ),

                "Weather_Stress_Index":
                    round(
                        field_data[
                            "Weather_Stress_Index"
                        ],
                        3
                    ),

                "Rainfall_Adjusted_Moisture":
                    round(
                        field_data[
                            "Rainfall_Adjusted_Moisture"
                        ],
                        3
                    )
            },

            "weather_forecast": weather_summary,

            "schedule": schedule
        }


    except requests.RequestException as e:

        raise HTTPException(
            status_code=502,
            detail=f"Weather API error: {str(e)}"
        )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# 14. RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
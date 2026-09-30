"use client";

import { useState } from "react";

export default function Home() {
  const [formData, setFormData] = useState({
    Soil_Moisture: 34.67,
    Soil_pH: 6.5,
    Organic_Carbon: 1.5,
    Electrical_Conductivity: 1.2,
    Soil_Type: "Loamy",

    Air_Temperature_C: 34.98,
    Relative_Humidity_pct: 60,
    Rainfall_Last_24h_mm: 793.12,
    Sunlight_Hours: 8,
    Wind_Speed_kmh: 10,

    Crop_Type: "Maize",
    Crop_Growth_Stage: "Sowing",

    Field_Area_hectare: 12.3,
    Previous_Irrigation_mm_24h: 60,

    Water_Source: "Canal",
    Irrigation_Type: "Drip",
    Season: "Kharif",
    Region: "Telangana",

    Mulching_Used: "Yes",

    Moisture_Deficit: 65.33,
    Previous_Water_per_Hectare: 4.878,
    Weather_Stress_Index: 1.766,
    Rainfall_Adjusted_Moisture: 827.79,
  });

  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (
    e: React.ChangeEvent<
      HTMLInputElement | HTMLSelectElement
    >
  ) => {
    const { name, value } = e.target;

    setFormData((previous) => ({
      ...previous,
      [name]: value,
    }));
  };

  const handleSubmit = async (
    e: React.FormEvent<HTMLFormElement>
  ) => {
    e.preventDefault();

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict-irrigation",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            ...formData,

            // Convert numeric input values
            // from strings to numbers
            Soil_Moisture: Number(formData.Soil_Moisture),
            Soil_pH: Number(formData.Soil_pH),
            Organic_Carbon: Number(formData.Organic_Carbon),
            Electrical_Conductivity: Number(
              formData.Electrical_Conductivity
            ),
            Air_Temperature_C: Number(
              formData.Air_Temperature_C
            ),
            Relative_Humidity_pct: Number(
              formData.Relative_Humidity_pct
            ),
            Rainfall_Last_24h_mm: Number(
              formData.Rainfall_Last_24h_mm
            ),
            Sunlight_Hours: Number(
              formData.Sunlight_Hours
            ),
            Wind_Speed_kmh: Number(
              formData.Wind_Speed_kmh
            ),
            Field_Area_hectare: Number(
              formData.Field_Area_hectare
            ),
            Previous_Irrigation_mm_24h: Number(
              formData.Previous_Irrigation_mm_24h
            ),
            Moisture_Deficit: Number(
              formData.Moisture_Deficit
            ),
            Previous_Water_per_Hectare: Number(
              formData.Previous_Water_per_Hectare
            ),
            Weather_Stress_Index: Number(
              formData.Weather_Stress_Index
            ),
            Rainfall_Adjusted_Moisture: Number(
              formData.Rainfall_Adjusted_Moisture
            ),
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Prediction failed"
        );
      }

      setResult(data);
    } catch (error) {
      setResult({
        error:
          error instanceof Error
            ? error.message
            : "Something went wrong",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-green-50 px-6 py-10">
      <div className="mx-auto max-w-6xl">

        {/* HEADER */}
        <div className="mb-10 text-center">
          <div className="mb-3 text-5xl">
            🌱
          </div>

          <h1 className="text-4xl font-bold text-green-800">
            Smart Irrigation System
          </h1>

          <p className="mt-3 text-gray-600">
            AI-powered irrigation prediction and
            water management
          </p>
        </div>

        {/* DASHBOARD */}
        <div className="grid gap-8 lg:grid-cols-2">

          {/* INPUT SECTION */}
          <div className="rounded-2xl bg-white p-7 shadow-lg">

            <h2 className="mb-6 text-2xl font-bold text-gray-800">
              Field Information
            </h2>

            <form
              onSubmit={handleSubmit}
              className="space-y-5"
            >

              <Input
                label="Soil Moisture (%)"
                name="Soil_Moisture"
                value={formData.Soil_Moisture}
                onChange={handleChange}
              />

              <Input
                label="Temperature (°C)"
                name="Air_Temperature_C"
                value={formData.Air_Temperature_C}
                onChange={handleChange}
              />

              <Input
                label="Humidity (%)"
                name="Relative_Humidity_pct"
                value={
                  formData.Relative_Humidity_pct
                }
                onChange={handleChange}
              />

              <Input
                label="Rainfall Last 24h (mm)"
                name="Rainfall_Last_24h_mm"
                value={
                  formData.Rainfall_Last_24h_mm
                }
                onChange={handleChange}
              />

              <Input
                label="Field Area (hectare)"
                name="Field_Area_hectare"
                value={
                  formData.Field_Area_hectare
                }
                onChange={handleChange}
              />

              {/* CROP */}
              <Select
                label="Crop Type"
                name="Crop_Type"
                value={formData.Crop_Type}
                onChange={handleChange}
                options={[
                  "Maize",
                  "Rice",
                  "Wheat",
                  "Cotton",
                  "Potato",
                  "Sugarcane",
                ]}
              />

              {/* GROWTH STAGE */}
              <Select
                label="Growth Stage"
                name="Crop_Growth_Stage"
                value={
                  formData.Crop_Growth_Stage
                }
                onChange={handleChange}
                options={[
                  "Sowing",
                  "Vegetative",
                  "Flowering",
                  "Harvest",
                ]}
              />

              {/* BUTTON */}
              <button
                type="submit"
                disabled={loading}
                className="mt-3 w-full rounded-xl bg-green-700 px-5 py-4 font-semibold text-white transition hover:bg-green-800 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading
                  ? "Analyzing Field..."
                  : "🌿 Predict Irrigation"}
              </button>

            </form>
          </div>

          {/* RESULT SECTION */}
          <div className="rounded-2xl bg-white p-7 shadow-lg">

            <h2 className="mb-6 text-2xl font-bold text-gray-800">
              AI Recommendation
            </h2>

            {/* INITIAL STATE */}
            {!result && (
              <div className="flex min-h-[400px] items-center justify-center rounded-xl bg-gray-50 text-center text-gray-500">
                <div>
                  <div className="mb-4 text-5xl">
                    💧
                  </div>

                  <p className="text-lg">
                    Enter your field information
                  </p>

                  <p className="mt-1 text-sm">
                    and click Predict Irrigation
                  </p>
                </div>
              </div>
            )}

            {/* ERROR */}
            {result?.error && (
              <div className="rounded-xl bg-red-50 p-5 text-red-700">
                <p className="font-semibold">
                  Prediction Error
                </p>

                <p className="mt-2 text-sm">
                  {result.error}
                </p>
              </div>
            )}

            {/* RESULT */}
            {result && !result.error && (
              <div className="space-y-4">

                {/* PREDICTION */}
                <div className="rounded-2xl bg-green-100 p-6">
                  <p className="text-sm font-medium text-green-700">
                    Irrigation Requirement
                  </p>

                  <p className="mt-1 text-4xl font-bold text-green-800">
                    {result.prediction}
                  </p>
                </div>

                <Result
                  label="Crop"
                  value={result.schedule.Crop}
                />

                <Result
                  label="Growth Stage"
                  value={
                    result.schedule[
                      "Growth Stage"
                    ]
                  }
                />

                <Result
                  label="Soil Moisture"
                  value={`${result.schedule["Soil Moisture (%)"]}%`}
                />

                <Result
                  label="Irrigation Depth"
                  value={`${result.schedule["Irrigation Depth (mm)"]} mm`}
                />

                <Result
                  label="Water Quantity"
                  value={`${result.schedule["Water Quantity (litres)"]} litres`}
                />

                <Result
                  label="Duration"
                  value={`${result.schedule["Duration (minutes)"]} minutes`}
                />

                <Result
                  label="Recommended Time"
                  value={
                    result.schedule[
                      "Recommended Time"
                    ]
                  }
                />

                {/* RECOMMENDATION */}
                <div className="rounded-xl bg-blue-50 p-5">
                  <p className="font-semibold text-blue-800">
                    Recommendation
                  </p>

                  <p className="mt-2 text-gray-700">
                    {result.schedule.Recommendation}
                  </p>
                </div>

              </div>
            )}

          </div>
        </div>

        {/* FOOTER */}
        <div className="mt-10 text-center text-sm text-gray-500">
          AI-Powered Smart Irrigation System
        </div>

      </div>
    </main>
  );
}


/* ============================================================
   INPUT COMPONENT
============================================================ */

function Input({
  label,
  name,
  value,
  onChange,
}: {
  label: string;
  name: string;
  value: number;
  onChange: (
    e: React.ChangeEvent<HTMLInputElement>
  ) => void;
}) {
  return (
    <div>
      <label className="mb-2 block text-sm font-medium text-gray-700">
        {label}
      </label>

      <input
        type="number"
        step="any"
        name={name}
        value={value}
        onChange={onChange}
        className="w-full rounded-xl border border-gray-300 p-3 text-gray-800 outline-none transition focus:border-green-600 focus:ring-2 focus:ring-green-100"
      />
    </div>
  );
}


/* ============================================================
   SELECT COMPONENT
============================================================ */

function Select({
  label,
  name,
  value,
  onChange,
  options,
}: {
  label: string;
  name: string;
  value: string;
  onChange: (
    e: React.ChangeEvent<HTMLSelectElement>
  ) => void;
  options: string[];
}) {
  return (
    <div>
      <label className="mb-2 block text-sm font-medium text-gray-700">
        {label}
      </label>

      <select
        name={name}
        value={value}
        onChange={onChange}
        className="w-full rounded-xl border border-gray-300 bg-white p-3 text-gray-800 outline-none focus:border-green-600 focus:ring-2 focus:ring-green-100"
      >
        {options.map((option) => (
          <option
            key={option}
            value={option}
          >
            {option}
          </option>
        ))}
      </select>
    </div>
  );
}


/* ============================================================
   RESULT COMPONENT
============================================================ */

function Result({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center justify-between rounded-xl border border-gray-200 p-4">
      <span className="text-gray-600">
        {label}
      </span>

      <span className="text-right font-semibold text-gray-800">
        {value}
      </span>
    </div>
  );
}
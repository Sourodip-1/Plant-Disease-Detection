import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import functions from existing inference script
from inference import (
    load_vision_model,
    load_disease_model,
    get_coords,
    fetch_weather,
    _weather_to_categorical,
    predict_symptoms,
    text_to_symptoms,
    predict_disease,
    TABULAR_SYMPTOMS
)

app = FastAPI(
    title="Plant Disease Detection API",
    description="API for predicting plant diseases from images and location data.",
    version="1.0.0"
)

# Allow CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup to keep inference fast
print("Loading models...")
vision_model = load_vision_model()
dt_model, encoders = load_disease_model()
print("Models loaded successfully.")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    plant_type: str = Form(...),
    location: str = Form(...),
    features: str = Form(None)
):
    try:
        # 1. Save uploaded file to a temporary location
        suffix = os.path.splitext(file.filename)[1]
        if not suffix:
            suffix = ".jpg"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # 2. Geolocation & Weather
        lat, lon = get_coords(location)
        weather, hourly_df = fetch_weather(lat, lon)
        temp_cat, hum_cat, soil_cat = _weather_to_categorical(
            weather["temperature_2m"],
            weather["relative_humidity_2m"],
            weather["soil_moisture_0_to_1cm"],
        )

        # 3. Vision Inference OR Feature Extraction
        if features and features.strip() != "":
            symptom_probs, symptom_binary, vision_binary_dict = text_to_symptoms(features)
        else:
            symptom_probs, symptom_binary, vision_binary_dict = predict_symptoms(temp_path, vision_model)

        # 4. Disease Prediction
        disease = predict_disease(
            plant_type.lower(), symptom_binary,
            temp_cat, hum_cat, soil_cat,
            dt_model, encoders
        )

        # 5. Clean up temp file
        os.remove(temp_path)

        # 6. Format Response
        # Convert numpy types to native Python types for JSON serialization
        symptom_probs_clean = {k: float(v) for k, v in symptom_probs.items()}
        detected_physical = [s for s, b in zip(TABULAR_SYMPTOMS, symptom_binary) if b]

        return {
            "status": "success",
            "data": {
                "diagnosis": disease.replace("_", " ").title(),
                "plant_type": plant_type,
                "location": {
                    "name": location,
                    "lat": float(lat),
                    "lon": float(lon)
                },
                "weather": {
                    "temperature_c": float(weather["temperature_2m"]),
                    "humidity_pct": float(weather["relative_humidity_2m"]),
                    "rain_mm": float(weather["rain"]),
                    "soil_moisture": float(weather["soil_moisture_0_to_1cm"]),
                    "temperature_category": temp_cat,
                    "humidity_category": hum_cat,
                    "soil_category": soil_cat
                },
                "symptoms": {
                    "vision_probabilities": symptom_probs_clean,
                    "detected_physical_symptoms": detected_physical
                }
            }
        }

    except Exception as e:
        # Clean up temp file in case of error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


# Serve static files from the Site folder
app.mount("/", StaticFiles(directory="site", html=True), name="site")


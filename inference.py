"""
inference.py
============
Plant Disease Detection — Full Inference Pipeline
--------------------------------------------------
Uses:
  • models/final_plant_model_v2.pth  → ResNet18 vision model  (7 binary symptoms)
  • disease_prediction_model.pkl     → Decision-Tree tabular model (disease name)
  • encoders.pkl                     → LabelEncoders for categorical features
  • Open-Meteo API                   → Real-time weather at the user's location
  • geopy / Nominatim                → Location name → (lat, lon)

Run:
  python inference.py
"""

# ── Standard & third-party imports ─────────────────────────────────────────────
import os
import sys
import textwrap

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# pyrefly: ignore [missing-import]
import openmeteo_requests
import pandas as pd
import requests_cache
import torch
import torch.nn as nn
from geopy.geocoders import Nominatim
from PIL import Image
from retry_requests import retry
from torchvision import models as tv_models, transforms

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
VISION_CLASSES = ['Leaf Spot', 'Blight', 'Rust', 'Mold', 'Scab', 'Healthy', 'Viral']
TABULAR_SYMPTOMS = ["spots", "yellowing", "wilting", "powder", "leaf_curl", "stem_damage", "mold_growth"]

VISION_CKPT   = os.path.join(os.path.dirname(__file__), "models", "final_plant_model_v2.pth")
DT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "disease_prediction_model.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "encoders.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VISION_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ═══════════════════════════════════════════════════════════════════════════════
# 2. VISION MODEL  (ResNet18 → 7 binary symptoms)
# ═══════════════════════════════════════════════════════════════════════════════
def _build_resnet(num_classes: int = len(VISION_CLASSES)) -> nn.Module:
    m = tv_models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_vision_model(ckpt: str = VISION_CKPT) -> nn.Module:
    if not os.path.exists(ckpt):
        sys.exit(f"[ERROR] Vision checkpoint not found: {ckpt}")
    m = _build_resnet()
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    m = m.to(DEVICE).eval()
    print(f"[OK] Vision model loaded from '{ckpt}'  (device: {DEVICE})")
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DISEASE PREDICTION MODEL  (Decision Tree from model.ipynb / model.py)
# ═══════════════════════════════════════════════════════════════════════════════
def load_disease_model():
    for path in (DT_MODEL_PATH, ENCODERS_PATH):
        if not os.path.exists(path):
            sys.exit(f"[ERROR] Required file not found: {path}\n"
                     f"        Run model.py / model.ipynb first to generate it.")
    dt_model = joblib.load(DT_MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    print(f"[OK] Disease prediction model loaded from '{DT_MODEL_PATH}'")
    return dt_model, encoders


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WEATHER  (Open-Meteo API)
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_weather(lat: float, lon: float) -> dict:
    """
    Returns a dict with current-hour values for:
      temperature_2m, relative_humidity_2m, rain, soil_moisture_0_to_1cm,
      precipitation, cloud_cover, soil_temperature_0cm
    """
    try:
        cache_session  = requests_cache.CachedSession(".cache", expire_after=3600)
        retry_session  = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo_c    = openmeteo_requests.Client(session=retry_session)

        url    = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude" : lat,
            "longitude": lon,
            "hourly"   : [
                "temperature_2m",
                "relative_humidity_2m",
                "rain",
                "soil_moisture_0_to_1cm",
                "precipitation",
                "cloud_cover",
                "soil_temperature_0cm",
            ],
        }

        responses = openmeteo_c.weather_api(url, params=params)
        response  = responses[0]

        hourly = response.Hourly()
        variables = [hourly.Variables(i).ValuesAsNumpy() for i in range(7)]

        hourly_df = pd.DataFrame({
            "date"                   : pd.date_range(
                start     = pd.to_datetime(hourly.Time(),    unit="s", utc=True),
                end       = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq      = pd.Timedelta(seconds=hourly.Interval()),
                inclusive = "left",
            ),
            "temperature_2m"          : variables[0],
            "relative_humidity_2m"    : variables[1],
            "rain"                    : variables[2],
            "soil_moisture_0_to_1cm"  : variables[3],
            "precipitation"           : variables[4],
            "cloud_cover"             : variables[5],
            "soil_temperature_0cm"    : variables[6],
        })

        print(f"\n[Weather] Coordinates: {response.Latitude():.4f}°N  "
              f"{response.Longitude():.4f}°E   "
              f"Elevation: {response.Elevation():.0f} m\n")

        # Use the most recent hour's data
        current = hourly_df.iloc[-1].to_dict()
        return current, hourly_df
    except Exception as e:
        print(f"[ERROR] Weather API failed: {e}. Using fallback weather data.")
        fallback = {
            "temperature_2m": 25.0,
            "relative_humidity_2m": 60.0,
            "rain": 0.0,
            "soil_moisture_0_to_1cm": 0.3,
            "precipitation": 0.0,
            "cloud_cover": 50.0,
            "soil_temperature_0cm": 25.0,
        }
        return fallback, pd.DataFrame([fallback])


def _weather_to_categorical(temp_c: float, humidity_pct: float,
                             soil_moisture: float) -> tuple[str, str, str]:
    """Convert numeric weather values to the categorical labels used in training."""
    # Temperature: low < 15°C, medium 15–28°C, high > 28°C
    if temp_c < 15:
        temp_cat = "low"
    elif temp_c <= 28:
        temp_cat = "medium"
    else:
        temp_cat = "high"

    # Humidity: low < 40%, medium 40–70%, high > 70%
    if humidity_pct < 40:
        hum_cat = "low"
    elif humidity_pct <= 70:
        hum_cat = "medium"
    else:
        hum_cat = "high"

    # Soil moisture: dry < 0.2, moderate 0.2–0.5, wet > 0.5
    if soil_moisture < 0.2:
        soil_cat = "dry"
    elif soil_moisture <= 0.5:
        soil_cat = "moderate"
    else:
        soil_cat = "wet"

    return temp_cat, hum_cat, soil_cat


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GEOLOCATION
# ═══════════════════════════════════════════════════════════════════════════════
def get_coords(location_name: str) -> tuple[float, float]:
    try:
        geolocator = Nominatim(user_agent="plant_disease_inference")
        loc = geolocator.geocode(location_name, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except Exception as e:
        print(f"[ERROR] Geolocation failed for '{location_name}': {e}")
        
    print(f"[WARN] Location '{location_name}' not found or API error. Using default (0°, 0°).")
    return 0.0, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SYMPTOM PREDICTION & MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
def predict_symptoms(image_path: str, model: nn.Module,
                     threshold: float = 0.4) -> tuple[dict, np.ndarray, dict]:
    img = Image.open(image_path).convert("RGB")
    tensor = VISION_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Binary flags for the vision classes
    binary_vision = (probs > threshold).astype(int)
    
    # Fallback: if nothing crossed the threshold, pick the highest probability (if not Healthy)
    if not binary_vision.any():
        max_idx = np.argmax(probs)
        if VISION_CLASSES[max_idx] != 'Healthy':
            binary_vision[max_idx] = 1

    vision_probs_dict = dict(zip(VISION_CLASSES, probs))
    vision_binary_dict = dict(zip(VISION_CLASSES, binary_vision))

    # Map Vision Classes -> Tabular Symptoms
    tab_binary = np.zeros(len(TABULAR_SYMPTOMS), dtype=int)
    
    # spots (Leaf Spot, Blight, Rust, Scab)
    if vision_binary_dict.get('Leaf Spot') or vision_binary_dict.get('Blight') or vision_binary_dict.get('Rust') or vision_binary_dict.get('Scab'):
        tab_binary[0] = 1
    # yellowing (Blight, Viral, Rust)
    if vision_binary_dict.get('Blight') or vision_binary_dict.get('Viral') or vision_binary_dict.get('Rust'):
        tab_binary[1] = 1
    # wilting (Blight)
    if vision_binary_dict.get('Blight'):
        tab_binary[2] = 1
    # powder (Mold)
    if vision_binary_dict.get('Mold'):
        tab_binary[3] = 1
    # leaf_curl (Viral)
    if vision_binary_dict.get('Viral'):
        tab_binary[4] = 1
    # stem_damage (no direct mapping, keep 0 unless we add logic)
    # mold_growth (Mold)
    if vision_binary_dict.get('Mold'):
        tab_binary[6] = 1

    return vision_probs_dict, tab_binary, vision_binary_dict


def text_to_symptoms(features: str) -> tuple[dict, np.ndarray, dict]:
    """
    Parses a text query to extract tabular symptoms, bypassing the CNN model.
    TABULAR_SYMPTOMS = ["spots", "yellowing", "wilting", "powder", "leaf_curl", "stem_damage", "mold_growth"]
    """
    features_lower = features.lower()
    tab_binary = np.zeros(len(TABULAR_SYMPTOMS), dtype=int)
    
    if any(k in features_lower for k in ["spot", "dot", "lesion"]):
        tab_binary[0] = 1
    if any(k in features_lower for k in ["yellow", "pale", "chlorosis", "brown"]):
        tab_binary[1] = 1
    if any(k in features_lower for k in ["wilt", "droop", "limp"]):
        tab_binary[2] = 1
    if any(k in features_lower for k in ["powder", "white", "dust", "ash"]):
        tab_binary[3] = 1
    if any(k in features_lower for k in ["curl", "roll", "twist"]):
        tab_binary[4] = 1
    if any(k in features_lower for k in ["stem", "stalk", "rot"]):
        tab_binary[5] = 1
    if any(k in features_lower for k in ["mold", "mould", "fungus", "fuzz"]):
        tab_binary[6] = 1

    # Dummy vision model outputs
    vision_probs_dict = {cls: 0.0 for cls in VISION_CLASSES}
    vision_binary_dict = {cls: 0 for cls in VISION_CLASSES}
    
    # If no symptoms were found and the user mentions healthy
    if "healthy" in features_lower or "normal" in features_lower and np.sum(tab_binary) == 0:
        vision_probs_dict["Healthy"] = 1.0
        vision_binary_dict["Healthy"] = 1
        
    return vision_probs_dict, tab_binary, vision_binary_dict


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DISEASE PREDICTION (tabular Decision Tree)
# ═══════════════════════════════════════════════════════════════════════════════
def predict_disease(plant_type: str, symptom_binary: np.ndarray,
                    temp_cat: str, hum_cat: str, soil_cat: str,
                    dt_model, encoders: dict) -> str:
    """
    Feature vector order (matches model.py training):
      plant_type, spots, yellowing, wilting, powder, leaf_curl,
      stem_damage, mold_growth, humidity, temperature, soil_moisture
    """
    try:
        p = encoders["plant_encoder"].transform([plant_type])[0]
    except ValueError:
        known = encoders["plant_encoder"].classes_.tolist()
        print(f"[WARN] Unknown plant '{plant_type}'. Known: {known}. Using first.")
        p = 0

    h = encoders["humidity_encoder"].transform([hum_cat])[0]
    t = encoders["temperature_encoder"].transform([temp_cat])[0]
    s = encoders["soil_encoder"].transform([soil_cat])[0]

    features = [[
        p,
        float(symptom_binary[0]), # spots
        float(symptom_binary[1]), # yellowing
        float(symptom_binary[2]), # wilting
        float(symptom_binary[3]), # powder
        float(symptom_binary[4]), # leaf_curl
        float(symptom_binary[5]), # stem_damage
        float(symptom_binary[6]), # mold_growth
        h,
        t,
        s
    ]]
    print(f"    [DEBUG] Passing features to Decision Tree: {features[0]}")
    pred_idx  = dt_model.predict(features)[0]
    disease   = encoders["disease_encoder"].inverse_transform([pred_idx])[0]
    
    # ── Sanity Check for Overfitted Decision Tree ──
    # The DT often predicts 'healthy' if it sees an unseen weather combination (e.g. sick plant in 'dry' soil).
    if disease == 'healthy' and np.sum(symptom_binary) > 0:
        print("[WARN] Tabular model predicted 'Healthy' despite visual symptoms due to extreme weather variables. Recalculating with moderate weather...")
        h_mod = encoders["humidity_encoder"].transform(['medium'])[0]
        t_mod = encoders["temperature_encoder"].transform(['medium'])[0]
        s_mod = encoders["soil_encoder"].transform(['moderate'])[0]
        
        features_mod = [[
            p,
            float(symptom_binary[0]), # spots
            float(symptom_binary[1]), # yellowing
            float(symptom_binary[2]), # wilting
            float(symptom_binary[3]), # powder
            float(symptom_binary[4]), # leaf_curl
            float(symptom_binary[5]), # stem_damage
            float(symptom_binary[6]), # mold_growth
            h_mod,
            t_mod,
            s_mod
        ]]
        print(f"    [DEBUG] Passing modified features to Decision Tree: {features_mod[0]}")
        pred_idx_mod = dt_model.predict(features_mod)[0]
        disease_mod = encoders["disease_encoder"].inverse_transform([pred_idx_mod])[0]
        
        if disease_mod != 'healthy':
            disease = disease_mod

    return disease


# ═══════════════════════════════════════════════════════════════════════════════
# 8. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════
COLORS = {
    "bar_detected" : "#e74c3c",
    "bar_clear"    : "#2ecc71",
    "weather_line" : "#3498db",
    "bg"           : "#1a1a2e",
    "panel"        : "#16213e",
    "text"         : "#eaeaea",
}

def _bar_color(prob: float) -> str:
    if prob > 0.6:
        return "#e74c3c"
    if prob > 0.35:
        return "#f39c12"
    return "#2ecc71"


def visualise_results(image_path: str, symptom_probs: dict,
                      symptom_binary: np.ndarray, disease: str,
                      weather: dict, plant_type: str,
                      location: str, vision_binary_dict: dict) -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor"  : COLORS["panel"],
        "text.color"      : COLORS["text"],
        "axes.labelcolor" : COLORS["text"],
        "xtick.color"     : COLORS["text"],
        "ytick.color"     : COLORS["text"],
        "axes.edgecolor"  : "#444",
    })

    fig = plt.figure(figsize=(18, 10), facecolor=COLORS["bg"])
    fig.suptitle("🌿  Plant Disease Detection Report",
                 fontsize=18, fontweight="bold",
                 color=COLORS["text"], y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # ── Panel 1: Input image ────────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[:, 0])
    img = Image.open(image_path).convert("RGB")
    ax_img.imshow(img)
    ax_img.set_title(f"📷  Input Image\n{os.path.basename(image_path)}",
                     fontsize=11, pad=8)
    ax_img.axis("off")

    # ── Panel 2: Symptom probability bar chart ───────────────────────────────
    ax_sym = fig.add_subplot(gs[0, 1])
    names  = list(symptom_probs.keys())
    probs  = list(symptom_probs.values())
    colors = [_bar_color(p) for p in probs]
    bars   = ax_sym.barh(names, probs, color=colors, edgecolor="#333", height=0.6)
    ax_sym.set_xlim(0, 1)
    ax_sym.axvline(0.4, color="#f1c40f", linewidth=1.2, linestyle="--",
                   label="threshold (0.4)")
    ax_sym.legend(fontsize=8, loc="lower right")
    for bar, p in zip(bars, probs):
        ax_sym.text(min(p + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                    f"{p:.2f}", va="center", fontsize=8.5)
    ax_sym.set_title("🔬  Symptom Probabilities", fontsize=11)
    ax_sym.set_xlabel("Confidence")

    # ── Panel 3: Detected vs clear binary lollipop ──────────────────────────
    ax_bin = fig.add_subplot(gs[1, 1])
    tab_names = TABULAR_SYMPTOMS
    for i, (name, b) in enumerate(zip(tab_names, symptom_binary)):
        col = "#e74c3c" if b else "#2ecc71"
        ax_bin.plot([0, b], [i, i], color=col, linewidth=2)
        ax_bin.scatter([b], [i], color=col, s=90, zorder=5)
    ax_bin.set_yticks(range(len(tab_names)))
    ax_bin.set_yticklabels(tab_names)
    ax_bin.set_xlim(-0.1, 1.3)
    ax_bin.set_xticks([0, 1])
    ax_bin.set_xticklabels(["Clear", "Detected"])
    ax_bin.set_title("🧪  Binary Symptom Flags", fontsize=11)

    # ── Panel 4: Weather summary ─────────────────────────────────────────────
    ax_wx = fig.add_subplot(gs[0, 2])
    ax_wx.axis("off")
    wx_lines = [
        f"📍 Location:      {location}",
        f"🌡  Temperature:   {weather.get('temperature_2m', 0):.1f} °C",
        f"💧 Humidity:      {weather.get('relative_humidity_2m', 0):.1f} %",
        f"🌧  Rain:          {weather.get('rain', 0):.2f} mm",
        f"🌧  Precipitation: {weather.get('precipitation', 0):.2f} mm",
        f"☁  Cloud cover:   {weather.get('cloud_cover', 0):.1f} %",
        f"🌱 Soil moisture: {weather.get('soil_moisture_0_to_1cm', 0):.3f} m³/m³",
        f"🌡  Soil temp:     {weather.get('soil_temperature_0cm', 0):.1f} °C",
    ]
    for j, line in enumerate(wx_lines):
        ax_wx.text(0.05, 0.92 - j * 0.12, line, fontsize=10,
                   transform=ax_wx.transAxes, va="top",
                   color=COLORS["text"])
    ax_wx.set_title("🌤  Live Weather Data", fontsize=11)

    # ── Panel 5: Diagnosis result ────────────────────────────────────────────
    ax_diag = fig.add_subplot(gs[1, 2])
    ax_diag.axis("off")
    is_healthy = "healthy" in disease.lower()
    diag_color = "#2ecc71" if is_healthy else "#e74c3c"
    emoji      = "✅" if is_healthy else "⚠️"
    ax_diag.text(0.5, 0.65, f"{emoji}  Final Diagnosis",
                 ha="center", va="center", fontsize=13, fontweight="bold",
                 color=COLORS["text"], transform=ax_diag.transAxes)
    ax_diag.text(0.5, 0.42,
                 disease.replace("_", " ").title(),
                 ha="center", va="center", fontsize=16, fontweight="bold",
                 color=diag_color, transform=ax_diag.transAxes,
                 bbox=dict(boxstyle="round,pad=0.6", facecolor=COLORS["panel"],
                           edgecolor=diag_color, linewidth=2))
    ax_diag.text(0.5, 0.18,
                 f"Plant: {plant_type.title()}",
                 ha="center", va="center", fontsize=10,
                 color=COLORS["text"], transform=ax_diag.transAxes)
    ax_diag.set_title("🏥  Diagnosis", fontsize=11)

    plt.savefig("diagnosis_report.png", dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    print("\n[OK] Report saved to 'diagnosis_report.png'")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MAIN INFERENCE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def run_inference() -> None:
    print("\n" + "═" * 60)
    print("     🌿  Plant Disease Detection — Inference Pipeline")
    print("═" * 60 + "\n")

    # ── Load models once ─────────────────────────────────────────────────────
    vision_model       = load_vision_model()
    dt_model, encoders = load_disease_model()

    known_plants = encoders["plant_encoder"].classes_.tolist()

    print()
    # ── User inputs ──────────────────────────────────────────────────────────
    image_path = input("📷  Path to plant image: ").strip().strip('"\'')
    if not os.path.exists(image_path):
        sys.exit(f"[ERROR] Image file not found: '{image_path}'")

    plant_type = input(
        f"🌱  Plant type {known_plants}: "
    ).strip().lower()

    location = input(
        "📍  Location (e.g. 'Asansol, India'): "
    ).strip()

    # ── Geolocation ──────────────────────────────────────────────────────────
    print("\n[→] Resolving location…")
    lat, lon = get_coords(location)
    print(f"    Coordinates: {lat:.4f}°N, {lon:.4f}°E")

    # ── Fetch weather ────────────────────────────────────────────────────────
    print("[→] Fetching live weather data…")
    weather, hourly_df = fetch_weather(lat, lon)

    temp_cat, hum_cat, soil_cat = _weather_to_categorical(
        weather["temperature_2m"],
        weather["relative_humidity_2m"],
        weather["soil_moisture_0_to_1cm"],
    )
    print(f"    Temperature : {weather['temperature_2m']:.1f} °C  → '{temp_cat}'")
    print(f"    Humidity    : {weather['relative_humidity_2m']:.1f} %   → '{hum_cat}'")
    print(f"    Soil moist. : {weather['soil_moisture_0_to_1cm']:.3f}      → '{soil_cat}'")

    # ── Vision inference ──────────────────────────────────────────────────────
    print("[→] Running visual symptom detection…")
    symptom_probs, symptom_binary, vision_binary_dict = predict_symptoms(image_path, vision_model)

    detected_vision = [s for s, b in vision_binary_dict.items() if b]
    print(f"    Detected vision classes: {detected_vision if detected_vision else ['none']}")
    
    detected_tabular = [s for s, b in zip(TABULAR_SYMPTOMS, symptom_binary) if b]
    print(f"    Mapped physical symptoms: {detected_tabular if detected_tabular else ['none']}")

    # ── Disease prediction ────────────────────────────────────────────────────
    print("[→] Running disease classification…")
    disease = predict_disease(
        plant_type, symptom_binary,
        temp_cat, hum_cat, soil_cat,
        dt_model, encoders,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(textwrap.dedent(f"""\
        PLANT   : {plant_type.title()}
        LOCATION: {location}  ({lat:.4f}°N, {lon:.4f}°E)

        WEATHER (current hour)
          Temperature  : {weather['temperature_2m']:.1f} °C
          Humidity     : {weather['relative_humidity_2m']:.1f} %
          Rain         : {weather['rain']:.2f} mm
          Precipitation: {weather['precipitation']:.2f} mm
          Cloud cover  : {weather['cloud_cover']:.1f} %
          Soil moisture: {weather['soil_moisture_0_to_1cm']:.3f} m³/m³
          Soil temp    : {weather['soil_temperature_0cm']:.1f} °C

        SYMPTOM PROBABILITIES (Vision Model)
    """))
    for sym, prob in symptom_probs.items():
        flag = "✔" if vision_binary_dict[sym] else " "
        bar  = "█" * int(prob * 20)
        print(f"    {flag} {sym:<14} {bar:<20} {prob:.3f}")
        
    print("\n        MAPPED PHYSICAL SYMPTOMS (Tabular Model)")
    for sym, b in zip(TABULAR_SYMPTOMS, symptom_binary):
        flag = "✔" if b else " "
        print(f"    {flag} {sym:<14}")

    print(f"""
        ──────────────────────────────────────────────────
        🏥  FINAL DIAGNOSIS : {disease.replace("_", " ").upper()}
        ──────────────────────────────────────────────────
    """)
    print("═" * 60)

    # ── Visualise ─────────────────────────────────────────────────────────────
    visualise_results(
        image_path, symptom_probs, symptom_binary,
        disease, weather, plant_type, location, vision_binary_dict
    )


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_inference()

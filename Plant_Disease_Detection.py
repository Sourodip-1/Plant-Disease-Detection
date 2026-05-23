# pyrefly: ignore [missing-import]
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import timm
from geopy.geocoders import Nominatim
import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Global Constants
SYMPTOM_NAMES = ['Leaf Spot', 'Blight', 'Rust', 'Mold', 'Scab', 'Healthy', 'Viral']

# Colab check
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Mappings
SYMPTOM_MAP = {
    'Apple___Apple_scab': [0, 0, 0, 0, 1, 0, 0],
    'Apple___Black_rot': [1, 0, 0, 0, 0, 0, 0],
    'Apple___Cedar_apple_rust': [0, 0, 1, 0, 0, 0, 0],
    'Apple___healthy': [0, 0, 0, 0, 0, 1, 0],
    'Potato___Early_blight': [0, 1, 0, 0, 0, 0, 0],
    'Potato___Late_blight': [0, 1, 0, 1, 0, 0, 0],
    'Potato___healthy': [0, 0, 0, 0, 0, 1, 0],
    'Tomato___Bacterial_spot': [1, 0, 0, 0, 0, 0, 0],
    'Tomato___Target_Spot': [1, 0, 0, 0, 0, 0, 0],
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [0, 0, 0, 0, 0, 0, 1],
    'Tomato___healthy': [0, 0, 0, 0, 0, 1, 0]
}

def get_symptoms(disease_label):
    return SYMPTOM_MAP.get(disease_label, [0, 0, 0, 0, 0, 0, 0])

# Dataset Class
class PlantSymptomDataset(Dataset):
    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.transform = transform
    def __len__(self):
        return len(self.metadata)
    def __getitem__(self, idx):
        item = self.metadata[idx]
        image = Image.open(item['path']).convert('RGB')
        label = torch.tensor(item['symptoms'], dtype=torch.float32)
        if self.transform: image = self.transform(image)
        return image, label

# EfficientNet-B0 builder (used in the large 52-cell notebook)
def build_model(num_classes=7, pretrained=True):
    m = timm.create_model('efficientnet_b0', pretrained=pretrained)
    in_features = m.classifier.in_features
    m.classifier = nn.Linear(in_features, num_classes)
    return m

# ResNet18 builder (used in the 5-cell/generate_notebook.py variant)
def build_resnet_model(num_classes=7, pretrained=True):
    from torchvision import models as tv_models
    m = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def load_model_auto(ckpt_path, num_classes=7, device=None):
    """Load a checkpoint, auto-detecting ResNet18 vs EfficientNet-B0 from keys."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    first_key = next(iter(state_dict))
    # ResNet18 checkpoints use 'conv1.weight' or 'layer1.*'
    if first_key.startswith('conv1') or first_key.startswith('layer') or 'fc.weight' in state_dict:
        m = build_resnet_model(num_classes=num_classes, pretrained=False)
        arch = 'ResNet18'
    else:
        m = build_model(num_classes=num_classes, pretrained=False)
        arch = 'EfficientNet-B0'
    m.load_state_dict(state_dict)
    m = m.to(device)
    m.eval()
    print(f'Loaded {arch} weights from: {ckpt_path}')
    return m

# Initialize Global Model — try to load the best available checkpoint
_CHECKPOINTS = (
    'models/plant_symptom_model.pth',
    'final_plant_model.pth',
    'models/final_plant_model.pth',
    'models/final_plant_model_v2.pth',
    'final_plant_model_v2.pth',
)
model = None
for _ckpt in _CHECKPOINTS:
    if os.path.exists(_ckpt):
        try:
            model = load_model_auto(_ckpt, num_classes=len(SYMPTOM_NAMES), device=device)
            break
        except Exception as _e:
            print(f'Skipping {_ckpt}: {_e}')
if model is None:
    print('No valid checkpoint found — initialising untrained EfficientNet-B0.')
    model = build_model(num_classes=len(SYMPTOM_NAMES), pretrained=True).to(device)

# Training utilities
def run_epoch(model, loader, optimizer, criterion, device, is_train=True):
    model.train() if is_train else model.eval()
    running_loss = 0.0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_train: optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

# Visualization
def plot_symptom_probs(model, dataset, idx, device):
    model.eval()
    img_tensor, label = dataset[idx]

    # Predict
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(device))
        probs = torch.sigmoid(output).squeeze().cpu().numpy()

    # Denormalize image for display
    img_display = img_tensor.permute(1, 2, 0).numpy()
    img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_display = np.clip(img_display, 0, 1)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(img_display)
    ax1.set_title(f"True Disease: {dataset.metadata[idx]['disease']}")
    ax1.axis('off')

    y_pos = np.arange(len(SYMPTOM_NAMES))
    ax2.barh(y_pos, probs, color='skyblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(SYMPTOM_NAMES)
    ax2.set_xlim(0, 1)
    ax2.set_title("Predicted Symptom Probabilities")
    ax2.set_xlabel("Probability")

    plt.tight_layout()
    plt.show()

# Inference v1
def predict_symptoms(image_path, model, transform, device, threshold=0.5):
    """Predicts binary symptoms and raw probabilities for a single image."""
    model.eval()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    predictions = (probs > threshold).astype(int)
    symptom_results = dict(zip(SYMPTOM_NAMES, probs))

    return symptom_results, predictions

# Inference v3
def predict_symptoms_v3(image_path, model, transform, device, threshold=0.3):
    """Improved symptom prediction with a lower sensitivity threshold."""
    model.eval()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Using a lower threshold to capture emerging symptoms
    predictions = (probs > threshold).astype(int)
    symptom_results = dict(zip(SYMPTOM_NAMES, probs))

    return symptom_results, predictions

# Geolocation
def get_coords(location_name):
    geolocator = Nominatim(user_agent="plant_app")
    location = geolocator.geocode(location_name)
    if location:
        return location.latitude, location.longitude
    else:
        print(f"Location '{location_name}' not found. Using default (0,0).")
        return 0.0, 0.0

# Weather
def get_real_time_weather(lat, lon):
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relative_humidity_2m", "soil_moisture_0_to_1cm"],
        "forecast_days": 1
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Get current (most recent) hourly values
    hourly = response.Hourly()
    temp = hourly.Variables(0).ValuesAsNumpy()[-1]
    humidity = hourly.Variables(1).ValuesAsNumpy()[-1]
    soil_moisture = hourly.Variables(2).ValuesAsNumpy()[-1]

    return {"temperature": temp, "humidity": humidity, "soil_moisture": soil_moisture}

# Tabular model variables and loader
tabular_model = None
tabular_data = None
le_dict = {}

def load_and_train_tabular_model():
    global tabular_model, tabular_data, le_dict
    csv_path = 'data.csv'
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Cannot initialize tabular model.")
        return

    tabular_data = pd.read_csv(csv_path)
    le_dict = {}
    for col in tabular_data.columns:
        if tabular_data[col].dtype == 'object':
            le = LabelEncoder()
            tabular_data[col] = le.fit_transform(tabular_data[col])
            le_dict[col] = le

    target_col = tabular_data.columns[-1]
    X_train = tabular_data.drop(columns=[target_col])
    y_train = tabular_data[target_col]

    tabular_model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=10,
        random_state=42
    )
    tabular_model.fit(X_train, y_train)

# Hybrid Diagnosis
def hybrid_diagnosis(image_path, lat, lon, cv_model, transform, device):
    results, visual_vector = predict_symptoms(image_path, cv_model, transform, device)
    env_data = get_real_time_weather(lat, lon)
    hybrid_features = list(visual_vector) + [env_data['temperature'], env_data['humidity'], env_data['soil_moisture']]

    print("--- Hybrid Diagnosis Report ---")
    print(f"Visual Symptoms Detected: {[SYMPTOM_NAMES[i] for i, val in enumerate(visual_vector) if val > 0]}")
    print(f"Environmental Stress: Temp {env_data['temperature']:.1f}C, Humidity {env_data['humidity']:.1f}%")

    if env_data['humidity'] > 80 and visual_vector[1] > 0:
        print("WARNING: High humidity detected. High risk of Blight spread.")
    elif visual_vector[5] > 0:
        print("STATUS: Plant appears healthy.")

    return hybrid_features

# Hybrid Diagnosis Final
def hybrid_diagnosis_final(image_path, lat, lon, cv_model, transform, device):
    results, visual_vector = predict_symptoms(image_path, cv_model, transform, device)
    env_data = get_real_time_weather(lat, lon)
    raw_features = list(visual_vector) + [env_data['temperature'], env_data['humidity'], env_data['soil_moisture']]

    global tabular_model, tabular_data, le_dict
    if tabular_model is None:
        load_and_train_tabular_model()

    if tabular_model is not None:
        if tabular_model.n_features_in_ == 11:
            final_vector = [0] + [float(f) for f in raw_features]
        else:
            final_vector = [float(f) for f in raw_features]

        prediction_idx = tabular_model.predict([final_vector])[0]
        target_col = tabular_data.columns[-1]
        final_result = le_dict[target_col].inverse_transform([prediction_idx])[0] if target_col in le_dict else prediction_idx
    else:
        final_result = "Tabular model not available"

    detected_symptoms = [SYMPTOM_NAMES[i] for i, val in enumerate(visual_vector) if val > 0 and i != 5]
    if str(final_result).lower() == 'healthy' and len(detected_symptoms) > 0:
        final_result = f"Potential {detected_symptoms[0]} (Risk detected by CV)"

    print("\n--- Fine-tuned Decision Support Report ---")
    print(f"Location: {lat}, {lon}")
    print(f"Detected Symptoms: {[SYMPTOM_NAMES[i] for i, val in enumerate(visual_vector) if val > 0]}")
    print(f"Local Weather: {env_data['temperature']:.1f}°C, {env_data['humidity']:.1f}% Hum")
    print(f"\nFINAL DIAGNOSIS: {final_result}")

    return final_result

# Run final inference functions
def run_final_inference():
    print("Step 1: Upload/Get your plant image")
    if IN_COLAB:
        uploaded = files.upload()
        if not uploaded:
            return
        image_path = list(uploaded.keys())[0]
    else:
        image_path = input("Enter the path to the plant photo: ").strip('\"\'')
        if not os.path.exists(image_path):
            print("File not found!")
            return

    print("\nStep 2: Enter Location Details")
    location_input = "asansol,west bengal,india" # @param {type:\"string\"}

    lat, lon = get_coords(location_input)
    print(f"Resolved Coordinates: {lat}, {lon}")

    global model
    features = hybrid_diagnosis(image_path, lat, lon, model, val_transform, device)
    plot_symptom_probs(model, PlantSymptomDataset([{'path': image_path, 'disease': 'User Upload', 'symptoms': [0]*7}], transform=val_transform), 0, device)

def run_final_inference_v2():
    print("Step 1: Upload/Get your plant image")
    if IN_COLAB:
        uploaded = files.upload()
        if not uploaded: return
        image_path = list(uploaded.keys())[0]
    else:
        image_path = input("Enter the path to the plant photo: ").strip('\"\'')
        if not os.path.exists(image_path):
            print("File not found!")
            return

    try:
        location_input = "asansol,west bengal,india" # @param {type:\"string\"}
        lat, lon = get_coords(location_input)

        global model
        hybrid_diagnosis_final(image_path, lat, lon, model, val_transform, device)
        plot_symptom_probs(model, PlantSymptomDataset([{'path': image_path, 'disease': 'User Upload', 'symptoms': [0]*7}], transform=val_transform), 0, device)

    except Exception as e:
        print(f"An error occurred during inference: {e}")

    finally:
        if IN_COLAB and os.path.exists(image_path):
            os.remove(image_path)
            print(f"\nCleanup: Deleted {image_path}")

def run_final_inference_v3():
    print("Step 1: Upload/Get your plant image")
    if IN_COLAB:
        uploaded = files.upload()
        if not uploaded: return
        image_path = list(uploaded.keys())[0]
    else:
        image_path = input("Enter the path to the plant photo: ").strip('\"\'')
        if not os.path.exists(image_path):
            print("File not found!")
            return

    try:
        location_input = "asansol,west bengal,india"
        lat, lon = get_coords(location_input)

        global model
        results, visual_vector = predict_symptoms_v3(image_path, model, val_transform, device, threshold=0.25)
        env_data = get_real_time_weather(lat, lon)
        detected = [SYMPTOM_NAMES[i] for i, val in enumerate(visual_vector) if val > 0 and i != 5]

        print("\n--- Improved Decision Support Report ---")
        print(f"Detected Symptoms: {detected if detected else 'None Significant'}")
        print(f"Weather: {env_data['temperature']:.1f}°C, {env_data['humidity']:.1f}% Humidity")

        plot_symptom_probs(model, PlantSymptomDataset([{'path': image_path, 'disease': 'User Upload', 'symptoms': [0]*7}], transform=val_transform), 0, device)

    finally:
        if IN_COLAB and os.path.exists(image_path): os.remove(image_path)

def final_production_inference():
    print("--- Plant Disease Diagnosis System ---")
    img_path = input("Enter the path to the plant photo: ").strip('\"\'')
    if not os.path.exists(img_path):
        print("File not found!")
        return

    loc = input("Enter location (e.g. 'asansol, india'): ")
    lat, lon = get_coords(loc)

    global model
    if os.path.exists('final_plant_model_v2.pth'):
        model.load_state_dict(torch.load('final_plant_model_v2.pth', map_location=device, weights_only=True))
        print("Loaded final_plant_model_v2.pth successfully.")
    elif os.path.exists('models/final_plant_model_v2.pth'):
        model.load_state_dict(torch.load('models/final_plant_model_v2.pth', map_location=device, weights_only=True))
        print("Loaded models/final_plant_model_v2.pth successfully.")

    # 1. Vision Analysis
    results, vector = predict_symptoms_v3(img_path, model, val_transform, device, threshold=0.4)

    # 2. Hybrid Report
    hybrid_diagnosis_final(img_path, lat, lon, model, val_transform, device)

    # 3. Visual Insight
    plot_symptom_probs(model, PlantSymptomDataset([{'path': img_path, 'disease': 'User Input', 'symptoms': [0]*7}], transform=val_transform), 0, device)

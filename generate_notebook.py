import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import json

nb = new_notebook()

nb.cells.append(new_markdown_cell('# Plant Disease Detection Pipeline\n\nThis notebook trains a symptom detection model on plant leaves and integrates a decision tree to predict the final disease.'))

code1 = '''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
'''
nb.cells.append(new_code_cell(code1.strip()))

code2 = '''
# Define symptoms
symptoms_list = ['spots', 'yellowing', 'wilting', 'powder', 'leaf_curl', 'stem_damage', 'mold_growth']

# Symptom Mapping Heuristics based on disease names
def get_symptoms(class_name):
    name = class_name.lower()
    symptoms = [0, 0, 0, 0, 0, 0, 0]
    
    if 'healthy' in name:
        return symptoms
        
    if any(x in name for x in ['spot', 'scab', 'blight', 'rust', 'measles']):
        symptoms[0] = 1 # spots
    if any(x in name for x in ['yellow', 'mosaic', 'greening']):
        symptoms[1] = 1 # yellowing
    if any(x in name for x in ['wilt', 'blight', 'scorch']):
        symptoms[2] = 1 # wilting
    if any(x in name for x in ['mildew', 'mold']):
        symptoms[3] = 1 # powder
        symptoms[6] = 1 # mold_growth
    if 'curl' in name:
        symptoms[4] = 1 # leaf_curl
    if any(x in name for x in ['rot', 'canker']):
        symptoms[5] = 1 # stem_damage
        
    return symptoms

class SymptomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                symptoms = get_symptoms(class_name)
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.JPG', '.png', '.PNG')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(symptoms)
                        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Create dataset and dataloaders
data_dir = 'PlantVillage-Dataset-master/raw/color'
transform = transforms.Compose([
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

dataset = SymptomDataset(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

print(f"Total images: {len(dataset)}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")
'''
nb.cells.append(new_code_cell(code2.strip()))

code3 = '''
# Build and Train the Model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7) # 7 symptoms
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10 # Increased epochs

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

torch.save(model.state_dict(), 'final_plant_model_v2.pth')
print("Model saved to final_plant_model_v2.pth")
'''
nb.cells.append(new_code_cell(code3.strip()))

code4 = '''
# Inference & Integration Pipeline
def predict_symptoms(img_path, model, transform, device, threshold=0.4):
    model.eval()
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
    symptoms_detected = (probs > threshold).astype(int)
    return symptoms_detected

def final_production_inference():
    print("--- Plant Disease Diagnosis System ---")
    img_path = input("Enter the path to the plant photo: ").strip('"\\'')
    if not os.path.exists(img_path):
        print("File not found!")
        return
        
    plant_type = input("Enter plant type (e.g. tomato, potato, apple): ")
    humidity = input("Enter humidity (low, medium, high): ")
    temperature = input("Enter temperature (low, medium, high): ")
    soil_moisture = input("Enter soil moisture (dry, moderate, wet): ")
    
    # 1. Vision Analysis for Symptoms
    symptoms = predict_symptoms(img_path, model, val_transform, device)
    print(f"Detected Symptoms: {dict(zip(symptoms_list, symptoms))}")
    
    # 2. Decision Tree Prediction
    try:
        dt_model = joblib.load('disease_prediction_model.pkl')
        encoders = joblib.load('encoders.pkl')
    except Exception as e:
        print("Error loading decision tree model or encoders. Make sure model.ipynb has been run.")
        return
        
    try:
        p_type = encoders['plant_encoder'].transform([plant_type])[0]
        h_type = encoders['humidity_encoder'].transform([humidity])[0]
        t_type = encoders['temperature_encoder'].transform([temperature])[0]
        s_type = encoders['soil_encoder'].transform([soil_moisture])[0]
    except Exception as e:
        print("Invalid input for environmental factors. Returning default prediction.")
        p_type, h_type, t_type, s_type = 0, 0, 0, 0
        
    # Feature vector: plant_type, spots, yellowing, wilting, powder, leaf_curl, stem_damage, mold_growth, humidity, temperature, soil_moisture
    features = [[p_type] + list(symptoms) + [h_type, t_type, s_type]]
    pred = dt_model.predict(features)
    disease_name = encoders['disease_encoder'].inverse_transform(pred)[0]
    
    print(f"\\n>>> Final Diagnosed Disease: {disease_name} <<<")

# Example run using a random image
example_img = dataset.image_paths[0]
print(f"\\nTesting inference with {example_img}")
symps = predict_symptoms(example_img, model, val_transform, device)
print("Predicted symptoms for test image:", symps)
'''
nb.cells.append(new_code_cell(code4.strip()))

with open('Plant_Disease_Detection.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print('Notebook Plant_Disease_Detection.ipynb created successfully!')

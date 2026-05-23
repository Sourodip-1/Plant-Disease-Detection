import json
import os

notebook_path = r'd:\Codes\Plant-Disease-Detection\Plant_Disease_Detection.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    "import os\n",
    "import torch\n",
    "\n",
    "def final_production_inference():\n",
    "    print(\"--- Plant Disease Diagnosis System ---\")\n",
    "    img_path = input(\"Enter the path to the plant photo: \").strip('\"\\'')\n",
    "    if not os.path.exists(img_path):\n",
    "        print(\"File not found!\")\n",
    "        return\n",
    "\n",
    "    loc = input(\"Enter location (e.g. 'asansol, india'): \")\n",
    "    lat, lon = get_coords(loc)\n",
    "\n",
    "    if os.path.exists('final_plant_model.pth'):\n",
    "        model.load_state_dict(torch.load('final_plant_model.pth', map_location=device, weights_only=True))\n",
    "        print(\"Loaded final_plant_model.pth successfully.\")\n",
    "    elif os.path.exists('models/final_plant_model.pth'):\n",
    "        model.load_state_dict(torch.load('models/final_plant_model.pth', map_location=device, weights_only=True))\n",
    "        print(\"Loaded models/final_plant_model.pth successfully.\")\n",
    "\n",
    "    # 1. Vision Analysis\n",
    "    results, vector = predict_symptoms_v3(img_path, model, val_transform, device, threshold=0.4)\n",
    "\n",
    "    # 2. Hybrid Report\n",
    "    hybrid_diagnosis_final(img_path, lat, lon, model, val_transform, device)\n",
    "\n",
    "    # 3. Visual Insight\n",
    "    plot_symptom_probs(model, PlantSymptomDataset([{'path': img_path, 'disease': 'User Input', 'symptoms': [0]*7}], transform=val_transform), 0, device)\n",
    "\n",
    "final_production_inference()\n"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        if 'def final_production_inference():' in source_text:
            cell['source'] = new_source
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color_plt=sns.color_palette()
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv("data.csv")
data["disease"].value_counts().plot(kind="bar", figsize=(10,5))

plt.title("Disease Distribution")
plt.xlabel("Disease")
plt.ylabel("Count")

plt.show()

symptoms = [
    "spots",
    "yellowing",
    "wilting",
    "powder",
    "leaf_curl",
    "stem_damage",
    "mold_growth"
]

data[symptoms].sum().plot(kind="bar", figsize=(10,5))

plt.title("Symptom Frequency")
plt.ylabel("Total Occurrences")
plt.show()

plt.figure(figsize=(10,6))

sns.heatmap(
    data[symptoms].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Symptom Correlation Heatmap")
plt.show()



pivot = data.groupby("disease")[symptoms].mean()

plt.figure(figsize=(12,6))

sns.heatmap(
    pivot,
    annot=True,
    cmap="YlGnBu"
)

plt.title("Average Symptoms per Disease")
plt.show()

from sklearn.preprocessing import LabelEncoder

plant_encoder = LabelEncoder()
humidity_encoder = LabelEncoder()
temperature_encoder = LabelEncoder()
soil_encoder = LabelEncoder()
disease_encoder = LabelEncoder()

data["plant_type"] = plant_encoder.fit_transform(data["plant_type"])

data["humidity"] = humidity_encoder.fit_transform(data["humidity"])

data["temperature"] = temperature_encoder.fit_transform(data["temperature"])

data["soil_moisture"] = soil_encoder.fit_transform(data["soil_moisture"])

data["disease"] = disease_encoder.fit_transform(data["disease"])


x = data.drop("disease", axis=1)
y = data["disease"]

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


train_counts = y_train.value_counts()
test_counts = y_test.value_counts()

fig, ax = plt.subplots(figsize=(12,5))

train_counts.plot(kind="bar", ax=ax, alpha=0.7, label="Train")
test_counts.plot(kind="bar", ax=ax, alpha=0.7, label="Test",color=color_plt[1])

plt.title("Train vs Test Disease Distribution")
plt.xlabel("Disease")
plt.ylabel("Count")
plt.legend()

plt.show()




symptoms = [
    "spots",
    "yellowing",
    "wilting",
    "powder",
    "leaf_curl",
    "stem_damage",
    "mold_growth"
]

train_avg = X_train[symptoms].mean()
test_avg = X_test[symptoms].mean()

comparison = {
    "Train": train_avg,
    "Test": test_avg
}

import pandas as pd

comparison_df = pd.DataFrame(comparison)

comparison_df.plot(kind="bar", figsize=(10,5))

plt.title("Average Symptom Presence")
plt.ylabel("Average Value")
plt.show()

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

# Generate predictions and predicted names for the test set
predictions = model.predict(X_test)
predicted_names = disease_encoder.inverse_transform(predictions)

# Serialize the decision tree model and label encoders
import joblib
joblib.dump(model, 'disease_prediction_model.pkl')
joblib.dump({
    'plant_encoder': plant_encoder,
    'humidity_encoder': humidity_encoder,
    'temperature_encoder': temperature_encoder,
    'soil_encoder': soil_encoder,
    'disease_encoder': disease_encoder
}, 'encoders.pkl')
print("Model and encoders saved successfully.")

actual_names = disease_encoder.inverse_transform(y_test)
plant_names = plant_encoder.inverse_transform(
    X_test["plant_type"]
)
c = 1

for i in range(len(predicted_names)):
    print("Test Case:", c)
    print("plant name:", plant_names[i])
    print("Input Features:")
    print(X_test.iloc[i].to_dict())
    print("Predicted Disease:", predicted_names[i])
    print("Actual Disease:", actual_names[i])
    print("-" * 50)
    c += 1


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

custom_input = [[
    plant_encoder.transform(["tomato"])[0],
    0.1,   # spots
    0.3,   # yellowing
    0.0,  # wilting
    0.0,     # powder
    0.8,   # leaf_curl
    0.0,   # stem_damage
    0.0,   # mold_growth
    humidity_encoder.transform(["high"])[0],
    temperature_encoder.transform(["high"])[0],
    soil_encoder.transform(["wet"])[0]
]]

prediction=model.predict(custom_input)

disease_name = disease_encoder.inverse_transform(prediction)

print("Predicted Disease:", disease_name[0])




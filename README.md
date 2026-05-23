---
title: Plant Disease Detection
emoji: 🌿
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Next Js integration with Api

```
    // Example Next.js API route or Client component function
    export async function diagnosePlant(imageFile, plantType, location) {
    // 1. Prepare form data matching the FastAPI endpoint parameters
    const formData = new FormData();
    formData.append("file", imageFile); // The actual image File object
    formData.append("plant_type", plantType); // e.g. "potato"
    formData.append("location", location); // e.g. "Asansol, India"

    const HF_SPACE_URL = "https://YOUR_HF_USERNAME-YOUR_SPACE_NAME.hf.space";

    try {
    const response = await fetch(`${HF_SPACE_URL}/predict`, {
    method: "POST",
    body: formData,
    // Note: Do NOT manually set Content-Type header;
    // the browser will automatically set it to multipart/form-data with boundary info
    });

        if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
        }

        const result = await response.json();
        return result; // Contains status, diagnosis, symptoms, weather, etc.

    } catch (error) {
    console.error("Failed to diagnose plant:", error);
    throw error;
    }
    }
```

# Plant Disease Detection — API & Dashboard

This repository hosts a hybrid machine learning pipeline for plant disease diagnosis:

- **Computer Vision Model (ResNet18)** to detect leaf symptoms.
- **Decision Tree Model** to combine symptoms and live weather data for the final classification.
- **FastAPI Backend** exposing prediction endpoints.
- **Glassmorphic Frontend Dashboard** to interact with the API.

## Hugging Face Spaces Deployment

This repository is configured to deploy directly to Hugging Face Spaces using Docker. It builds a lightweight container, installs CPU-only PyTorch, and serves the dashboard directly on the root URL.

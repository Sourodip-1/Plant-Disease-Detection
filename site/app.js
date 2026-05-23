document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('file-dropzone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const dropzoneContent = dropzone.querySelector('.dropzone-content');
    
    const form = document.getElementById('diagnosis-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnSpinner = document.getElementById('btn-spinner');
    const resultsSection = document.getElementById('results-section');
    
    // Result elements
    const diagnosisCard = document.getElementById('diagnosis-card');
    const diagName = document.getElementById('diag-name');
    const diagPlantInfo = document.getElementById('diag-plant-info');
    const visionSymptomsList = document.getElementById('vision-symptoms-list');
    const physicalSymptomsList = document.getElementById('physical-symptoms-list');
    
    // Weather elements
    const wxTemp = document.getElementById('wx-temp');
    const wxTempCat = document.getElementById('wx-temp-cat');
    const wxHumidity = document.getElementById('wx-humidity');
    const wxHumidityCat = document.getElementById('wx-humidity-cat');
    const wxSoil = document.getElementById('wx-soil');
    const wxSoilCat = document.getElementById('wx-soil-cat');
    const wxRain = document.getElementById('wx-rain');
    
    let selectedFile = null;

    // Auto-detect API URL based on origin if hosted on a server
    if (window.location.protocol.startsWith('http')) {
        document.getElementById('api-url-input').value = window.location.origin;
    }

    // ==========================================================================
    // DROPZONE & FILE UPLOAD HANDLERS
    // ==========================================================================
    dropzone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag-and-drop visual indicators
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            fileInput.files = files; // Sync to file input
            handleFile(files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (PNG, JPG, or JPEG).');
            return;
        }
        
        selectedFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            dropzoneContent.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    // ==========================================================================
    // FORM SUBMISSION & API CALLED
    // ==========================================================================
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!selectedFile) {
            alert('Please select or upload a leaf image first.');
            return;
        }

        const apiBaseUrl = document.getElementById('api-url-input').value.replace(/\/$/, "");
        const plantType = document.getElementById('plant-type-select').value;
        const location = document.getElementById('location-input').value;

        // Set Loading State
        submitBtn.disabled = true;
        btnSpinner.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('plant_type', plantType);
        formData.append('location', location);

        try {
            const response = await fetch(`${apiBaseUrl}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Server responded with status code ${response.status}`);
            }

            const json = await response.json();
            
            if (json.status === 'success') {
                renderResults(json.data);
            } else {
                throw new Error(json.message || 'API request failed');
            }

        } catch (error) {
            console.error(error);
            alert(`Diagnosis Failed: ${error.message}`);
        } finally {
            // Restore button state
            submitBtn.disabled = false;
            btnSpinner.classList.add('hidden');
        }
    });

    // ==========================================================================
    // RENDER OUTCOMES
    // ==========================================================================
    function renderResults(data) {
        const isHealthy = data.diagnosis.toLowerCase().includes('healthy');

        // 1. Diagnosis card styling
        if (isHealthy) {
            diagnosisCard.className = 'diagnosis-card healthy';
            diagName.textContent = 'Healthy Plant';
        } else {
            diagnosisCard.className = 'diagnosis-card';
            diagName.textContent = data.diagnosis;
        }

        diagPlantInfo.textContent = `${capitalize(data.plant_type)} — diagnosed at ${data.location.name} (${data.location.lat.toFixed(2)}°N, ${data.location.lon.toFixed(2)}°E)`;

        // 2. Weather mapping
        wxTemp.textContent = `${data.weather.temperature_c.toFixed(1)} °C`;
        wxTempCat.textContent = capitalize(data.weather.temperature_category);
        
        wxHumidity.textContent = `${data.weather.humidity_pct.toFixed(0)} %`;
        wxHumidityCat.textContent = capitalize(data.weather.humidity_category);
        
        wxSoil.textContent = `${data.weather.soil_moisture.toFixed(3)} m³/m³`;
        wxSoilCat.textContent = capitalize(data.weather.soil_category);
        
        wxRain.textContent = `${data.weather.rain_mm.toFixed(2)} mm`;

        // 3. Vision probabilities
        visionSymptomsList.innerHTML = '';
        const sortedVision = Object.entries(data.symptoms.vision_probabilities)
            .sort((a, b) => b[1] - a[1]); // Descending probabilities

        sortedVision.forEach(([symptom, prob]) => {
            const group = document.createElement('div');
            group.className = 'symptom-bar-group';
            
            const isDetected = prob >= 0.40;
            const barColor = isDetected ? 'var(--accent-danger)' : 'var(--accent-emerald)';
            if (symptom === 'Healthy') {
                // Flip colors for 'Healthy'
                group.style.opacity = prob >= 0.5 ? '1' : '0.65';
            }

            group.innerHTML = `
                <div class="symptom-info">
                    <span class="symptom-name">${symptom} ${isDetected && symptom !== 'Healthy' ? '⚠️' : ''}</span>
                    <span class="symptom-val">${(prob * 100).toFixed(1)}%</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: 0%; background: ${barColor}"></div>
                </div>
            `;
            visionSymptomsList.appendChild(group);

            // Trigger animation in next microtask
            setTimeout(() => {
                group.querySelector('.progress-fill').style.width = `${prob * 100}%`;
            }, 50);
        });

        // 4. Binary Symptoms (Decision Tree mapping)
        physicalSymptomsList.innerHTML = '';
        
        // Let's get the active symptoms from the API response array
        const activeSymptoms = new Set(data.symptoms.detected_physical_symptoms);
        
        // Define all target features in the DT model
        const allSymptoms = ["spots", "yellowing", "wilting", "powder", "leaf_curl", "stem_damage", "mold_growth"];
        
        allSymptoms.forEach(symptom => {
            const hasSymptom = activeSymptoms.has(symptom);
            const tag = document.createElement('div');
            tag.className = `binary-tag ${hasSymptom ? 'detected' : 'clear'}`;
            tag.innerHTML = `
                <span>${capitalize(symptom.replace('_', ' '))}</span>
                <span>${hasSymptom ? '⚠️' : '✓'}</span>
            `;
            physicalSymptomsList.appendChild(tag);
        });

        // Show Results Section
        resultsSection.classList.remove('hidden');
        
        // Smooth scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Update timestamp
        const now = new Date();
        document.getElementById('report-timestamp').textContent = `Report: ${now.toLocaleTimeString()}`;
    }

    // Helper functions
    function capitalize(string) {
        if (!string) return '';
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
});

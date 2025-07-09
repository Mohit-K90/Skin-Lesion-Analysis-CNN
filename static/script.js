const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const fileInfo = document.getElementById('fileInfo');
const predictBtn = document.getElementById('predictBtn');
const resetBtn = document.getElementById('resetBtn');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const errorMessage = document.getElementById('errorMessage');
const results = document.getElementById('results');

let selectedFile = null;

// Initialize event listeners
function init() {
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    predictBtn.addEventListener('click', makePrediction);
    resetBtn.addEventListener('click', resetForm);
}

// File selection handler
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        displayPreview(file);
    }
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave() {
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            selectedFile = file;
            displayPreview(file);
        } else {
            showError('Please select a valid image file');
        }
    }
}

// Display image preview
function displayPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        fileInfo.textContent = `${file.name} (${formatFileSize(file.size)})`;
        predictBtn.disabled = false;
        hideError();
        hideResults();
    };
    reader.readAsDataURL(file);
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Make prediction
async function makePrediction() {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }

    showLoading();
    hideError();
    hideResults();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (err) {
        showError(`Network error: ${err.message}`);
    } finally {
        hideLoading();
    }
}

// Display prediction results
function displayResults(data) {
    // Top prediction (no confidence bar)
    const mainClass = document.getElementById('mainClass');
    if (mainClass) mainClass.textContent = data.predicted_class;

    const mainConfidenceBar = document.getElementById('mainConfidenceBar');
    const mainConfidence = document.getElementById('mainConfidence');
    if (mainConfidenceBar) mainConfidenceBar.style.width = '0%';
    if (mainConfidence) mainConfidence.textContent = '';

    // Top 2 predictions only
    const topPredictionsDiv = document.getElementById('topPredictions');
    topPredictionsDiv.innerHTML = '';

    data.top_predictions.slice(0, 2).forEach(pred => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <div class="prediction-name">${pred.class_name}</div>
            <div class="prediction-confidence">
                <span>${(pred.confidence * 100).toFixed(2)}%</span>
                <div class="mini-bar">
                    <div class="mini-fill" style="width: ${pred.confidence * 100}%"></div>
                </div>
            </div>
        `;
        topPredictionsDiv.appendChild(item);
    });

    // Grad-CAM visualization
    const gradcamImage = document.getElementById('gradcamImage');
    const gradcamSection = document.getElementById('gradcamSection');
    if (gradcamImage && gradcamSection && data.gradcam_image) {
        gradcamImage.src = `data:image/png;base64,${data.gradcam_image}`;
        gradcamSection.style.display = 'block';
    }

    results.style.display = 'block';
    results.scrollIntoView({ behavior: 'smooth' });
}

// Reset form
function resetForm() {
    selectedFile = null;
    fileInput.value = '';
    previewSection.style.display = 'none';
    predictBtn.disabled = true;
    hideError();
    hideResults();
    hideLoading();
}

// Loading state
function showLoading() {
    loading.style.display = 'block';
    predictBtn.disabled = true;
}

function hideLoading() {
    loading.style.display = 'none';
    predictBtn.disabled = selectedFile ? false : true;
}

// Error handling
function showError(message) {
    errorMessage.textContent = message;
    error.style.display = 'block';
    error.scrollIntoView({ behavior: 'smooth' });
}

function hideError() {
    error.style.display = 'none';
}

// Results handling
function hideResults() {
    results.style.display = 'none';
}

// Validate file type
function isValidImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff'];
    return validTypes.includes(file.type);
}

// Initialize the application
document.addEventListener('DOMContentLoaded', init);

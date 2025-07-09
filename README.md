# Skin Lesion Classification with EfficientNetB3 and CBAM

A deep learning project focused on the automated classification of skin lesions using dermoscopic images from the HAM10000 dataset. The model is built on EfficientNetB3 with integrated CBAM (Convolutional Block Attention Module) for enhanced feature attention and class imbalance handling via Focal Loss.

---

## 📁 Dataset

- **HAM10000**: Human Against Machine with 10,000+ dermoscopic images
- **Classes (7 total)**:
  - `akiec` — Actinic keratoses
  - `bcc` — Basal cell carcinoma
  - `bkl` — Benign keratosis-like lesions
  - `df` — Dermatofibroma
  - `nv` — Melanocytic nevi
  - `mel` — Melanoma
  - `vasc` — Vascular lesions
- [Optional]: Used a class-balanced version with 3,500 samples per class ([Augmented HAM10000](https://www.kaggle.com/datasets/aranyasaha/augmented-ham10000-3500-image-per-class))

---

## 🧠 Model Overview

- **Backbone**: EfficientNetB3 (ImageNet pretrained)
- **Attention**: CBAM (spatial + channel)
- **Loss Function**: Focal Loss for improved recall on minority classes
- **Optimization**: Adam optimizer with learning rate scheduling (ReduceLROnPlateau)

---

## ⚙️ Training Highlights

- Applied extensive data augmentation (rotation, brightness, zoom, shifts, flips)
- EarlyStopping and learning rate reduction callbacks
- Trained on class-balanced set for improved per-class performance
- Achieved:
  - **84% overall accuracy**
  - **0.77 macro-averaged F1-score**
  - **91% F1 on melanoma**
  - **96% F1 on rare class (df)**

---

## 🔍 Model Explainability

Integrated both:
- **Grad-CAM** — Visual heatmaps highlighting important lesion regions
- **SHAP** — Feature importance scores for model interpretability

---

## 🛠️ Project Structure
Skin-Lesion-Analysis-CNN/
├── app.py # Flask app for local deployment
├── model/ # Trained model (not uploaded due to GitHub size limit)
├── notebooks/ # Training and evaluation notebooks
├── static/ # Grad-CAM outputs (optional)
├── utils.py # Visualization and helper functions
├── requirements.txt
└── README.md
---

## 🚀 Deployment

To run the Flask app locally:

```bash
pip install -r requirements.txt
python app.py
```
👤 Author

Mohit Kamkhalia

📧 mohitkamkhalia@gmail.com

🔗 GitHub | LinkedIn

Disclaimer: This tool is intended for academic and research purposes. It is not a substitute for clinical diagnosis.

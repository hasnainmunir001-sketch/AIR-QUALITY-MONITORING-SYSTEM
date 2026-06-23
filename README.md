# 🌍 Advanced Air Quality Monitoring System

Benzene pollution levels classify karne ke liye complete ML project using **UCI Air Quality dataset**.

## Problem
Air pollution public health ke liye risk hai. Yeh project hourly **benzene (C6H6)** concentration ko **Low, Medium, High** classes mein predict karta hai.

## Dataset
- **Source**: [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
- **Records**: ~9,358 hourly readings
- **Target**: `C6H6(GT)` → 3 classes by quantiles

## Approaches
1. **Supervised Learning**
   - Random Forest
   - SVM (RBF)
   - Neural Network (MLP)
2. **Unsupervised Learning**
   - K-Means clustering
3. **Semi-Supervised Learning**
   - Self-Training with 10% labels

## Run Locally

```bash
cd air-quality-monitoring
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_models.py
streamlit run app.py
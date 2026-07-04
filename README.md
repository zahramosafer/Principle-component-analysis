# Principle component analysis
# PCA from Scratch vs Scikit-learn

This project implements **Principal Component Analysis (PCA)** from scratch using NumPy and compares it with the Scikit-learn implementation.

---

## 📊 Dataset
- Iris dataset (150 samples, 4 features, 3 classes)

---

## 🧠 Methods

### 1. PCA from Scratch
Steps:
- Center the data
- Compute covariance matrix
- Perform eigen decomposition
- Sort eigenvectors
- Project data to 2D space

### 2. Scikit-learn PCA
- Uses optimized SVD-based implementation

---

## 📈 Results

Both methods produce nearly identical 2D embeddings of the dataset.

---

<img width="559" height="435" alt="output" src="https://github.com/user-attachments/assets/8287763e-6f02-4618-a116-38a96719b76b" />

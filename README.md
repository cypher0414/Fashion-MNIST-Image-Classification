# Fashion MNIST Image Classification

A deep learning project using **Convolutional Neural Networks (CNNs)** to classify images from the [Fashion MNIST dataset](https://www.kaggle.com/zalando-research/fashionmnist) into 10 categories of clothing and accessories.

---

## Overview
This project explores how CNN architectures perform on the Fashion MNIST dataset — a modern alternative to the classic MNIST digits.  
Two CNN models were trained and compared:
- **Baseline CNN** — a deeper, standard convolutional model  
- **Small CNN** — a lightweight variant with fewer parameters  

The goal is to evaluate the trade-offs between model complexity and accuracy.

---

## Dataset
The **Fashion MNIST dataset** consists of:
- 60,000 training images  
- 10,000 test images  
- Grayscale 28×28 pixel images  
- 10 categories:

---

## Model Architectures

### 1. Baseline CNN
- 3 convolutional layers (32 → 64 → 128 filters)  
- ReLU activations, MaxPooling layers  
- Dense(128) → Dense(10, softmax)  
- Optimizer: Adam  
- Loss: Categorical Crossentropy  

### 2. Small CNN
- 2 convolutional layers (32 → 64 filters)  
- Dense(64) → Dense(10, softmax)  
- Optimizer: Adam  
- Loss: Categorical Crossentropy  

---

## Results

| Model         | Test Accuracy |
|----------------|----------------|
| Baseline CNN   | 0.8974 |
| Small CNN      | 0.9025 |

The smaller CNN performed slightly better while being more efficient — showing that simpler architectures can sometimes outperform deeper ones on lightweight datasets.

---

## Evaluation & Visualization
- Training/validation **accuracy and loss curves**
- **Confusion matrix** and **classification report**
- Model comparison summary table

---

## Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib & Seaborn  
- scikit-learn  

---

## How to Run
1. Clone the repository:
 ```bash
 git clone https://github.com/yourusername/fashion-mnist-cnn.git
 cd fashion-mnist-cnn

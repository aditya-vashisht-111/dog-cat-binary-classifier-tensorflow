# 🐶 Cats vs Dogs Image Classifier (TensorFlow & Keras)

This project implements a Convolutional Neural Network (CNN) using TensorFlow 2 and Keras to classify images of cats and dogs.  
The model is trained using data augmentation techniques and achieves approximately **65–70% validation accuracy**.

---

## 📦 Project Description

This project demonstrates end-to-end image classification using a custom CNN architecture. Images are preprocessed using Keras `ImageDataGenerator`, the model is trained on labeled data, evaluated on validation data, and finally used to predict on unlabeled test images.

This work was completed as part of the FreeCodeCamp Machine Learning curriculum.

---

## 🧠 Model Overview

The architecture consists of:

```
Conv2D + MaxPool
Conv2D + MaxPool
Conv2D + MaxPool
Flatten
Dense(512) + ReLU
Dense(1) + Sigmoid
```

- **Loss:** Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Metric:** Accuracy  

---

## 📁 Dataset

Dataset includes:

| Split | Images | Classes |
|---|---|---|
| Train | 2,000 | Cat, Dog |
| Validation | 1,000 | Cat, Dog |
| Test | 50 | Unlabeled |

Test set predictions are visualized with confidence scores.

---

## 🔄 Data Augmentation

To reduce overfitting, the following augmentations were applied:

- Rotation
- Zoom
- Width & Height shift
- Shear transform
- Horizontal flip
- Rescaling

---

## 📊 Results

- **Training Accuracy:** ~70%
- **Validation Accuracy:** ~65–70%
- **Generalization:** Strong for small dataset and basic CNN

Training history, accuracy/loss plots, and example predictions can be saved under a `results/` directory.

---

## 🗂 Repository Structure

```
cats-vs-dogs-cnn-classifier/
│
├── notebook/
│   └── Cat_and_Dog_Image_Classifier.ipynb
├── results/            # optional visualization outputs
├── model/              # optional saved model (.h5)
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### **1. Install Requirements**
```bash
pip install -r requirements.txt
```

### **2. Open Notebook**
```bash
jupyter notebook
```

### **3. Train & Evaluate**
Run all cells in order.

---

## 🧩 Dependencies

- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Pillow

All dependencies are listed in `requirements.txt`.

---

## 🎯 Future Improvements

- Add transfer learning (MobileNetV2 / ResNet50 / EfficientNet)
- Increase dataset size for improved generalization
- Hyperparameter tuning (learning rate, batch size, regularization)
- Deploy model using Streamlit / Gradio / Flask

---

## 📚 Skills Demonstrated

- Deep Learning
- Convolutional Neural Networks
- Binary Classification
- Data Augmentation
- TensorFlow & Keras
- Computer Vision Fundamentals

---

## 📜 License

This project is intended for educational and portfolio purposes.

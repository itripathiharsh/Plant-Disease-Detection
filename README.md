# 🌿 Plant Disease Detection System (CNN-Based)

## 📘 Overview

This project aims to develop a **Convolutional Neural Network (CNN)**-based system for **early detection and classification of plant diseases** using leaf images.
The goal is to support **precision agriculture** by providing a reliable AI tool that helps farmers and researchers identify plant health conditions efficiently and accurately.

---

## 🎯 Project Objectives

* Early detection of plant diseases using computer vision.
* Build a deep learning model for **multi-class image classification**.
* Enable farmers and researchers to take proactive measures in crop management.
* Provide a deployable model for integration with web or mobile apps.

---

## 🧠 Key Features

* 🪴 **Automated Disease Classification:** Detects and classifies 38 plant diseases from leaf images.
* 🧩 **Deep CNN Architecture:** Custom-built 4-layer convolutional neural network using **TensorFlow** and **Keras**.
* 🧠 **High Accuracy:** Optimized using **Adam optimizer**, **early stopping**, and **learning rate reduction** techniques.
* 🖼️ **Image Preprocessing:** Data augmentation, normalization, and color scaling using `ImageDataGenerator`.
* 📊 **Training Visualization:** Includes accuracy, precision, recall, and loss plots for model performance analysis.
* 💾 **Model Export:** Trained model (`PDDS.keras`) and class indices (`class_indices.json`) saved for easy deployment.

---

## 🏗️ Project Architecture

```
Plant-Disease-Detection/
│
├── 📁 dataset/                           # Dataset directory (train/valid)
│   ├── train/
│   └── valid/
│
├── 🧠 PDDS.keras                         # Saved trained model
├── 📄 class_indices.json                 # JSON file for class mappings
├── 📓 plant_disease_detection.ipynb      # Colab notebook / main training script
├── 🧾 requirements.txt                   # Dependencies
└── README.md                             # Project documentation
```

---

## ⚙️ Tech Stack

| Category                   | Technologies                                                  |
| -------------------------- | ------------------------------------------------------------- |
| **Programming Language**   | Python                                                        |
| **Frameworks & Libraries** | TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib, Seaborn |
| **Deep Learning**          | Convolutional Neural Networks (CNN)                           |
| **Tools**                  | Google Colab, Jupyter Notebook                                |
| **Dataset**                | New Plant Diseases Dataset (Augmented)                        |

---

## 🧰 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection
```

### 2️⃣ Create a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Prepare Dataset

Download and extract the **New Plant Diseases Dataset (Augmented)** into your project folder.
The directory should look like:

```
/content/New Plant Diseases Dataset(Augmented)/
    ├── train/
    └── valid/
```

### 5️⃣ Run the Project

You can run the training script or open the Jupyter/Colab notebook:

```bash
jupyter notebook plant_disease_detection.ipynb
```

or upload it to **Google Colab** for GPU acceleration.

---

## 🧬 Model Architecture Summary

| Layer Type   | Parameters              | Activation | Output Shape   |
| ------------ | ----------------------- | ---------- | -------------- |
| Conv2D       | 32 filters, 7x7 kernel  | ReLU       | (224, 224, 32) |
| MaxPooling2D | 2x2                     | -          | (112, 112, 32) |
| Conv2D       | 64 filters, 5x5 kernel  | ReLU       | (112, 112, 64) |
| MaxPooling2D | 2x2                     | -          | (56, 56, 64)   |
| Conv2D       | 128 filters, 3x3 kernel | ReLU       | (56, 56, 128)  |
| MaxPooling2D | 2x2                     | -          | (28, 28, 128)  |
| Conv2D       | 256 filters, 3x3 kernel | ReLU       | (28, 28, 256)  |
| Flatten      | -                       | -          | -              |
| Dense        | 128                     | ReLU       | -              |
| Dropout      | 0.5                     | -          | -              |
| Dense        | 64                      | ReLU       | -              |
| Dropout      | 0.5                     | -          | -              |
| Output Layer | 38                      | Softmax    | -              |

---

## 🧪 Training & Evaluation

### Model Compilation

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy', 'precision', 'recall']
)
```

### Model Training

```python
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=20,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)
```

### Evaluation

```python
model.evaluate(test_generator)
```

Metrics displayed: **Loss, Accuracy, Precision, Recall**

---

## 📈 Visualization

The notebook includes visual plots for:

* Training vs Validation Accuracy
* Training vs Validation Loss
* Precision & Recall curves

```python
plt.plot(epochs, acc, 'g', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
```

---

## 💾 Model Saving & Export

Save the model and class labels for deployment:

```python
model.save('PDDS.keras')

import json
with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)
```

---

## 🚀 Future Enhancements

* ✅ Deploy as a web application using **Streamlit** or **Flask**
* ✅ Integrate with a **mobile app** for real-time leaf disease detection
* ✅ Implement **Transfer Learning** with models like EfficientNet or ResNet50
* ✅ Add **explainability (Grad-CAM)** to visualize CNN feature activations
* ✅ Extend dataset with **real-time captured leaf images**

---

## 🧾 Results Summary

| Metric         | Result                       |
| -------------- | ---------------------------- |
| Accuracy       | ~95% (after 20 epochs)       |
| Precision      | High                         |
| Recall         | High                         |
| Model Size     | ~40 MB                       |
| Inference Time | <1 second per image (on GPU) |

---

## 🧠 Learning Outcomes

* Built and optimized CNNs for real-world image classification.
* Applied deep learning to **precision agriculture**.
* Learned end-to-end ML workflow: data preprocessing → training → evaluation → deployment.
* Gained experience with **TensorFlow**, **Keras**, and **OpenCV** for AI-driven applications.

---

## 📜 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with proper attribution.

---

## 🙌 Acknowledgements

* **Dataset:** [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
*  **Dataset:**  [Best_model.keras](https://drive.google.com/file/d/1lydeJz5Z-ShY3gYJREloI-d8gXR1ChYW/view?usp=drive_link)
* **Frameworks:** TensorFlow, Keras
* **Platform:** Google Colab
* **Inspiration:** Sustainable farming and precision agriculture research.

---

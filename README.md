# Project 9: CNN Development for Image Classification on CIFAR-10

## 📋 Project Overview

This project designs, trains, and evaluates a **Convolutional Neural Network (CNN)** for
multi-class image classification on the **CIFAR-10** dataset. The implementation applies
data augmentation to improve generalization and includes feature map visualization to
interpret the CNN's internal learning process.

> **Final Test Accuracy: 86.23%** with a minimal overfitting gap of 0.15%

---

## 🎯 Objectives

1. **Dataset Preparation** — Load and preprocess CIFAR-10 (normalize, one-hot encode)
2. **CNN Architecture** — Design a 3-block CNN with Conv2D, BatchNorm, Pooling, and Dropout
3. **Data Augmentation** — Apply on-the-fly augmentation using Keras ImageDataGenerator
4. **Training & Evaluation** — Train for 50 epochs; monitor accuracy and loss curves
5. **Feature Map Visualization** — Extract and visualize intermediate layer activations

---

## 📊 Dataset

- **Source:** CIFAR-10 (built-in via `tensorflow.keras.datasets`)
- **Classes:** 10 — Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **Image Size:** 32×32 RGB (3 channels)

| Split | Size |
|---|---|
| Training | 50,000 images |
| Test | 10,000 images |

### Preprocessing Steps
- Normalized pixel values from `[0, 255]` → `[0, 1]`
- One-hot encoded target labels for 10-class classification

---

## 🏗️ Model Architecture

Three convolutional blocks with progressively increasing filter depth, followed by
dense classification layers:
Input (32x32x3)
↓
[Conv2D(32) + BatchNorm + ReLU] × 2 → MaxPool(2x2) → Dropout(0.2)
↓
[Conv2D(64) + BatchNorm + ReLU] × 2 → MaxPool(2x2) → Dropout(0.3)
↓
[Conv2D(128) + BatchNorm + ReLU] × 2 → MaxPool(2x2) → Dropout(0.4)
↓
Flatten → Dense(256) + BatchNorm + Dropout(0.5) → Dense(10, Softmax)

| Parameter | Value |
|---|---|
| Total Parameters | 816,938 (~3.12 MB) |
| Trainable Parameters | 815,530 (~3.11 MB) |
| Non-trainable Parameters | 1,408 (5.50 KB) |

---

## 🔄 Data Augmentation

Applied on-the-fly to training images using `ImageDataGenerator`:

| Technique | Configuration |
|---|---|
| Rotation Range | ±15 degrees |
| Width / Height Shift | ±10% |
| Horizontal Flip | Enabled |
| Zoom Range | ±10% |
| Shear Range | ±10% |

**Benefit:** Artificially expands the effective training set, reducing overfitting
without collecting additional data.

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (lr = 0.001) |
| Loss Function | Categorical Crossentropy |
| Batch Size | 64 |
| Epochs | 50 |

---

## 📈 Results

### Overall Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | **86.23%** |
| Test Loss | 0.4277 |
| Final Training Accuracy | 86.08% |
| Final Validation Accuracy | 86.23% |
| Overfitting Gap | 0.0015 (Minimal) |

### Per-Class Accuracy

| Class | Accuracy |
|---|---|
| Airplane | 85.60% |
| Automobile | 94.90% |
| Bird | 76.80% |
| Cat | 63.00% |
| Deer | 87.40% |
| Dog | 76.20% |
| **Frog** | **96.00%** |
| **Horse** | **95.90%** |
| Ship | 89.50% |
| **Truck** | **97.00%** |

- **Best:** Truck (97%), Frog (96%), Horse (95.9%)
- **Worst:** Cat (63%), Bird (76.8%), Dog (76.2%)
- Cat and Dog are frequently confused due to similar visual textures

---

## 📁 Generated Files

| File | Description |
|---|---|
| `cifar10_samples.png` | Sample images from the dataset |
| `augmentation_comparison.png` | Original vs augmented image comparison |
| `training_history.png` | Training/validation accuracy and loss curves |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `per_class_accuracy.png` | Bar chart of per-class accuracy |
| `feature_maps.png` | Feature maps from convolutional layers |
| `deep_feature_maps.png` | Average activation heatmaps per layer |
| `downsampling_progression.png` | Spatial resolution progression visualization |
| `sample_predictions.png` | Test predictions (correct/incorrect marked) |
| `augmentation_impact.png` | With vs without augmentation comparison |

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| Python 3.x | Core language |
| TensorFlow / Keras | CNN construction, augmentation, training |
| NumPy | Numerical operations |
| Matplotlib | Training curves and feature map visualization |
| Scikit-learn | Confusion matrix and evaluation metrics |
| Seaborn | Heatmap styling |

---

## 🔧 How to Run

1. **Clone the repository**
```bash
   git clone <repository-url>
   cd project-9-cnn-cifar10
```

2. **Install dependencies**
```bash
   pip install tensorflow numpy matplotlib scikit-learn seaborn
```

3. **Run the notebook**
```bash
   jupyter notebook ML_Task_9_Tayyabah_Rehman.ipynb
```

   The notebook will automatically download CIFAR-10, train the CNN, generate all
   visualizations, and save outputs to the working directory.

---

## 💡 Key Insights

- **Data Augmentation Impact:** Reduced the overfitting gap to just 0.15% between
  training and validation accuracy
- **Spatial Progression:** Resolution decreases 32 → 16 → 8 pixels across pooling
  layers while filter depth increases 32 → 64 → 128
- **Feature Hierarchy:** Early layers detect low-level edges and textures; deeper
  layers capture object-level parts and shapes
- **Class Difficulty:** Cat and Dog are the hardest classes due to overlapping visual
  features and fine-grained texture differences

---

## 📝 Conclusion

The CNN with data augmentation achieves **86.23% accuracy** on CIFAR-10 with minimal
overfitting. The architecture effectively learns hierarchical features, while augmentation
significantly improves generalization. Future improvements could include residual
connections (ResNet-style), learning rate scheduling, more aggressive augmentation,
or ensemble methods for higher accuracy.

---

## 👩‍💻 Author

**Tayyabah Rehman**
📅 June 2026

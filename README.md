# **Low-Power Animal Audio Classifier for Biodiversity Monitoring**

*A TinyML Project for Real-Time Species Detection on Edge Devices*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)](https://www.tensorflow.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12Spuza3D_Qb27iK9Ls4D4jU59zvYmocv?usp=sharing)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains a complete end-to-end workflow for building a **deep learning model** that classifies animals based on their **audio calls**, and deploying it on **low-power microcontrollers** such as ESP32 and Raspberry Pi Pico using **TensorFlow Lite**.

The entire workflow—from preprocessing, training, evaluation, and TFLite quantization—was developed by **Amal Madhu**.

---

## **Key Features**

- **99.23% Test Accuracy** on 10 species classification
- **Real-time audio processing** with 3-second audio segments
- **Mel Spectrogram feature extraction** for robust audio representation
- **Dynamic range quantization (int8)** reducing model size by 72.6%
- **Ultra-lightweight**: 0.22 MB quantized model
- **Fast inference**: ~15 ms per prediction
- **Comprehensive species database** with detailed information and images
- **Interactive visualizations** including confusion matrices and performance charts
- **Complete deployment pipeline** for ESP32, Raspberry Pi Pico, and similar microcontrollers
- **Detailed species cards** with conservation status and behavioral information

---

## **Model Performance**

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 99.23% |
| **Test Loss** | 0.0898 |
| **Training Accuracy** | 98.92% |
| **Validation Accuracy** | 97.67% |
| **Inference Time** | 14.9 ms |
| **Model Size (Original)** | 0.79 MB |
| **Model Size (Quantized)** | 0.22 MB |
| **Size Reduction** | 72.6% |

### Per-Species Performance
All species achieve >95% accuracy, with most achieving 100% accuracy.

**Confusion Matrix Analysis:**
- Only 2 misclassifications out of 259 test samples
- 1 Cat misclassified as Chicken
- 1 Cow misclassified as Cat

---

## **Dataset Structure**

Dataset source: [YashNita/Animal-Sound-Dataset](https://github.com/YashNita/Animal-Sound-Dataset.git)

```
dataset/
├── Bird/           # 100 samples
├── Cat/            # 100 samples
├── Chicken/        # 30 samples
├── Cow/            # 75 samples
├── Dog/            # 100 samples
├── Donkey/         # 25 samples
├── Frog/           # 35 samples
├── Lion/           # 45 samples
├── Monkey/         # 25 samples
└── Sheep/          # 39 samples
```

**Total**: 574 audio samples (imbalanced dataset)

**After Data Augmentation**: 1,722 samples
- **Training**: 1,205 samples (70%)
- **Validation**: 258 samples (15%)
- **Test**: 259 samples (15%)

### Naming Note
Some datasets may have class folders in non-English names (e.g., Turkish). If your environment fails to recognize them, rename folders to English equivalents (e.g., `Kedi → Cat`, `Aslan → Lion`) to avoid path and Unicode issues.

### Data Augmentation Techniques:
1. **Noise Addition**: Gaussian noise (factor 0.005)
2. **Time Shifting**: Random temporal shifts (up to 20%)

---

## **Model Architecture**

**Input Shape**: `(128, 130, 1)` - Mel Spectrogram  
**Total Parameters**: 208,362
**Trainable Parameters**: 207,210
**Model Type**: Lightweight 2D CNN with Batch Normalization

**Architecture:**
```
Input (128, 130, 1)
    ↓
[Block 1]
Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.3)
    ↓
[Block 2]
Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.4)
    ↓
[Block 3]
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.5)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256) → Dropout(0.5)
    ↓
Dense(128) → Dropout(0.5)
    ↓
Dense(10, Softmax)
```

**Key Features:**
- Batch normalization for training stability
- Dropout for regularization (0.3-0.5)
- Class weights for imbalanced dataset
- Adam optimizer with learning rate scheduling
- Early stopping (patience: 15)
- ReduceLROnPlateau (factor: 0.5, patience: 7)

---

## **How to Run the Notebook**

### **Quick Start - Google Colab (Recommended)**

<a href="https://colab.research.google.com/drive/12Spuza3D_Qb27iK9Ls4D4jU59zvYmocv?usp=sharing" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="40"/>
</a>

**Click the badge above to run the complete project in Google Colab with free GPU!**

**Steps:**
1. Click the Colab badge
2. Sign in with your Google account
3. Runtime → Change runtime type → GPU (T4)
4. Run all cells (Runtime → Run all)
5. Download generated models from `saved_files/`

---

### **Option 2: Local Jupyter Setup**

1. **Clone the repository**
```bash
git clone https://github.com/AbyssDrn/Animal-Sound-Classifier.git
cd Animal-Sound-Classifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download dataset**
```bash
git clone https://github.com/YashNita/Animal-Sound-Dataset.git dataset/
```

4. **Launch notebook**
```bash
jupyter notebook SiMoni.ipynb
```

5. **Update paths and run all cells**

---

## **TensorFlow Lite Optimization**

The model undergoes dynamic range quantization:

**Before Quantization:**
- Format: Keras (float32)
- Size: 0.79 MB
- Accuracy: 99.23%

**After Quantization:**
- Format: TFLite (int8)
- Size: 0.22 MB (72.6% reduction)
- Accuracy: 99.23% (no degradation)
- Inference: 14.9 ms

This makes the model suitable for:
- ESP32 microcontrollers
- Raspberry Pi Pico
- Arduino Nano 33 BLE Sense
- STM32 boards
- Other ARM Cortex-M devices

---

## 📦 **Generated Outputs**

After running the notebook, these files are created in `saved_files/`:

```
saved_files/
├── animal_classifier_full.keras          # Full Keras model (0.79 MB)
├── animal_classifier_quantized.tflite    # Quantized TFLite (0.22 MB)
├── best_model.keras                      # Best checkpoint during training
├── label_classes.npy                     # Label encoder classes
├── model_metadata.json                   # Model configuration & stats
├── species_database.json                 # Species information database
└── training_history.csv                  # Training metrics log
```

---

## 📧 **Contact & Support**

**Amal Madhu**  
GitHub: [@AbyssDrn](https://github.com/AbyssDrn)  
Email: [amalmadhu04022001@gmail.com]  

**Project Repository**: [https://github.com/AbyssDrn/Animal-Sound-Classifier](https://github.com/AbyssDrn/Animal-Sound-Classifier)  
**Google Colab Notebook**: [Open in Colab](https://colab.research.google.com/drive/12Spuza3D_Qb27iK9Ls4D4jU59zvYmocv?usp=sharing)

---

**Made with ❤️ for wildlife conservation and TinyML enthusiasts**

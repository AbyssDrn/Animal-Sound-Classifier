# 🐾 **Low-Power Animal Audio Classifier for Biodiversity Monitoring**

*A TinyML Project for Real-Time Species Detection on Edge Devices*

This repository contains a complete end-to-end workflow for building a **deep learning model** that classifies animals based on their **audio calls**, and deploying it on **low-power microcontrollers** such as the Raspberry Pi Pico using **TensorFlow Lite Micro**.

The entire workflow—from preprocessing, training, evaluation, and TFLite quantization—was developed by **Amal Madhu**.

---

# 📌 **Table of Contents**

* [Project Overview](#project-overview)
* [Supported Species](#supported-species)
* [Machine Learning Pipeline](#machine-learning-pipeline)
* [Model Architecture](#model-architecture)
* [Model Performance](#model-performance)
* [TinyML Deployment (Raspberry Pi Pico)](#tinyml-deployment-raspberry-pi-pico)
* [Repository Structure](#repository-structure)
* [How to Run the Notebook](#how-to-run-the-notebook)
* [How to Deploy on Raspberry Pi Pico](#how-to-deploy-on-raspberry-pi-pico)
* [Example MicroPython Inference Script](#example-micropython-inference-script)
* [Future Enhancements](#future-enhancements)
* [Author](#author)

---

# 🐦 **Project Overview**

This project implements a **2D Convolutional Neural Network (CNN)** trained on **Mel Spectrograms** of animal audio recordings.

It is designed for:

✔ Offline wildlife monitoring
✔ Low-power embedded systems
✔ Nature conservation projects
✔ Real-time species identification

Training is performed in **Google Colab** using TensorFlow/Keras.
The final model is converted to a **quantized TensorFlow Lite model** suitable for TinyML deployment.

---

# 🐾 **Supported Species**

The classifier currently recognizes **10 species**:

| ID | Species |
| -- | ------- |
| 1  | Bird    |
| 2  | Cat     |
| 3  | Chicken |
| 4  | Cow     |
| 5  | Dog     |
| 6  | Donkey  |
| 7  | Frog    |
| 8  | Lion    |
| 9  | Monkey  |
| 10 | Sheep   |

You can expand this by adding your own audio dataset.

---

# 🧠 **Machine Learning Pipeline**

The file **SiMoni.ipynb** contains the entire workflow:

### **1️⃣ Environment Setup**

Installs required libraries such as:

* TensorFlow / Keras
* Librosa
* Scikit-learn
* Matplotlib
* NumPy

---

### **2️⃣ Data Loading**

Dataset expected structure:

```
dataset/
   ├── Bird/
   ├── Cat/
   ├── Chicken/
   ├── …
```

Each folder contains `.wav` or `.mp3` audio files.

---

### **3️⃣ Preprocessing & Feature Extraction**

✔ Audio loaded at **22,050 Hz**
✔ Trimmed/padded to **3 seconds**
✔ Converted into **Mel Spectrograms** (2D image-like input)

This is the model’s primary input.

---

### **4️⃣ CNN Architecture Overview**

A typical configuration:

* Conv2D → BatchNorm → ReLU
* MaxPooling2D
* Conv2D → BatchNorm → ReLU
* Dropout
* Flatten
* Dense (Softmax for classification)

---

### **5️⃣ Training**

* Achieves up to **~97.7% validation accuracy**
* Includes accuracy/loss learning curves

---

### **6️⃣ Evaluation**

You get:

* Accuracy plot
* Loss plot
* Confusion matrix
* Classification report (precision, recall, f1-score)

---

### **7️⃣ TensorFlow Lite Quantization**

Model is converted using:

```
tf.float16 quantization
```

This results in:

✔ Smaller file size
✔ Suitable for Raspberry Pi Pico / ESP32 / Arduino Nano BLE Sense
✔ Faster inference

---

# 🧩 **Model Performance**

Your notebook automatically generates:

* **Confusion Matrix**
* **Accuracy & Loss Graphs**
* **Precision/Recall/F1 Report**

These files are saved in the repo after training.

---

# 🐍 **TinyML Deployment (Raspberry Pi Pico)**

> **Important:** This project uses **audio**, not camera vision.
> You will need a **microphone module**, not a camera.

### ✔ Recommended Hardware

| Component                      | Purpose                     |
| ------------------------------ | --------------------------- |
| **Raspberry Pi Pico / Pico W** | Main MCU                    |
| **I2S Microphone** (INMP441)   | Best for high-quality audio |
| **PDM Microphone** (MSM261D)   | Ultra-low-power option      |
| **OLED Display (Optional)**    | Show detected species       |
| **SD Card Module (Optional)**  | Store audio / model files   |

---

# ⚙️ **Deployment Workflow**

### **1. Flash MicroPython to Pico**

Download UF2 → drag into Pico storage.

---

### **2. Install TensorFlow Lite Micro**

Use `tflm` MicroPython library or custom build.

---

### **3. Upload Deployment Files**

Upload these generated files via Thonny:

```
animal_classifier_quantized.tflite
model_labels.txt
species_database.json
main.py
```

---

### **4. Run inference on live microphone audio**

MicroPython script provided below.

---

# 🧪 **Example MicroPython Inference Script**

```python
import tflite_micro as tflm
import audio_processor
import time

# Load labels
with open('model_labels.txt', 'r') as f:
    labels = [line.strip() for line in f]

# Load model
model_data = open('animal_classifier_quantized.tflite', 'rb').read()
interpreter = tflm.runtime.Interpreter(model_data)

print("--- Starting Animal Audio Monitor ---")

while True:
    # Record 3 seconds of audio
    audio_buffer = audio_processor.record_audio(duration=3, sample_rate=22050)

    # Convert audio to Mel Spectrogram
    spectrogram = audio_processor.compute_mel_spectrogram(audio_buffer)

    # Set input tensor
    interpreter.set_input(spectrogram, 0)

    # Run inference
    interpreter.invoke()

    # Get output probabilities
    output = interpreter.get_output(0)
    predicted_index = output.argmax()
    confidence = output[predicted_index]
    predicted_species = labels[predicted_index]

    if confidence > 0.75:
        print(f"Detected: {predicted_species} ({confidence*100:.1f}% confidence)")

    time.sleep(1)
```

---

# 📁 **Repository Structure**

```
📦 animal-audio-classifier
│
├── SiMoni.ipynb                   # Main Colab notebook
├── saved_files.zip
|     └── animal_classifier_full.keras
|     └── animal_classifier_quantizedtflite
|     └── best_model.keras
|     └── label_classes.npy
|     └── model_metadata.json
|     └── species_database.json
|     └── trainig_history.csv
│
├── species_data/
│     └── images/                  # Downloaded species images
│
│
└── README.md                      # Project documentation
```

---

# 🖥️ **How to Run the Notebook**

### **1. Clone the repository**

```bash
git clone [https://github.com/your-username/animal-audio-classifier.git](https://github.com/AbyssDrn/Animal-Sound-Classifier.git)
cd animal-audio-classifier
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

*(Create `requirements.txt` from your Colab pip installs.)*

### **3. Open in Google Colab**

* Upload **SiMoni.ipynb**
* Upload your audio dataset
* Update `DATA_PATH`
* Run all cells

### **4. Export the files**

Download:

* `.tflite`
* `.json`
* `.txt`

---

# 🔮 **Future Enhancements**

✔ Add more species
✔ Implement real-time streaming processing
✔ Build a mobile app interface
✔ Support environmental sound classification
✔ Add GPS tagging + SD card storage
✔ Build a solar-powered edge device

---

# 👤 **Author**

**Amal Madhu**
Developer • AI & TinyML Researcher
GitHub: *AbyssDrn*

---


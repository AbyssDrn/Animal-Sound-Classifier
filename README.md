Low-Power Animal Audio Classifier for Biodiversity Monitoring
This project details a complete pipeline for creating a deep learning model that classifies animal species from their audio calls. The model is optimized for low-power edge devices, making it ideal for offline biodiversity monitoring in remote locations.

This repository is based on the work by Amal Madhu.

1. Project Overview
The core of this project is a 2D Convolutional Neural Network (CNN) trained on Mel Spectrograms generated from 3-second audio clips. The model is trained in a Google Colab environment using TensorFlow/Keras and then converted to a quantized TensorFlow Lite (TFLite) file, which is small and efficient enough to run on microcontrollers.

The final model can identify the following 10 species:

Bird

Cat

Chicken

Cow

Dog

Donkey

Frog

Lion

Monkey

Sheep

2. The Machine Learning Pipeline
The SiMoni.ipynb notebook details the entire end-to-end process:

Environment Setup: Installs necessary libraries like TensorFlow, Librosa, and Scikit-learn.

Data Loading: Loads a dataset of audio files (e.g., .wav, .mp3) from Google Drive. The script expects the data to be organized in folders by species name.

Preprocessing & Feature Extraction:

Audio files are loaded, resampled to 22050Hz, and truncated or padded to 3 seconds.

The raw audio waveform is converted into a Mel Spectrogram, which is a 2D image-like representation of the audio's frequency content over time. This is the input to our 2D CNN.

Model Architecture: A 2D CNN is built using tensorflow.keras.models.Sequential. The architecture typically includes:

Conv2D layers (for feature detection).

BatchNormalization (to stabilize training).

MaxPooling2D (to downsample).

Dropout (to prevent overfitting).

Flatten and Dense layers (for classification).

Model Training: The model is trained on the labeled spectrograms, achieving a high validation accuracy (e.g., ~97.7%).

Model Evaluation: Performance is analyzed using:

Accuracy and loss plots over epochs.

A detailed classification report showing precision, recall, and f1-score for each species.

A confusion matrix to visualize misclassifications.

TFLite Conversion (Quantization):

The trained Keras model is converted into a TensorFlow Lite (.tflite) model.

Post-training quantization (using tf.float16) is applied to dramatically reduce the model size and make it suitable for "TinyML" applications on microcontrollers.

3. Deployment on a Raspberry Pi Pico (TinyML)
You asked specifically about deploying this to a Raspberry Pi Pico. This is an excellent "TinyML" use case.

Important Note: Your project is audio-based, not visual. You would use a microphone module, not a camera module.

Hardware Required:
Raspberry Pi Pico W: The "W" model with Wi-Fi is useful, but a standard Pico will work for offline inference.

Microphone Module: You need a digital microphone.

I2S Microphone: (e.g., INMP441) Recommended for high-quality audio.

PDM Microphone: (e.g., MSM261D) A simpler digital option.

(Optional) A small display (OLED or TFT) to show the detected species.

(Optional) SD card module for storing the model and species database.

Deployment Workflow:
Flash MicroPython: Install the MicroPython firmware on your Pico.

Install TFLite Micro: Use the official TensorFlow Lite Micro library for MicroPython.

Generate Features on-device: This is the most challenging part. The C/C++ SDK is often better for this, but for MicroPython, you would need:

A library to capture audio from your I2S/PDM microphone.

A fast audio processing library (like uNumpy or a custom C module) to compute the Mel Spectrogram on the Pico itself. This must generate the exact same spectrogram shape as your Colab notebook.

Load Files onto Pico:

animal_classifier_quantized.tflite: The model file.

model_labels.txt: The list of species names.

main.py: Your script to run the inference.

Inference Script (main.py):

Python

# This is a simplified pseudo-code example for MicroPython
import tflite_micro as tflm
import audio_processor # Your custom library
import time

# 1. Load labels
with open('model_labels.txt', 'r') as f:
    labels = [line.strip() for line in f]

# 2. Load the TFLite model
model_data = open('animal_classifier_quantized.tflite', 'rb').read()
interpreter = tflm.runtime.Interpreter(model_data)

print("--- Starting Animal Audio Monitor ---")

while True:
    # 3. Record 3 seconds of audio
    audio_buffer = audio_processor.record_audio(duration=3, sample_rate=22050)

    # 4. Generate the spectrogram (This is the hard part)
    # Input shape must match model's expected input
    spectrogram = audio_processor.compute_mel_spectrogram(audio_buffer)

    # 5. Set the input tensor
    interpreter.set_input(spectrogram, 0)

    # 6. Run inference
    interpreter.invoke()

    # 7. Get the output tensor (probabilities)
    output = interpreter.get_output(0)

    # 8. Find the animal with the highest probability
    predicted_index = output.argmax()
    confidence = output[predicted_index]
    predicted_species = labels[predicted_index]

    if confidence > 0.75: # Confidence threshold
        print(f"Detected: {predicted_species} (Confidence: {confidence*100:.1f}%)")

    time.sleep(1)
4. Files Created by this Project
The Colab notebook generates several key files that you should organize in your repository:

animal_classifier_quantized.tflite: The optimized, deployable ML model.

model_labels.txt: A simple text file listing the 10 species names, one per line.

species_database.json: A rich JSON file containing detailed information for each species (fun facts, conservation status, etc.) used by the demo UI.

species_data/images/: A folder containing downloaded images (e.g., Dog.jpg, Cat.jpg) for the visual interface.

model_performance.png: (Or similar) Plots for accuracy, loss, and the confusion matrix.

5. How to Use this Repository
Clone the Repository:

Bash

git clone https://github.com/your-username/animal-audio-classifier.git
cd animal-audio-classifier
Set up Environment:

Bash

pip install -r requirements.txt
(You will need to create a requirements.txt file from the notebook's !pip install commands).

Run in Colab:

Upload the SiMoni.ipynb notebook to Google Colab.

Upload your audio dataset to Google Drive and update the DATA_PATH variable in Section 2 of the notebook.

Run all cells to train the model and generate the deployment files.

Download the .tflite, .txt, and .json files from the Colab environment.

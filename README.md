# Wavefront-reconstruction-Using-Deep-Leaning-For-Holography-
Deep Holography CNN

This project implements a simple Convolutional Neural Network (CNN) for reconstructing object fields from synthetic holograms. It is designed for applications in computational optics and digital holography, and is based on training paired data of holograms and their corresponding reconstructed images.

Project Overview

Title: Deep Holography: CNN-Based Reconstruction of Object Fields from Synthetic Holograms

Author: Alhassan Kpahambang Fuseini

Tools Used: Python, TensorFlow/Keras, NumPy, Matplotlib, PIL

Dataset: Synthetic hologram dataset (64x64 grayscale images)

Objective: Train a CNN to map holograms to object fields in a supervised learning setup.


Dataset

The dataset includes:

Holograms: 64x64 grayscale PNG files

Object fields (labels): 64x64 grayscale PNG files


Paths:

X_train_path = "../input/synthetic-holograms64x64x1/X_train_HOLO/Labels/"
y_train_path = "../input/synthetic-holograms64x64x1/y_train_obj/objects/"

Model Architecture

Input (64x64x1)
 └── Conv2D (20 filters, 3x3, stride=2, ReLU)
     └── Conv2DTranspose (1 filter, 3x3, stride=2, ReLU)

This simple encoder-decoder structure aims to reduce dimensionality and reconstruct the spatial features of the object field.

Training Configuration

Optimizer: Adam

Loss: Mean Squared Error (MSE)

Epochs: 200

Batch Size: 16


Key Results

The model was able to reduce training loss to ~25.8 MSE after 200 epochs.

Generated reconstructions visually approximate target object fields.


Sample Visuals

Visualization of input holograms and labels

Real-time loss monitoring

Model architecture plot


File Structure

project-directory/
|── model.png         # Visual diagram of the model
|── recon_output/     # Output reconstructions (if saved)
|── train_visuals/   # Plots of input/output during training
|── model.py         # Python script for loading data and training
|── README.md        # Project overview

Installation & Running

# Clone this repository
$ git clone https://github.com/yourusername/deep-holography-cnn.git
$ cd deep-holography-cnn

# Install dependencies
$ pip install -r requirements.txt

# Run training
$ python model.py

Applications

Wavefront reconstruction

Optical imaging systems

Educational tool for machine learning in optics

License: MIT

# Pneumonia Classification CNN

Pneumonia detection from chest X-ray images using PyTorch and a Convolutional Neural Network (ResNet-18).

## Overview
This project provides an end-to-end machine learning pipeline for medical image classification. It trains a ResNet-18 model to classify chest X-rays as either 'Normal' or 'Pneumonia' and deploys the trained model via a Flask web application. The deployment includes support for standard image formats as well as medical DICOM files, and features Grad-CAM visualization for explainable AI.

## Google Colab Notebooks
You can view and run the code directly in Google Colab using the links below:

* **[Training Notebook](https://colab.research.google.com/drive/1Ds3kjw_lSII2yCQdrgwX6s2sFG5B4e3-?usp=sharing)**: Handles data downloading, image preprocessing, and the PyTorch training loop.
* **[Deployment Notebook](https://colab.research.google.com/drive/1Gl6SR1YOxrb0P0tRVLZflaolQLmC_jxJ?usp=sharing)**: Runs a Flask web server with Cloudflare tunneling, processes uploaded `.dcm` or `.jpg` files, and generates predictions with Grad-CAM heatmaps.

## Features
* **Model Training:** Automated training pipeline using PyTorch and Torchvision.
* **Web Interface:** Flask-based web application for real-time predictions.
* **DICOM Support:** Specifically handles raw medical imaging data using `pydicom`.
* **Explainable AI:** Includes Grad-CAM visualizations to highlight the areas of the lung the model focuses on to make its prediction.

## App Output
Below is an example of the web application interface and prediction output, including the Grad-CAM visualization:

<img width="453" height="746" alt="App Output showing prediction and Grad-CAM" src="https://github.com/user-attachments/assets/5f31f48a-2d19-4177-9b3a-b9137007e099" />

## Requirements
To run the deployment notebook locally or in Colab, the following dependencies are required:
* PyTorch / Torchvision
* Flask
* pydicom
* OpenCV (cv2)
* Pillow
* Cloudflared (for Colab deployment tunneling)

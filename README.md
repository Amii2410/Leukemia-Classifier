# 🧬 Leukemia Classification using Deep Learning

This project focuses on classifying leukemia from microscopic blood smear images using deep learning. It uses a pretrained MobileNetV2 model with transfer learning to accurately identify different types of leukemia.

## 📌 Overview
The model classifies images into:
- Acute Lymphoblastic Leukemia (ALL)
- Acute Myeloid Leukemia (AML)
- Chronic Lymphocytic Leukemia (CLL)
- Chronic Myeloid Leukemia (CML)
- Healthy

## ⚙️ Approach
- Transfer Learning using MobileNetV2 (pretrained on ImageNet)  
- Image preprocessing and normalization  
- Classification using a Softmax output layer  

## 🚀 Features
- High accuracy (~96–98%)  
- Fast and efficient model  
- Streamlit-based web interface for easy testing  

## 🛠️ Tech Stack
- Python  
- TensorFlow / Keras  
- Streamlit  
- NumPy  

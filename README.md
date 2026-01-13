# Smartphone Price Estimation System using Machine Learning

Accurately estimating smartphone prices is challenging due to rapidly changing features, brands, and market trends. This project builds a Machine Learning–based system to predict the price range of smartphones based on their technical specifications such as RAM, storage, battery, camera, and processor.

The model is trained using a publicly available Kaggle dataset containing real-world smartphone features. A supervised ML model learns patterns between hardware specifications and price categories, enabling fast and consistent price estimation.

## Objectives
- Predict smartphone price range based on specifications  
- Assist users and sellers in fair price estimation  
- Build a scalable backend ML pipeline  

## Dataset
- Source: Kaggle – *Mobile Price Classification*  
- Features: RAM, ROM, Battery, Camera, CPU, Screen Size, etc.  
- Target: Price Range (Low, Medium, High, Premium)  
- Preprocessing:
  - Handling missing values  
  - Feature scaling  
  - Label encoding  

## My Role
- Implemented backend ML pipeline  
- Handled data preprocessing and feature engineering  
- Built model training and evaluation logic  
- Integrated prediction flow for input features  
- Ensured efficient and reliable backend execution  

## Model Architecture
- Supervised ML model (Random Forest / Logistic Regression / SVM)
- Pipeline:
  - Feature normalization  
  - Model training  
  - Prediction engine  

## Workflow
1. Dataset collection from Kaggle  
2. Data cleaning & preprocessing  
3. Feature engineering  
4. Train-test split  
5. Model training  
6. Evaluation  
7. Backend prediction integration  

## Results
The model predicts smartphone price ranges with high accuracy, demonstrating effective learning from hardware specifications and enabling real-time backend price estimation.

## Tech Stack
- Python  
- Scikit-learn  
- Pandas, NumPy  
- Flask (for backend integration)  
- Matplotlib  

## How to Run
```bash
pip install pandas numpy scikit-learn flask matplotlib
python app.py

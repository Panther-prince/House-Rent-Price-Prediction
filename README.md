# House Rent Price Prediction

A machine learning project to predict house rent prices based on features like BHK, size, location, and amenities. Includes data preprocessing, model training, and a user-friendly **Streamlit web application** for real-time predictions.

---

## 🏠 Project Overview

- Predicts estimated monthly rent for houses based on various features.
- Utilizes regression models like **Gradient Boosting Regressor** (optional support for XGBoost, Random Forest, Linear Regression, etc.).
- Interactive UI created with **Streamlit**.
- Useful for landlords, tenants, or real estate enthusiasts to estimate rent prices accurately.

---

## 🔧 Features

- **Input Features:**  
  - BHK (number of bedrooms)  
  - Size (square feet)  
  - Location  
  - Additional amenities (if available)  
- **Prediction Models:** Gradient Boosting, XGBoost (optional)  
- **Interactive UI:** Streamlit app for easy input and instant predictions  
- **Data Preprocessing:** Handles missing values, categorical encoding, and scaling  

---

## 📊 Dataset

- Contains house details and their rent prices.  
- Features include `BHK`, `Size`, `Location`, and `Rent`.  
- Cleaned and preprocessed for model training.

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/House-Rent-Price-Prediction.git
   cd House-Rent-Price-Prediction

2. Create a virtual environment:

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


3. Install dependencies:

pip install -r requirements.txt

# 🏠 House Price Prediction

An **end-to-end Machine Learning project** to predict house prices using Python and scikit-learn.  
This repository demonstrates the complete ML workflow — from data loading and preprocessing to model training and evaluation.

---

## 📌 Table of Contents

- [About](#about)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [ML Workflow](#ml-workflow)
- [Model Training](#model-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## 📖 About

This project focuses on predicting house prices using supervised machine learning techniques.  
It is designed as a **portfolio-ready ML project**, following best practices such as:

- Clear feature separation (categorical vs numerical)
- Proper train-test split
- Scikit-learn pipelines
- Standard evaluation metrics

---

## 📁 Dataset

The dataset used in this project:

```
Housing.csv
```

It contains structured housing data with multiple features and a target variable representing house prices.

---

## 🗂 Project Structure

```
HousePricePrediction/
│
├── Housing.csv
├── HousePricePrediction_Source_code.ipynb
├── HousePricePrediction_Source_code.py
├── README.md
└── requirements.txt
```

---

## 🧰 Tech Stack

- Python 3.x
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 🛠 Setup & Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/damodaranselvaraj/HousePricePrediction.git
cd HousePricePrediction
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

- **Windows**
```bash
venv\Scripts\activate
```

- **macOS / Linux**
```bash
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Option 1: Run using Jupyter Notebook

```bash
jupyter notebook
```

Open:

```
HousePricePrediction_Source_code.ipynb
```

### Option 2: Run as Python script

```bash
python HousePricePrediction_Source_code.py
```

Ensure `Housing.csv` is present in the same directory.

---

## 🔁 ML Workflow

1. Load dataset
2. Explore and understand data
3. Handle missing values
4. Separate numerical and categorical features
5. Encode categorical variables
6. Scale numerical features
7. Split data into training and testing sets
8. Train regression model
9. Evaluate model performance

---

## 🧠 Model Training

The model training process includes:

- Feature preprocessing
- Regression model fitting (e.g., Linear Regression)
- Prediction on unseen test data

Example:

```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## 📊 Evaluation Metrics

The model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Example:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R2:", r2_score(y_test, y_pred))
```

---

## 📈 Results

Sample output (example):

```
Model Used: Linear Regression
R² Score: 0.78
RMSE: 150000
```

*(Actual results may vary depending on preprocessing and feature selection.)*

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

---

⭐ If you found this project helpful, consider giving it a star!


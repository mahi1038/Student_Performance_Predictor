# Student Performance Predictor

A machine learning web application that predicts the math score of students based on their reading and writing scores, and various categorical attributes. This project is modular, with a proper training and prediction pipeline, and includes a simple Flask-based frontend interface.

---

## 📌 Features

- End-to-end ML pipeline using `scikit-learn`
- Feature engineering and transformation for both categorical and numerical features
- Model training and selection from multiple regressors
- Simple UI for prediction using Flask

---

## 🧠 Model Details

The following regressors are evaluated, and the best-performing one is selected based on accuracy:

- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- Linear Regressor
- K-Nearest Neighbour
- XGBoost Regressor
- CatBoost Regressor
- AdaBoost Regressor

---

## 🧾 Input Features

### 📊 Numerical Features
- Reading Score
- Writing Score

### 🏷️ Categorical Features
- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch
- Test Preparation Course

---

## 🧰 Tech Stack

- Python
- Flask
- scikit-learn
- catboost
- pandas, numpy


---

## ⚙️ Installation & Running the App

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <project-directory>

2. Create an active virtual environment

3. run on terminal 'python app.py'
   The app will run on http://localhost:5000


This project was developed as part of a learning exercise with the help of a tutorial. It focuses on building a well-structured ML pipeline and integrating it with a simple web interface.




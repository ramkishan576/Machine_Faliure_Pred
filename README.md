<h1 align="center">⚙️ Machine Failure Prediction using Streamlit</h1>
<img src="https://www.animatedimages.org/data/media/562/animated-line-image-0184.gif" width="100%" />

<h3 align="center">A Machine Learning powered web app to predict machine failure using RandomForest, XGBoost, and SMOTE balancing.</h3>

---

## About the Project

This project is a **Streamlit web app** that predicts whether a machine is likely to fail using historical sensor and operational data. It uses **Random Forest** and **XGBoost** models and handles class imbalance using **SMOTE**. The app also allows live predictions using user inputs.

---

##  Tools & Libraries Used

- Streamlit
- Pandas, Numpy
- Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost

---

## Folder Structure

Machine failure predication/
│
├── app.py # Main Streamlit App
├── run.py # Optional Python script to launch app
├── .env # Optional: Config file (e.g., threshold)
├── requirements.txt # Required libraries
├── machine_failure_cleaned.csv # Dataset
├── hello/ # Virtual environment folder (if created)




---

## 🚀 How to Run

### 1. Create and activate virtual environment

```bash
python -m venv hello
hello\Scripts\activate     # Windows

pip install -r requirements.txt

streamlit run app.py



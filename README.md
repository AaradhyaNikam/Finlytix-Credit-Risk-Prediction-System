# 💰 Finlytix: AI Credit Risk Intelligence System

An end-to-end **machine learning-powered credit risk assessment system** with explainable AI, interactive dashboard, and natural language interface.

---

## 💡 Problem Statement
Financial institutions need reliable systems to evaluate loan applicants and minimize default risk.  
Finlytix simulates a **real-world loan approval system** using ML + Explainable AI.

---

## 🚀 Key Features
- Credit Risk Prediction (XGBoost)
- Explainable AI using SHAP
- Interactive Streamlit Dashboard
- Natural Language Query Interface
- Risk Visualization

---

## 📊 Results
- Accuracy: ~95% (observed during evaluation)  
- Evaluation Metric: ROC-AUC (used during validation)

---

## 📊 Dataset & Model Performance
* **Source:** *Give Me Some Credit* Dataset (Kaggle)
* **Features:** Core financial attributes including monthly income, debt ratio, and historical credit lines.
* **Validation Accuracy:** ~95% 
* **Evaluation Focus:** ROC-AUC was prioritized during model validation to properly account for the inherent class imbalances found in real-world credit default datasets.

---

## 📸 Screenshots

### Dashboard
![Dashboard](images/dashboard.png)

### SHAP Waterfall Explanation
![Waterfall](images/waterfall.png)

### Risk Factors (Increase/Decrease)
![Risk](images/risk_factors.png)

---

## 🛠️ Tech Stack
Python | XGBoost | SHAP | Streamlit | Matplotlib | Seaborn | SQL

---

## ⚙️ Run
```bash
git clone https://github.com/AaradhyaNikam/Finlytix-project.git
cd Finlytix-project

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run final_app.py
```

---

## 👨‍💻 Author
Aaradhya Aashish Nikam 2nd-Year B.Tech Student, D.Y. Patil Engineering College, Pune * LinkedIn: https://www.linkedin.com/in/aaradhya-nikam-02a69b32a/

Email: nikamaaradhya97@gmail.com
=======

# 🛡️ Fraud Detection Project

A machine learning project to detect fraudulent transactions using structured financial data and the **XGBoost** algorithm.  

---

## 📌 Features
- ✅ End-to-end fraud detection pipeline  
- ✅ Multiple ML models evaluated, with **XGBoost chosen as final model**  
- ✅ Dataset support (`paysim1.zip` and `PS_20174392719_1491204439457_log.csv`)  
- ✅ Git LFS support for handling large dataset files  
- ✅ Organized project structure for reproducibility  

---

## 📂 Project Structure
- `xgb_model.json` → Trained XGBoost model  
- `requirements.txt` → Python dependencies  
- `data/` → Contains dataset files (`paysim1.zip`, `PS_2017...csv`)  
- `notebooks/` → Jupyter notebooks for model training & experiments  
- `utils/` → Helper functions  
- `.gitignore` → Prevents pushing sensitive files (e.g., `kaggle.json`)  

---

## ⚙️ Installation
- Clone repository:
  ```bash
  git clone https://github.com/pooja49-git/fraud_detection.git
  cd fraud_detection

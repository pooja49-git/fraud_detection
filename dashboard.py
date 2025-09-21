# import streamlit as st
# import json
# import pandas as pd
# from pyspark.sql import SparkSession
# from pyspark.sql.types import StructType, StructField, DoubleType, StringType
# from sparkxgb import XGBoostClassificationModel

# # ⚡ Must be the very first Streamlit command
# st.set_page_config(page_title="Fraud Detection Dashboard", layout="centered")

# # -----------------------
# # Initialize Spark session
# # -----------------------
# spark = SparkSession.builder \
#     .appName("FraudDetectionDashboard") \
#     .getOrCreate()

# # -----------------------
# # Load results & model
# # -----------------------
# results = {}
# model = None
# try:
#     with open("results.json", "r") as f:
#         results = json.load(f)
#     best_model_path = results["best_model"]

#     # ✅ Load XGBoost model instead of Spark Pipeline
#     model = XGBoostClassificationModel.load(best_model_path)

# except Exception as e:
#     st.error(f"Error loading model/metrics: {e}")

# # -----------------------
# # Streamlit UI
# # -----------------------
# st.title("💳 Fraud Detection Dashboard")

# # Show metrics if available
# if "all_metrics" in results:
#     st.subheader("📊 Model Performance Metrics")
#     metrics_df = pd.DataFrame(results["all_metrics"]).T
#     st.dataframe(metrics_df)

# st.markdown("---")

# # -----------------------
# # Fraud Prediction Section
# # -----------------------
# st.subheader("🔮 Test Fraud Prediction")

# with st.form("prediction_form"):
#     amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=10.0)
#     old_balance = st.number_input("Old Balance", min_value=0.0, value=500.0, step=10.0)
#     new_balance = st.number_input("New Balance", min_value=0.0, value=400.0, step=10.0)
#     transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
#     submitted = st.form_submit_button("Predict Fraud")

# if submitted and model:
#     try:
#         # Define schema (make sure it matches your training data)
#         schema = StructType([
#             StructField("type", StringType(), True),
#             StructField("amount", DoubleType(), True),
#             StructField("oldbalanceOrg", DoubleType(), True),
#             StructField("newbalanceOrig", DoubleType(), True)
#         ])

#         # Create Spark DataFrame
#         data = [(transaction_type, float(amount), float(old_balance), float(new_balance))]
#         input_df = spark.createDataFrame(data, schema=schema)

#         # Run prediction
#         predictions = model.transform(input_df).toPandas()

#         # Show result
#         pred_class = int(predictions["prediction"].iloc[0])
#         prob = predictions["probability"].iloc[0]

#         st.success(f"✅ Prediction: {'FRAUD' if pred_class == 1 else 'NOT FRAUD'}")
#         st.write(f"🔹 Probability: {prob}")

#     except Exception as e:
#         st.error(f"Prediction error: {e}")

# elif submitted:
#     st.warning("⚠️ Model not loaded. Please check results.json and model path.")

import streamlit as st
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from groq import Groq  # ✅ for LLaMA 3 explanations

# ⚡ Streamlit config must be first
st.set_page_config(page_title="Fraud Detection Dashboard", layout="centered")

# -----------------------
# Load model & results
# -----------------------
results = {}
booster = None
try:
    with open("results.json", "r") as f:
        results = json.load(f)

    # ✅ Load XGBoost model (pure Python)
    booster = xgb.Booster()
    booster.load_model("xgb_model.json")

except Exception as e:
    st.error(f"Error loading model/metrics: {e}")

# -----------------------
# Initialize Groq Client (set your API key in environment)
# -----------------------
import os
groq_api_key = os.getenv("GROQ_API_KEY")
llm_client = None
if groq_api_key:
    llm_client = Groq(api_key=groq_api_key)

# -----------------------
# Streamlit UI
# -----------------------
st.title("💳 Fraud Detection Dashboard")

# Show metrics if available
if "all_metrics" in results:
    st.subheader("📊 Model Performance Metrics")
    metrics_df = pd.DataFrame(results["all_metrics"]).T
    st.dataframe(metrics_df)

st.markdown("---")

# -----------------------
# Fraud Prediction Section
# -----------------------
st.subheader("🔮 Test Fraud Prediction")

with st.form("prediction_form"):
    amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=10.0)
    old_balance = st.number_input("Old Balance", min_value=0.0, value=500.0, step=10.0)
    new_balance = st.number_input("New Balance", min_value=0.0, value=400.0, step=10.0)
    transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
    submitted = st.form_submit_button("Predict Fraud")

if submitted and booster:
    try:
        # Encode transaction type
        type_mapping = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3}
        type_encoded = type_mapping[transaction_type]

        features = np.array([[amount, old_balance, new_balance, type_encoded]])

        # Convert to DMatrix for XGBoost
        dmatrix = xgb.DMatrix(features)

        # Prediction
        pred_prob = booster.predict(dmatrix)[0]
        pred_class = int(pred_prob > 0.5)

        st.success(f"✅ Prediction: {'FRAUD' if pred_class == 1 else 'NOT FRAUD'}")
        st.write(f"🔹 Probability of Fraud: {pred_prob:.4f}")

        # -----------------------
        # LLM Explanation
        # -----------------------
        if llm_client:
            with st.spinner("💡 Generating AI explanation..."):
                user_prompt = f"""
                A transaction is being checked for fraud.

                Transaction details:
                - Type: {transaction_type}
                - Amount: {amount}
                - Old Balance: {old_balance}
                - New Balance: {new_balance}

                The model predicted: {"FRAUD" if pred_class == 1 else "NOT FRAUD"}
                Probability: {pred_prob:.4f}

                Explain in simple words why the model might classify it this way.
                """
                response = llm_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are a helpful fraud detection assistant."},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                explanation = response.choices[0].message.content
                st.subheader("🧠 AI Explanation")
                st.write(explanation)
        else:
            st.info("ℹ️ No LLM explanation available. Set GROQ_API_KEY to enable it.")

    except Exception as e:
        st.error(f"Prediction error: {e}")

elif submitted:
    st.warning("⚠️ Model not loaded. Please check xgb_model.json and results.json.")


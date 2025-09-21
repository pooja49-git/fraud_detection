# from openai import OpenAI

# client = OpenAI()  # Make sure you set OPENAI_API_KEY in your terminal

# def llm_analysis(transaction_dict):
#     fraud_status = "FRAUD" if transaction_dict.get("predicted_label") == 1 else "NOT FRAUD"

#     prompt = f"""
#     You are a fraud detection assistant.
#     Transaction details:
#     {transaction_dict}

#     Model predicted: {fraud_status}.

#     Explain in simple words:
#     1. Why it might be {fraud_status}.
#     2. Which attributes look suspicious (like large amount, strange balance).
#     3. Suggest what a fraud analyst should do next.
#     """

#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.3
#     )

#     return response.choices[0].message.content
# llm_analysis.py
import os
from groq import Groq
import time

# Initialize client (reads GROQ_API_KEY env var)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Choose a supported model name for your Groq account.
# If you get "decommissioned" errors, check your Groq console and update the name.
DEFAULT_MODEL = "llama-3.3-70b-versatile"  # update if needed

def make_prompt(txn: dict) -> str:
    """
    Create a concise prompt for LLaMA based on the transaction dict.
    txn is expected to include keys:
    amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type, predicted_label
    """
    pred = "FRAUD" if txn.get("predicted_label") == 1 else "NOT FRAUD"
    prompt = (
        "You are a helpful fraud-detection assistant. "
        "Read the transaction below and in simple English do three short things:\n\n"
        "1) One-line explanation why the model predicted: " + pred + ".\n"
        "2) List up to 3 attributes that look suspicious (amount, balances, type etc.).\n"
        "3) Suggest one short next action (e.g., hold transaction, ask customer, investigate account).\n\n"
        f"Transaction: {txn}\n\n"
        "Keep the answer short (max 120 words). Use clear bullet points or numbered lines."
    )
    return prompt

def llm_analysis(txn: dict, model_name: str = DEFAULT_MODEL, max_tokens: int = 200, timeout: int = 15):
    """
    Call Groq LLaMA to create a short explanation for a transaction.
    Returns a plain string. Handles errors and timeouts.
    """
    prompt = make_prompt(txn)

    try:
        # small sleep/cool-down if you want to avoid bursts
        time.sleep(0.1)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a concise assistant for financial fraud explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )

        # Groq SDK returns objects with .message.content
        text = response.choices[0].message.content
        if text is None:
            return "LLM returned an empty response."
        return text.strip()

    except Exception as e:
        # Return error as safe string so Streamlit won't crash
        return f"[LLM error] {str(e)}"


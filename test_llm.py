from groq import Groq
import os

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Call LLaMA-3 model
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",   # update to a supported model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

# Print the reply
print(response.choices[0].message.content)

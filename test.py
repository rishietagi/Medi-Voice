from huggingface_hub import InferenceClient, login

login(token="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")

client = InferenceClient("EleutherAI/gpt-neo-1.3B")

prompt = "Suggest a common Indian male name for a 24-year-old."

try:
    response = client.text_generation(prompt=prompt, max_new_tokens=20)
    print("Response:", response)
except Exception as e:
    print("API Call Failed:", e)

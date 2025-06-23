from feat_gender import extract_features_from_audio
from feat_age import extract_age_features
import pandas as pd
import os
import re
from tqdm import tqdm
import joblib
import random

name_dict = {
    "male": {
        "young": [
            "Aarav", "Vivaan", "Ishaan", "Reyansh", "Krish", "Daksh", "Kabir", "Neil", "Ayaan", "Aryan",
            "Rudra", "Shaurya", "Arnav", "Veer", "Yuvaan", "Atharv", "Hridaan", "Divit", "Nivaan", "Advait"
        ],
        "mature": [
            "Rohit", "Arjun", "Kunal", "Aditya", "Anirudh", "Siddharth", "Amit", "Rajeev", "Nikhil", "Varun",
            "Vikram", "Manish", "Saurabh", "Rahul", "Rajat", "Harsh", "Akhil", "Ramesh", "Naveen", "Sandeep"
        ],
        "old": [
            "Rajesh", "Suresh", "Ramesh", "Vijay", "Prakash", "Mahesh", "Subramaniam", "Bhaskar", "Chandrashekar", "Hariharan",
            "Govind", "Gopal", "Keshav", "Muralidhar", "Omprakash", "Shivaram", "Bharat", "Satish", "Narayan", "Yogesh"
        ]
    },
    "female": {
        "young": [
            "Anaya", "Kiara", "Myra", "Aadhya", "Ira", "Saanvi", "Avni", "Diya", "Tara", "Meher",
            "Riya", "Zoya", "Navya", "Aanya", "Pari", "Aarohi", "Lavanya", "Ishita", "Amaira", "Vanya"
        ],
        "mature": [
            "Sneha", "Pooja", "Priya", "Deepa", "Divya", "Swati", "Shweta", "Neha", "Anjali", "Kavya",
            "Aishwarya", "Bhavana", "Rekha", "Preeti", "Sheetal", "Archana", "Manju", "Bindu", "Jyoti", "Madhu"
        ],
        "old": [
            "Latha", "Kavitha", "Radha", "Geetha", "Usha", "Sumathi", "Janaki", "Malathi", "Savitri", "Lakshmi",
            "Kamala", "Vimala", "Ambika", "Seetha", "Rukmini", "Parvati", "Saraswati", "Annapurna", "Durga", "Meenakshi"
        ]
    }
}

gender_model = joblib.load("gender_classifier.pkl")
gender_scaler = joblib.load("gender_scaler.pkl")
age_model = joblib.load("age_classifier.pkl")

def predict_age(filepath):
    try:
        age_features = extract_age_features(filepath)
        age = age_model.predict(age_features)[0]  
        return age
    except Exception as e:
        print(f"Error predicting age for {filepath}: {e}")
        return "unknown"

def predict_gender(filepath):
    try:
        gender_features = extract_features_from_audio(filepath)
        features_scaled = gender_scaler.transform(gender_features)
        gender = gender_model.predict(features_scaled)[0]
        return gender
    except Exception as e:
        print(f"Error predicting gender for {filepath}: {e}")
        return "unknown"


# genai.configure(api_key="gemini api")
# model = genai.GenerativeModel(model_name="models/gemini-pro")
# client = OpenAI(api_key = "open ai api")


# login(token="mistra api token") 

# client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.1")

# def generate_name(prompt):
#     try:
#         response = model.generate_content(prompt)
#         reply = response.text.strip()
#         names = re.findall(r"[A-Z][a-z]+", reply)
#         return names[0] if names else "Unknown"
#     except Exception as e:
#         print("Gemini API error:", e)
#         return "Unknown"

def generate_name_offline(age_group, gender):
    try:
        gender = gender.lower()
        age_group = age_group.lower()
        name_pool = name_dict.get(gender, {}).get(age_group, [])
        if not name_pool:
            return "Unknown"
        return random.choice(name_pool)
    except Exception as e:
        print("Name generation error:", e)
        return "Unknown"

df = pd.read_csv("ecapa_embeddings.csv")
AUDIO_FOLDER = r"C:\Users\Rishi S Etagi\Desktop\medivoice\LIBRI"

ages, genders, full_names, first_letters = [], [], [], []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating names"):
    filename = row['filename']
    filepath = os.path.join(AUDIO_FOLDER, filename)

    age = predict_age(filepath)
    gender = predict_gender(filepath)

    if age == "unknown" or gender == "unknown":
        full_name = "Unknown"
        first_letter = "U"
    else:
        full_name = generate_name_offline(age, gender)
        first_letter = full_name[0] if full_name != "Unknown" else "U"

    ages.append(age)
    genders.append(gender)
    full_names.append(full_name)
    first_letters.append(first_letter)

df["age"] = ages
df["gender"] = genders
df["generated_name"] = full_names
df["first_letter"] = first_letters

df.to_csv("ecapa_embeddings_with_labels.csv", index=False)
print("Saved: ecapa_embeddings_with_labels.csv")

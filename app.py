from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
from sklearn.feature_extraction.text import CountVectorizer

app = FastAPI()

# Load the trained model
model = joblib.load('naive_bayes_model.joblib')

# Load the vocabulary
with open('vocabulary.json', 'r') as vocab_file:
    vocabulary = json.load(vocab_file)

# Define request body model
class TextData(BaseModel):
    text: str

# Function to vectorize new data
def vectorize_text(text):
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    return vectorizer.transform([text])

@app.post("/predict/")
def predict(text_data: TextData):
    text = text_data.text
    text_vectorized = vectorize_text(text)
    prediction = model.predict(text_vectorized)
    return {"category": prediction[0]}


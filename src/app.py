from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    vec = vectorizer.transform([input.text])
    prediction = model.predict(vec)[0]
    return {"text": input.text, "prediction": prediction}

@app.get("/")
def root():
    return {"message": "TanglishGuard API is running"}

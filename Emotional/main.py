from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from pydantic import BaseModel
import uvicorn

# ----------------------
# Load model & tokenizer
# ----------------------
model = load_model('text_classification_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('max_length.pkl', 'rb') as f:
    max_length = pickle.load(f)

# ----------------------
# FastAPI app
# ----------------------
app = FastAPI(title="Text Classification App")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ----------------------
# Stopwords remover
# ----------------------
def remove_stopwords(text):
    stopwords = set(['a', 'an', 'the', 'is', 'am', 'are', 'i', 'you'])
    return ' '.join([word for word in text.lower().split() if word not in stopwords])

# ----------------------
# Classification function
# ----------------------
def classify_text(input_text):
    input_text_cleaned = remove_stopwords(input_text).lower() 
    
    # Keywords
    emotion_keywords = {
        'sadness': 'sadness', 'joy': 'joy', 'love': 'love', 'anger': 'anger',
        'fear': 'fear', 'surprise': 'surprise', 'happy': 'joy', 'unhappy': 'sadness', 'afraid': 'fear'
    }
    violence_keywords = {
        'sexual violence': 'sexual_violence', 'physical violence': 'physical_violence',
        'emotional violence': 'emotional_violence', 'harmful traditional practice': 'Harmful_traditional_practice',
        'economic violence': 'economic_violence'
    }
    hate_keywords = {'offensive speech': 'Offensive speech', 'hate speech': 'Hate Speech'}

    # Keyword check
    for word, label in emotion_keywords.items():
        if word in input_text_cleaned:
            return "Emotion", label
    for word, label in violence_keywords.items():
        if word in input_text_cleaned:
            return "Violence", label
    for word, label in hate_keywords.items():
        if word in input_text_cleaned:
            return "Hate", label

    # Model prediction
    input_seq = tokenizer.texts_to_sequences([input_text_cleaned])
    input_pad = pad_sequences(input_seq, maxlen=max_length, padding='post')
    predictions = model.predict({
        'emotion_input': input_pad,
        'violence_input': input_pad,
        'hate_input': input_pad
    })

    emotion_pred = np.argmax(predictions[0], axis=1)[0]
    violence_pred = np.argmax(predictions[1], axis=1)[0]
    hate_pred = np.argmax(predictions[2], axis=1)[0]

    confidences = [np.max(predictions[0]), np.max(predictions[1]), np.max(predictions[2])]
    major_labels = ['Emotion', 'Violence', 'Hate']
    major_index = np.argmax(confidences)
    major_label = major_labels[major_index]

    if major_label == "Emotion":
        sub_label = ['sadness','joy','love','anger','fear','surprise'][emotion_pred]
    elif major_label == "Violence":
        sub_label = ['sexual_violence','physical_violence','emotional_violence',
                     'Harmful_traditional_practice','economic_violence'][violence_pred]
    else:
        sub_label = ['Offensive speech','Neither','Hate Speech'][hate_pred]

    return major_label, sub_label

# ----------------------
# JSON request model
# ----------------------
class TextRequest(BaseModel):
    text: str

# ----------------------
# Routes
# ----------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
async def api_predict(request: TextRequest):
    major, sub = classify_text(request.text)
    return {"major_label": major, "sub_label": sub}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# app.py
from fastapi import FastAPI, UploadFile, File
from detect_sentiment import detect_sentiment
import shutil

app = FastAPI()

@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    temp_path = f"temp_{audio.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    result = detect_sentiment(temp_path)
    return result
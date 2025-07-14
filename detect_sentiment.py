
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, pipeline


# Load pre-trained model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


# Models
asr_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")


def convert_sentiment(label):
    if "1" in label or "2" in label:
        return "Negatif"
    elif "3" in label:
        return "Neutre"
    else:
        return "Positif"

def detect_sentiment(path_audio):
    # read audio
    speech, rate = librosa.load(path_audio, sr=16000)
    
    # Transcription
    input_values = asr_tokenizer(speech, return_tensors='pt').input_values
    with torch.no_grad():
        logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    text = asr_tokenizer.decode(predicted_ids[0])

    # Transcription
    input_values = tokenizer(speech, return_tensors='pt').input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    text = tokenizer.decode(predicted_ids[0])

    # Analyse de sentiment
    result = sentiment_model(text)[0]
    label = result["label"]
    mapped_label = convert_sentiment(label)

    return {"text": text, "raw_label": label, "sentiment": mapped_label}
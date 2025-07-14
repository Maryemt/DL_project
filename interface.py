# Gradio interface

import gradio as gr
from detect_sentiment import detect_sentiment

def gradio_interface(audio_file):
    result = detect_sentiment(audio_file)
    return result["text"], result["sentiment"]

gr.Interface(
    fn=gradio_interface,
    inputs=gr.Audio(type="filepath", label="Audio"),
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Sentiment détecté:")
    ],
    title="Detection Automatique de Sentiment dans des Appels Vocaux",
    description="Transcription de l audiio et Detection du sentiment (positif, neutre, negatif)"
).launch()

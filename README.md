# Détection Automatique de Sentiment dans des Appels Vocaux à l’aide de Wav2Vec 2.0 et BERT

Ce projet propose un pipeline intelligent qui transcrit un appel vocal et déduit le sentiment du locuteur (positif, neutre ou négatif), 
en combinant un modèle de transcription audio (`Wav2Vec2`) et un modèle de NLP (BERT).

Les modèles utilisés sont:
 - Wav2Vec 2.0 [https://huggingface.co/docs/transformers/model_doc/wav2vec2](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
 - bert-base-multilingual-uncased-sentiment https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment

**Architecture / Workflow**

  1. Chargement et traitement des fichiers audio 
  
  2. Transcription vocale en texte avec Wav2Vec 2.0
  
  3. Analyse de sentiment à partir du texte avec BERT ; 
  
  4. Classification du sentiment de la transcription pour les modalités "positif", "négaltif" et "neutre"

### Pipeline

audio.wav (voix client)  →  [Wav2Vec2] Transcription texte  → [BERT Sentiment] Analyse du sentiment  →   Classification : "positif", "neutre", "négatif"


### Installation

```
git clone https://github.com/Maryemt/DL_project.git
cd DL_project
```

Creer environnement virtuel 
```
python3 -m venv .venv
source .venv/bin/activate
```

Installer les dependances 
```
pip install -r required.txt
```

Lancer Gradio 
```
python interface.py
```
App disponible au http://127.0.0.1:7861 

API REST
```
uvicorn main:app --reload
```

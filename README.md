# Sentiment & Emotion Analysis – Multimodal Pipeline

This project implements a multimodal sentiment and emotion analysis system capable of processing both text and audio input. The goal is to compare traditional neural network models with commercial LLM (Claude) APIs like Claude. The system supports binary sentiment classification and multi-class emotion detection using local models as well as LLM-based inference.

The full machine learning pipeline includes: text preprocessing and vectorization, neural network model training (MLP, LSTM, Transformers, BERT), speech-to-text transcription using OpenAI Whisper, Claude API integration for benchmarking, and dashboard deployment for real-time user interaction.

---

## Required Files & Structure

Ensure the following folders are included in your local setup:

```
Emotion_Model/Binary2gram/
Sentiment_Model/Binary2gram/
Whisper_Model/Audios/
```

---

## Setup Environment

1. Create virtual environment:
```bash
conda env create -f APA_venv.yaml
conda activate APA_venv
```

2. Set Claude API Key:  
Update your API key in `compare_models.py`:
```python
claude = Anthropic(api_key="your_api_key_here")
```

3. Launch Jupyter to run standalone scripts:  
Model deployment: `Model_Pipeline_2.ipynb`
Model training: `ATPA_Emotions.ipynb`, `ATPA_Sentiments.ipynb`

---

## Dashboard

- **Sentiment Diary** is a web app that captures live speech, transcribes it, and analyzes emotional tone and sentiment live.
- Includes a weekly sentiment tracker, real-time emotion detection, and a clean, intuitive dashboard.
- For full setup instructions refer to the Project Sharing Guide in the link below.

  - Open a terminal on the src level and run the following command in your terminal to start the dashboard:
```python3 main.py
```
  - Go to Safari / Chrome and open this website http://127.0.0.1:5001 to view and use the dashboard

Files must be downloaded via the link, as model too large for GH:  
https://ucppt-my.sharepoint.com/:f:/g/personal/s-tbajramaj_ucp_pt/ELrBDSRIsdBJjhZA-SV9MWcBRu7MU50duUdYsX9Y_z7ZmA?e=cP3e0U

---

## Project Components

- `Model_Pipeline_2.ipynb`: Core notebook running the complete sentiment + emotion model pipeline (other modules will be imported)
- `Model_Deployment_functions.py`: Utility functions for model loading, predictions, and evaluation
- `Claude_functions.py`: Claude API integration for benchmarking
- `Whisper_functions.py`: Speech-to-text preprocessing using OpenAI Whisper
- `Result_functions.py`: Performance tracking and evaluation helpers

---

## Further Explanation

**Model Deployment**  
Models are deployed locally using `.h5` and config files. Emotion and Sentiment detection run on top of transcribed audio using Whisper.

**Function Overview**
- `Whisper_functions.py`: Transcribe `.m4a` audios to text using Whisper
- `Claude_functions.py`: Query Claude with user input or Whisper transcriptions
- `Model_Deployment_functions.py`: Run emotion/sentiment predictions with trained MLP/Transformer models
- `Result_functions.py`: Format and visualize results

---

## References

This project is part of the *Advanced Predictive Analytics* course at Católica Lisbon. Full technical documentation, model comparison, and benchmark results can be found in:

- [APA Final Report](https://github.com/timbaaacodes/APA_shared)
- [GoEmotions Dataset](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/)
- [OpenAI Whisper GitHub](https://github.com/openai/whisper)
---

## Authors

Tim Bajramaj, André Rodrigues, Claudius Kroflin, Sophie Nüssel
*Advanced Predictive Analytics – May 2025*






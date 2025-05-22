import whisper
import os
import pandas as pd
from jiwer import wer, cer, compute_measures
import subprocess
import numpy as np
from tqdm import tqdm

# Env set up
cd = os.getcwd()
print(cd)

# Ensure ffmpeg is visible to Python
try:
    subprocess.run(["ffmpeg", "-version"], check=True)
    print("FFmpeg is available to Python!")
except FileNotFoundError:
    print("FFmpeg is still NOT visible to Python.")
    os.environ["PATH"] = "/opt/anaconda3/bin:" + os.environ["PATH"]
    print("Try again!")

# Transcribe memo file with whisper model to text
def speech_to_text(input_file):

    # Load base model
    model = whisper.load_model("base", device="cpu")

    # Transcribe
    result = model.transcribe(input_file,fp16=False)
    
    # Output text
    text = result["text"]
    #print(text)

    return text

# Function that imports main audio summary
def get_audio_metrics():

    file_path = f'{cd}/Whisper_Model/Audios/Audio_memos.csv'

    audio_summary = pd.read_csv(file_path, index_col=0, delimiter=',')

    return audio_summary

# Function that compares transcribed text from whisper to original text
def compare_texts(input_file, true_text, print_ind):

    # Transcribe one text - input file must be unique
    transcribed_text = speech_to_text(input_file)

    # Compute Word Error Rate
    error = wer(true_text, transcribed_text)
    
    # Compute Character Error Rate (CER) ignoring punctuation
    error_cer = cer(true_text, transcribed_text)
    if print_ind == 'Y':
        print(f"CER: {error_cer * 100:.2f}%")
        print(f"WER: {error * 100:.2f}%")

    return error_cer


def benchmark_speech_to_text(audio_summary):

    true_texts = []
    transcribed_texts = []

    print_results_ind = input("Print texts? (Y/N)")

    # Ensure 'Error CER' column exists
    if 'Error CER' not in audio_summary.columns:
        audio_summary['Error CER'] = None

    for i in tqdm(range(len(audio_summary)), total=len(audio_summary), desc="Processing files"):

        memo = audio_summary.iloc[i]

        # Get input file
        input_file = f'{cd}/Whisper_Model/Audios/id_{i+1}.m4a'
        

        true_text = memo['text']
        true_texts.append(true_text)
        
        transcribed_text = speech_to_text(input_file)
        transcribed_texts.append(transcribed_text)
        # Compare number of correct words
        error_cer = compare_texts(input_file, true_text=true_text, print_ind=print_results_ind)

        # Ensure error_cer is a scalar
        if isinstance(error_cer, (list, tuple, np.ndarray)):
            error_cer = error_cer[0]  # Extract the first value

        if print_results_ind == "Y":
            print(f'True text: {true_text}')
            print(f'Transcribed text: {transcribed_text}')

        # Add error to the DataFrame
        audio_summary.at[i, 'Error CER'] = error_cer

    # Calculate average rate of correctness
    avg_error_cer_rate_transcribing = audio_summary['Error CER'].mean()

    return avg_error_cer_rate_transcribing, transcribed_texts, true_texts

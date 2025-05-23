# compare_models.py
import os
import sys
import re
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from anthropic import Anthropic
from tqdm import tqdm

def create_single_comparison_file(api_key='', audio_memos_path=None):

    # New helper function to find model files recursively
    def find_model_files(model_type, base_dir=None):
        """
        Recursively searches for model CSV files across the project directory
        
        Parameters:
        -----------
        model_type : str
            Type of model to find ('Emotion' or 'Sentiment')
        base_dir : str, optional
            Base directory to start the search, if None uses current working directory
            
        Returns:
        --------
        list
            List of tuples (model_name, file_path)
        """
        if base_dir is None:
            base_dir = os.getcwd()
            
        model_files = []
        
        # Search directories that might contain model files
        search_dirs = [
            base_dir,
            os.path.join(base_dir, "Model_results"),
            os.path.join(base_dir, f"{model_type}_Model")
        ]
        
        # Check if we need case-insensitive search
        model_type_lower = model_type.lower()
        
        # Walk through all directories
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith('.csv') and (model_type in file or model_type_lower in file.lower()):
                            # Get full path and model name
                            file_path = os.path.join(root, file)
                            model_name = os.path.splitext(file)[0]
                            model_files.append((model_name, file_path))
                            
        print(f"Found {len(model_files)} {model_type} model files:")
        for name, path in model_files:
            print(f" - {name}: {path}")
            
        return model_files
    
    # Find the Audio_memos file (ground truth)
    if audio_memos_path is None:
        # Try several possible locations based on your folder structure
        possible_paths = [
            f"{os.getcwd()}/Whisper_Model/Audios/audio_memos.csv",
            f"{os.getcwd()}/Whisper_Model/Audios/Audio_memos.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                audio_memos_path = path
                break
                
        if audio_memos_path is None:
            raise FileNotFoundError("Could not find Audio_memos.csv file. Please provide the path.")
    
    print(f"Using benchmark file: {audio_memos_path}")
    
    # Load ground truth data
    ground_truth = pd.read_csv(audio_memos_path)
    
    # Ensure we have 'text', 'emotion' and 'sentiment' columns
    if 'text' not in ground_truth.columns:
        # Try to find the text column
        text_cols = ['transcript', 'content', 'transcription']
        for col in text_cols:
            if col in ground_truth.columns:
                ground_truth = ground_truth.rename(columns={col: 'text'})
                break
    
    # Ensure we have emotion and sentiment columns
    emotion_col = None
    sentiment_col = None
    
    for col in ground_truth.columns:
        if col.lower() == 'emotion' or 'emotion' in col.lower():
            emotion_col = col
        if col.lower() == 'sentiment' or 'sentiment' in col.lower():
            sentiment_col = col
    
    if emotion_col is None or sentiment_col is None:
        raise ValueError(f"Could not identify emotion or sentiment columns in {audio_memos_path}")
    
    # Create the master comparison DataFrame
    comparison_df = pd.DataFrame()
    comparison_df['text'] = ground_truth['text']
    comparison_df['ground_truth_emotion'] = ground_truth[emotion_col]
    comparison_df['ground_truth_sentiment'] = ground_truth[sentiment_col]
    
    # Setup results directory for output
    results_dir = f"{os.getcwd()}/Model_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Dictionary to track accuracy and similarity
    accuracy_stats = {
        'emotion': {},
        'sentiment': {}
    }
    
    # Find emotion model files
    emotion_models = find_model_files("Emotion")
    
    # If no models found, use default list
    if not emotion_models:
        print("No emotion models found. Checking default locations...")
        for model_name in ['Binary2gram_Emotion', 'TESTER_Emotion']:
            model_path = f"{results_dir}/{model_name}.csv"
            if os.path.exists(model_path):
                emotion_models.append((model_name, model_path))
    
    # Process each emotion model
    for model_name, model_path in emotion_models:
        try:
            print(f"Processing emotion model: {model_name} from {model_path}")
            model_df = pd.read_csv(model_path)
            
            # Ensure the text column matches up
            if 'text' in model_df.columns:
                emotion_cols = [col for col in model_df.columns if col not in ['iterator', 'text']]
                if emotion_cols:
                    # Get top emotion for each row
                    model_df['top_emotion'] = model_df[emotion_cols].idxmax(axis=1)
                    model_df['top_emotion_score'] = model_df[emotion_cols].max(axis=1)
                    
                    # Add to comparison DataFrame
                    comparison_df[f'{model_name}_prediction'] = model_df['top_emotion']
                    
                    # Calculate accuracy
                    comparison_df[f'{model_name}_accuracy'] = (
                        comparison_df['ground_truth_emotion'] == comparison_df[f'{model_name}_prediction']
                    ).astype(int)
                    
                    # Track accuracy
                    accuracy_stats['emotion'][model_name] = comparison_df[f'{model_name}_accuracy'].mean()
                    print(f"Successfully processed emotion model: {model_name}")
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    # Find sentiment model files
    sentiment_models = find_model_files("Sentiment")
    
    # If no sentiment models found, use default list
    if not sentiment_models:
        print("No sentiment models found. Checking default locations...")
        sentiment_model_files = ['Binary2gram_Sentiment']
        for model_name in sentiment_model_files:
            model_path = f"{results_dir}/{model_name}.csv"
            if os.path.exists(model_path):
                sentiment_models.append((model_name, model_path))
    
    # Process sentiment models
    for model_name, model_path in sentiment_models:
        try:
            print(f"Processing sentiment model: {model_name} from {model_path}")
            model_df = pd.read_csv(model_path)
            
            # Check if this is a sentiment score model
            if 'positive_rate' in model_df.columns:
                # Add categorical prediction if ground truth is categorical
                if ground_truth[sentiment_col].dtype == object:
                    comparison_df[f'{model_name}_prediction'] = np.where(
                        model_df['positive_rate'] > .50, 'positive', 'negative'
                    )
                    
                    # Calculate accuracy for categorical sentiment
                    comparison_df[f'{model_name}_accuracy'] = (
                        comparison_df['ground_truth_sentiment'] == comparison_df[f'{model_name}_prediction']
                    ).astype(int)
                else:
                    # For numeric sentiment, calculate accuracy based on threshold
                    threshold = .20  # Adjust as needed
                    
                    # Store the numeric sentiment for later use in accuracy calculation
                    temp_score = model_df['positive_rate']
                    
                    comparison_df[f'{model_name}_accuracy'] = (
                        abs(comparison_df['ground_truth_sentiment'] - temp_score) < threshold
                    ).astype(int)
                
                # Track accuracy
                accuracy_stats['sentiment'][model_name] = comparison_df[f'{model_name}_accuracy'].mean()
                print(f"Successfully processed sentiment model: {model_name}")
            else:
                print(f"Warning: {model_name} doesn't have positive_rate column")
                
                # Try to find alternative columns for sentiment
                all_cols = list(model_df.columns)
                print(f"Available columns: {all_cols}")
                
                # If there's a clear sentiment column, use it
                sentiment_candidates = [col for col in all_cols if 'sentiment' in col.lower() or 'positive' in col.lower()]
                if sentiment_candidates:
                    sentiment_col_name = sentiment_candidates[0]
                    print(f"Using alternative sentiment column: {sentiment_col_name}")
                    
                    # Process as if this was positive_rate
                    comparison_df[f'{model_name}_prediction'] = np.where(
                        model_df[sentiment_col_name] > .50, 'positive', 'negative'
                    )
                    
                    # Calculate accuracy
                    comparison_df[f'{model_name}_accuracy'] = (
                        comparison_df['ground_truth_sentiment'] == comparison_df[f'{model_name}_prediction']
                    ).astype(int)
                    
                    # Track accuracy
                    accuracy_stats['sentiment'][model_name] = comparison_df[f'{model_name}_accuracy'].mean()
                    print(f"Successfully processed sentiment model using {sentiment_col_name}")
                
        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    # If we didn't find any sentiment models, try to run Binary2gram model as sentiment model
    if len(accuracy_stats['sentiment']) == 0:
        print("Warning: No sentiment models found. Creating a synthetic sentiment model from Binary2gram_Emotion")
        
        # Try to find Binary2gram_Emotion file
        emotion_files = find_model_files("Emotion")
        binary2gram_path = None
        
        for name, path in emotion_files:
            if "Binary2gram" in name:
                binary2gram_path = path
                break
        
        if binary2gram_path is None:
            binary2gram_path = f"{results_dir}/Binary2gram_Emotion.csv"
        
        try:
            # Check if Binary2gram_Emotion exists and has the right structure
            if os.path.exists(binary2gram_path):
                model_df = pd.read_csv(binary2gram_path)
                
                # Create a synthetic sentiment score based on positive/negative emotions
                if 'text' in model_df.columns:
                    # Define positive and negative emotions
                    positive_emotions = ['joy', 'love', 'excitement', 'gratitude', 'pride', 
                                         'amusement', 'approval', 'admiration', 'optimism']
                    negative_emotions = ['anger', 'fear', 'sadness', 'disgust', 'disappointment', 
                                         'annoyance', 'grief', 'remorse', 'nervousness', 'embarrassment']
                    
                    # Calculate sentiment score as sum of positive emotions minus sum of negative emotions
                    pos_cols = [col for col in model_df.columns if col in positive_emotions]
                    neg_cols = [col for col in model_df.columns if col in negative_emotions]
                    
                    print(f"Positive emotion columns found: {pos_cols}")
                    print(f"Negative emotion columns found: {neg_cols}")
                    
                    if pos_cols and neg_cols:
                        temp_sentiment = .50 + .50 * (
                            model_df[pos_cols].sum(axis=1) - model_df[neg_cols].sum(axis=1)
                        ) / (model_df[pos_cols].sum(axis=1) + model_df[neg_cols].sum(axis=1) + 1e-10)
                        
                        # Add categorical prediction if ground truth is categorical
                        if ground_truth[sentiment_col].dtype == object:
                            comparison_df['Binary2gram_Synthetic_Sentiment_prediction'] = np.where(
                                temp_sentiment > .50, 'positive', 'negative'
                            )
                            
                            # Calculate accuracy
                            comparison_df['Binary2gram_Synthetic_Sentiment_accuracy'] = (
                                comparison_df['ground_truth_sentiment'] == 
                                comparison_df['Binary2gram_Synthetic_Sentiment_prediction']
                            ).astype(int)
                        else:
                            # For numeric sentiment
                            threshold = .20
                            comparison_df['Binary2gram_Synthetic_Sentiment_accuracy'] = (
                                abs(comparison_df['ground_truth_sentiment'] - temp_sentiment) < threshold
                            ).astype(int)
                        
                        # Track accuracy
                        accuracy_stats['sentiment']['Binary2gram_Synthetic_Sentiment'] = comparison_df['Binary2gram_Synthetic_Sentiment_accuracy'].mean()
        except Exception as e:
            print(f"Error creating synthetic sentiment model: {e}")
    
    # Rest of the function remains the same...
    # Run Claude analysis
    print("Running Claude analysis...")
    
    client = Anthropic(api_key=api_key)
    emotion_labels = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
        "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
        "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
    ]
    
    # Add columns for Claude predictions
    comparison_df['Claude_Emotion_prediction'] = ""
    # Temporarily store sentiment scores for calculations
    temp_claude_sentiment_scores = np.zeros(len(comparison_df))
    
    # Process each text
    for i, text in tqdm(enumerate(comparison_df['text']), total=len(comparison_df['text']), desc="Processing texts"):
        
        # Emotion analysis
        emotion_prompt = f"""
        Analyze the emotional content in this text. Rate each emotion on a scale from 0 to 100, 
        where 0 means the emotion is not present and 100 means the emotion is very strongly present.
        
        Text: "{text}"
        
        Rate each of these emotions:
        {', '.join(emotion_labels)}
        
        Provide your response in a JSON format with emotion names as keys and numeric scores as values.
        For example: {{"joy": 85.5, "sadness": 0.0, ...}}
        """
        
        try:
            emotion_response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                system="You are an expert in emotion analysis. Provide accurate numeric ratings for emotions in text.",
                messages=[{"role": "user", "content": emotion_prompt}]
            )
            
            # Extract JSON
            match = re.search(r'\{.*\}', emotion_response.content[0].text, re.DOTALL)
            if match:
                emotion_results = json.loads(match.group(0))
                
                # Find top emotion
                top_emotion = max(emotion_results.items(), key=lambda x: x[1])
                comparison_df.at[i, 'Claude_Emotion_prediction'] = top_emotion[0]
        except Exception as e:
            print(f"Error with Claude emotion analysis: {e}")
        
        # Sentiment analysis
        sentiment_prompt = f"""
        Analyze the sentiment in this text on a scale from 0 to 100, where:
        - 0 is extremely negative
        - 50 is neutral
        - 100 is extremely positive
        
        Text: "{text}"
        
        Provide ONLY a single number between 0 and 100 as your response.
        """
        
        try:
            sentiment_response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=50,
                temperature=0,
                system="You are an expert in sentiment analysis. Provide a single numeric rating.",
                messages=[{"role": "user", "content": sentiment_prompt}]
            )
            
            # Extract numeric value
            match = re.search(r'(\d+(\.\d+)?)', sentiment_response.content[0].text)
            if match:
                sentiment_score = float(match.group(1))
                # Store sentiment score temporarily for calculations but don't include in final DataFrame
                temp_claude_sentiment_scores[i] = sentiment_score
        except Exception as e:
            print(f"Error with Claude sentiment analysis: {e}")
    
    # Claude sentiment prediction
    if ground_truth[sentiment_col].dtype == object:
        comparison_df['Claude_Sentiment_prediction'] = np.where(
            temp_claude_sentiment_scores > 50, 'positive', 'negative'
        )
    
    # Calculate Claude accuracy
    comparison_df['Claude_Emotion_accuracy'] = (
        comparison_df['ground_truth_emotion'] == comparison_df['Claude_Emotion_prediction']
    ).astype(int)
    
    if ground_truth[sentiment_col].dtype == object:
        comparison_df['Claude_Sentiment_accuracy'] = (
            comparison_df['ground_truth_sentiment'] == comparison_df['Claude_Sentiment_prediction']
        ).astype(int)
    else:
        threshold = 20
        comparison_df['Claude_Sentiment_accuracy'] = (
            abs(comparison_df['ground_truth_sentiment'] - temp_claude_sentiment_scores) < threshold
        ).astype(int)
    
    # Track Claude accuracy
    accuracy_stats['emotion']['Claude_Emotion'] = comparison_df['Claude_Emotion_accuracy'].mean()
    accuracy_stats['sentiment']['Claude_Sentiment'] = comparison_df['Claude_Sentiment_accuracy'].mean()
    
    # Calculate similarity between models and Claude
    for model_name in [m for m in accuracy_stats['emotion'] if m != 'Claude_Emotion']:
        if f'{model_name}_prediction' in comparison_df.columns:
            comparison_df[f'{model_name}_claude_similarity'] = (
                comparison_df[f'{model_name}_prediction'] == comparison_df['Claude_Emotion_prediction']
            ).astype(int)
            
            # Track similarity
            accuracy_stats['emotion'][f'{model_name}_claude_similarity'] = comparison_df[f'{model_name}_claude_similarity'].mean()
    
    # Calculate sentiment similarity with Claude
    sentiment_models_to_check = list(accuracy_stats['sentiment'].keys())  # Create a static list first
    for model_name in sentiment_models_to_check:
        if model_name != 'Claude_Sentiment' and f'{model_name}_prediction' in comparison_df.columns:
            if 'Claude_Sentiment_prediction' in comparison_df.columns:
                comparison_df[f'{model_name}_claude_similarity'] = (
                    comparison_df[f'{model_name}_prediction'] == comparison_df['Claude_Sentiment_prediction']
                ).astype(int)
                
                # Track similarity
                accuracy_stats['sentiment'][f'{model_name}_claude_similarity'] = comparison_df[f'{model_name}_claude_similarity'].mean()
    
    # Add summary rows
    # Create a new row for accuracy summary
    accuracy_row = {'text': 'AVERAGE ACCURACY'}
    
    # Add accuracy for each model
    for col in comparison_df.columns:
        if col.endswith('_accuracy'):
            accuracy_row[col] = comparison_df[col].mean()
    
    # Add prediction summary for readability
    for col in comparison_df.columns:
        if col.endswith('_prediction'):
            accuracy_row[col] = "↑ Accuracy score"
    
    # Add ground truth columns
    accuracy_row['ground_truth_emotion'] = "Accuracy"
    accuracy_row['ground_truth_sentiment'] = "Accuracy"
    
    # Create a new row for Claude similarity summary
    similarity_row = {'text': 'CLAUDE SIMILARITY'}
    
    # Add similarity measures
    for col in comparison_df.columns:
        if col.endswith('_claude_similarity'):
            similarity_row[col] = comparison_df[col].mean()
    
    # Add prediction summary for readability
    for col in comparison_df.columns:
        if col.endswith('_prediction'):
            similarity_row[col] = "↑ Similarity to Claude"
    
    # Add ground truth columns
    similarity_row['ground_truth_emotion'] = "Similarity"
    similarity_row['ground_truth_sentiment'] = "Similarity"
    
    # Add rows to DataFrame
    comparison_df = pd.concat([comparison_df, pd.DataFrame([accuracy_row, similarity_row])], ignore_index=True)
    
    # Save the complete comparison
    comparison_df.to_csv(f"{results_dir}/complete_comparison.csv", index=False)
    
    # Create separate emotion and sentiment comparison CSVs for easier analysis
    emotion_cols = ['text', 'ground_truth_emotion'] + [col for col in comparison_df.columns if '_Emotion_' in col]
    sentiment_cols = ['text', 'ground_truth_sentiment'] + [col for col in comparison_df.columns if '_Sentiment_' in col]
    
    # Save separate comparison files
    comparison_df[emotion_cols].to_csv(f"{results_dir}/emotion_only_comparison.csv", index=False)
    comparison_df[sentiment_cols].to_csv(f"{results_dir}/sentiment_only_comparison.csv", index=False)
    
    # Create and save summary statistics
    emotion_metrics = {k: v for k, v in accuracy_stats['emotion'].items()}
    sentiment_metrics = {k: v for k, v in accuracy_stats['sentiment'].items()}
    
    summary_df = pd.DataFrame({
        'metric': list(emotion_metrics.keys()) + list(sentiment_metrics.keys()),
        'value': list(emotion_metrics.values()) + list(sentiment_metrics.values()),
        'category': ['emotion'] * len(emotion_metrics) + ['sentiment'] * len(sentiment_metrics)
    })
    summary_df.to_csv(f"{results_dir}/accuracy_summary.csv", index=False)
    
    # Print summary
    print(f"\n===== ACCURACY SUMMARY =====")
    print(f"\nEMOTION METRICS:")
    for metric, value in emotion_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nSENTIMENT METRICS:")
    for metric, value in sentiment_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nResults saved to:")
    print(f"- Complete comparison: {os.getcwd()}/Model_results/complete_comparison.csv")
    print(f"- Emotion comparison: {os.getcwd()}/Model_results/emotion_only_comparison.csv")
    print(f"- Sentiment comparison: {os.getcwd()}/Model_results/sentiment_only_comparison.csv")
    print(f"- Summary statistics: {os.getcwd()}/Model_results/accuracy_summary.csv")
    
    return comparison_df

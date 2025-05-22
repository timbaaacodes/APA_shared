# Add to Result_functions.py


def run_claude_analysis(transcribed_texts, api_key=None, debugg_mode='N'):
    """Run Claude analysis on the transcribed texts"""
    import pandas as pd
    import os
    import Claude_functions as cf
    import gc
    import json
    
    # Create Claude instance
    claude_instance = cf.create_claude_instance(api_key=api_key, debugg_mode=debugg_mode)
    
    print('Running Claude classification tasks')
    
    emotion_labels = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
        "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
        "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
    ]
    
    # For Claude emotion analysis
    emotion_results_df = pd.DataFrame(columns=['iterator', 'text'] + emotion_labels).set_index('iterator')
    
    # For Claude sentiment analysis
    sentiment_results_df = pd.DataFrame(columns=['iterator', 'text', 'positive_rate']).set_index('iterator')
    
    for iterator, text in enumerate(transcribed_texts):
        print(f'Processing text {iterator+1}/{len(transcribed_texts)} with Claude')
        
        # Get Claude emotion predictions
        emotion_result = claude_instance.predict_emotion(text)
        
        # Add to emotion dataframe
        new_emotion_row = pd.DataFrame([[iterator + 1, text] + [emotion_result.get(label, 0.0) for label in emotion_labels]], 
                           columns=['iterator', 'text'] + emotion_labels)
        new_emotion_row = new_emotion_row.set_index('iterator')
        emotion_results_df = pd.concat([emotion_results_df, new_emotion_row])
        
        # Get Claude sentiment prediction
        sentiment_result = claude_instance.predict_sentiment(text)
        
        # Add to sentiment dataframe
        new_sentiment_row = pd.DataFrame([[iterator + 1, text, sentiment_result]], 
                          columns=['iterator', 'text', 'positive_rate'])
        new_sentiment_row = new_sentiment_row.set_index('iterator')
        sentiment_results_df = pd.concat([sentiment_results_df, new_sentiment_row])
        
        # Clear memory
        gc.collect()
    
    # Store Claude results
    store_model_results(model_results=emotion_results_df, model_name="Claude_Emotion")
    store_model_results(model_results=sentiment_results_df, model_name="Claude_Sentiment")
    
    # Save cost summary
    cost_summary = claude_instance.get_cost_summary()
    results_dir = f"{os.getcwd()}/Model_results"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/claude_cost_summary.json", "w") as f:
        json.dump(cost_summary, f, indent=2)
        
    with open(f"{results_dir}/claude_prompt_designs.json", "w") as f:
        json.dump(claude_instance.get_prompt_documentation(), f, indent=2)
    
    print('Claude analysis completed')
    
    return {
        "claude_emotion": emotion_results_df,
        "claude_sentiment": sentiment_results_df,
        "cost_summary": cost_summary
    }


def store_model_results(model_results, model_name):
    """Store model results to a CSV file."""
    import os
    
    # Create the Model_results directory if it doesn't exist
    results_dir = f"{os.getcwd()}/Model_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the model results to a CSV file
    output_path = f"{results_dir}/{model_name}.csv"
    model_results.to_csv(output_path, index=False)
    
    print(f"Saved result csv for {model_name}")
    
    return output_path


def create_comparison_table(audio_memos_path=None):
    """
    Creates a comprehensive comparison table of emotion and sentiment predictions
    against ground truth from Audio_memos file.
    
    Parameters:
    -----------
    audio_memos_path : str, optional
        Path to the Audio_memos.csv file. If None, will look in default locations.
    
    Returns:
    --------
    tuple
        (emotion_comparison_df, sentiment_comparison_df, emotion_summary, sentiment_summary)
    """
    import pandas as pd
    import os
    import numpy as np
    
    # Find the Audio_memos file (ground truth)
    if audio_memos_path is None:
        # Try several possible locations based on your folder structure
        possible_paths = [
            f"{os.getcwd()}/Whisper_Model/Archive/audio_memos.csv",
            f"{os.getcwd()}/Whisper_Model/Audios/Audio_memos.csv",
            f"{os.getcwd()}/Whisper_Model/Archive/Audio_memos.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                audio_memos_path = path
                break
                
        if audio_memos_path is None:
            raise FileNotFoundError("Could not find Audio_memos.csv file. Please provide the path.")
    
    # Load ground truth data
    ground_truth = pd.read_csv(audio_memos_path)
    
    # Standardize column names if needed
    if 'text' not in ground_truth.columns:
        # Try to find the text column (might be named transcript, content, etc.)
        text_col_candidates = ['transcript', 'content', 'transcription']
        for col in text_col_candidates:
            if col in ground_truth.columns:
                ground_truth = ground_truth.rename(columns={col: 'text'})
                break
                
    # Check for emotion and sentiment columns
    emotion_col = None
    sentiment_col = Noner
    
    # Common names for emotion and sentiment columns
    emotion_candidates = ['emotion', 'primary_emotion', 'emotion_label']
    sentiment_candidates = ['sentiment', 'sentiment_label', 'sentiment_score', 'positive_rate']
    
    for col in emotion_candidates:
        if col in ground_truth.columns:
            emotion_col = col
            break
            
    for col in sentiment_candidates:
        if col in ground_truth.columns:
            sentiment_col = col
            break
    
    if emotion_col is None or sentiment_col is None:
        raise ValueError("Could not identify emotion or sentiment columns in the Audio_memos file")
    
    # Create base comparison DataFrames
    emotion_comparison = pd.DataFrame()
    sentiment_comparison = pd.DataFrame()
    
    # Add text and ground truth
    emotion_comparison['text'] = ground_truth['text']
    emotion_comparison['ground_truth'] = ground_truth[emotion_col]
    
    sentiment_comparison['text'] = ground_truth['text']
    sentiment_comparison['ground_truth'] = ground_truth[sentiment_col]
    
    # Find model result files in Model_results directory
    results_dir = f"{os.getcwd()}/Model_results"
    model_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    
    # Process emotion model results
    emotion_models = [f.replace('.csv', '') for f in model_files if 'Emotion' in f]
    for model_name in emotion_models:
        model_path = f"{results_dir}/{model_name}.csv"
        
        try:
            model_df = pd.read_csv(model_path)
            
            # Get the top emotion for each text
            emotion_cols = [col for col in model_df.columns if col not in ['iterator', 'text']]
            
            if not emotion_cols:
                print(f"No emotion columns found in {model_name}")
                continue
                
            # Create a column with the highest emotion and its score
            model_df['top_emotion'] = model_df[emotion_cols].idxmax(axis=1)
            model_df['top_emotion_score'] = model_df[emotion_cols].max(axis=1)
            
            # Add to comparison DataFrame
            emotion_comparison[f'{model_name}_prediction'] = model_df['top_emotion']
            emotion_comparison[f'{model_name}_score'] = model_df['top_emotion_score']
            
            # Calculate accuracy (1 if correct, 0 if not)
            emotion_comparison[f'{model_name}_accuracy'] = (
                emotion_comparison['ground_truth'] == emotion_comparison[f'{model_name}_prediction']
            ).astype(int)
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    # Process sentiment model results
    sentiment_models = [f.replace('.csv', '') for f in model_files if 'Sentiment' in f]
    for model_name in sentiment_models:
        model_path = f"{results_dir}/{model_name}.csv"
        
        try:
            model_df = pd.read_csv(model_path)
            
            # Add to comparison DataFrame
            if 'positive_rate' in model_df.columns:
                sentiment_comparison[f'{model_name}_score'] = model_df['positive_rate']
                
                # For sentiment, convert scores to labels if ground truth is categorical
                if ground_truth[sentiment_col].dtype == 'object':
                    # Assuming >50 is positive, otherwise negative
                    sentiment_comparison[f'{model_name}_prediction'] = np.where(
                        model_df['positive_rate'] > 50, 'positive', 'negative'
                    )
                else:
                    # If ground truth is numeric, use raw scores
                    sentiment_comparison[f'{model_name}_prediction'] = model_df['positive_rate']
                
                # Calculate accuracy
                if ground_truth[sentiment_col].dtype == 'object':
                    # For categorical sentiment
                    sentiment_comparison[f'{model_name}_accuracy'] = (
                        sentiment_comparison['ground_truth'] == sentiment_comparison[f'{model_name}_prediction']
                    ).astype(int)
                else:
                    # For numeric sentiment, use a threshold (e.g., difference less than 20)
                    sentiment_comparison[f'{model_name}_accuracy'] = (
                        abs(sentiment_comparison['ground_truth'] - sentiment_comparison[f'{model_name}_prediction']) < 20
                    ).astype(int)
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    # Calculate similarity between model predictions and Claude
    if 'Claude_Emotion' in emotion_models:
        for model_name in [m for m in emotion_models if m != 'Claude_Emotion']:
            emotion_comparison[f'{model_name}_claude_similarity'] = (
                emotion_comparison[f'{model_name}_prediction'] == emotion_comparison['Claude_Emotion_prediction']
            ).astype(int)
    
    if 'Claude_Sentiment' in sentiment_models:
        for model_name in [m for m in sentiment_models if m != 'Claude_Sentiment']:
            if ground_truth[sentiment_col].dtype == 'object':
                # For categorical sentiment
                sentiment_comparison[f'{model_name}_claude_similarity'] = (
                    sentiment_comparison[f'{model_name}_prediction'] == 
                    sentiment_comparison['Claude_Sentiment_prediction']
                ).astype(int)
            else:
                # For numeric sentiment, use a threshold
                sentiment_comparison[f'{model_name}_claude_similarity'] = (
                    abs(sentiment_comparison[f'{model_name}_score'] - 
                        sentiment_comparison['Claude_Sentiment_score']) < 20
                ).astype(int)
    
    # Calculate summary statistics
    emotion_summary = {}
    sentiment_summary = {}
    
    # Emotion accuracy summaries
    for model_name in emotion_models:
        col = f'{model_name}_accuracy'
        if col in emotion_comparison.columns:
            emotion_summary[f'{model_name}_accuracy'] = emotion_comparison[col].mean()
    
    # Sentiment accuracy summaries
    for model_name in sentiment_models:
        col = f'{model_name}_accuracy'
        if col in sentiment_comparison.columns:
            sentiment_summary[f'{model_name}_accuracy'] = sentiment_comparison[col].mean()
    
    # Claude similarity summaries
    if 'Claude_Emotion' in emotion_models:
        for model_name in [m for m in emotion_models if m != 'Claude_Emotion']:
            col = f'{model_name}_claude_similarity'
            if col in emotion_comparison.columns:
                emotion_summary[f'{model_name}_claude_similarity'] = emotion_comparison[col].mean()
    
    if 'Claude_Sentiment' in sentiment_models:
        for model_name in [m for m in sentiment_models if m != 'Claude_Sentiment']:
            col = f'{model_name}_claude_similarity'
            if col in sentiment_comparison.columns:
                sentiment_summary[f'{model_name}_claude_similarity'] = sentiment_comparison[col].mean()
    
    # Save comparison tables
    emotion_comparison.to_csv(f"{results_dir}/emotion_comparison.csv", index=False)
    sentiment_comparison.to_csv(f"{results_dir}/sentiment_comparison.csv", index=False)
    
    # Create summary DataFrames
    emotion_summary_df = pd.DataFrame([emotion_summary])
    sentiment_summary_df = pd.DataFrame([sentiment_summary])
    
    # Save summaries
    emotion_summary_df.to_csv(f"{results_dir}/emotion_summary.csv", index=False)
    sentiment_summary_df.to_csv(f"{results_dir}/sentiment_summary.csv", index=False)
    
    return emotion_comparison, sentiment_comparison, emotion_summary_df, sentiment_summary_df



    
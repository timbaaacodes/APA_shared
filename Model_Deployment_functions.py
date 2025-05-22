import pandas as pd
import numpy as np
import os, sys
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import tensorflow as tf
import json
import gc
import traceback
from tensorflow.keras.layers import TextVectorization

from keras_nlp.models import BertBackbone

import Result_functions as rf

class ModelApplication:

    def __init__(self, model_directory, model_name, tokenizer_config_dir, tokenizer_vocab_dir, debugg_mode = 'Y'):
        self.model_dir = model_directory
        self.model_name = model_name
        self.tokenizer_vocab_dir = tokenizer_vocab_dir
        self.tokenizer_config_dir = tokenizer_config_dir
        self.debugg_mode = debugg_mode

    def tokenize_strings(self, text_vectorization, input_string):
        """
        Tokenizes the input string using the TextVectorization layer.
        """
        return text_vectorization(tf.constant([input_string]))

    def get_tokenizer(self):
        """
        Loads the model and recreates the TextVectorization layer using the config and vocab files.
        """
        
        def load_config_and_vocab(config_path, vocab_path):
            try:
                with open(self.tokenizer_config_dir, 'r') as f:
                    config = json.load(f)
                with open(self.tokenizer_vocab_dir, 'r') as f:
                    vocab = json.load(f)
                return config, vocab
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise ValueError(f"Error loading config or vocab: {e}")

        model_name_string, _ = self.model_name.split('_', 1)

        if model_name_string == 'Binary2gram':
            if self.debugg_mode =='Y':
                print(f'Imported tokenizer for {model_name_string}')

            config, vocab = load_config_and_vocab(self.tokenizer_config_dir, self.tokenizer_vocab_dir)
            text_vectorization = tf.keras.layers.TextVectorization(
                max_tokens=config.get("max_tokens"),
                standardize=config.get("standardize"),
                split=config.get("split"),
                ngrams=config.get("ngrams"),
                output_mode=config.get("output_mode"),
                output_sequence_length=config.get("output_sequence_length"),
                pad_to_max_tokens=config.get("pad_to_max_tokens"),
            )
            text_vectorization.set_vocabulary(vocab)
            return text_vectorization

        elif  model_name_string == 'transformerEmotion' or model_name_string == 'transformerSentiment':
            

            config, vocab = load_config_and_vocab(self.tokenizer_config_dir, self.tokenizer_vocab_dir)
            
            text_vectorization = tf.keras.layers.TextVectorization(
                max_tokens=config.get("max_tokens", 10000),
                standardize=config.get("standardize", "lower_and_strip_punctuation"),
                split=config.get("split", "whitespace"),
                ngrams=config.get("ngrams", None),
                output_mode=config.get("output_mode", "int"),
                output_sequence_length=config.get("output_sequence_length", 250),
                pad_to_max_tokens=config.get("pad_to_max_tokens", False),
                sparse=config.get("sparse", False),
                ragged=config.get("ragged", False),
                encoding=config.get("encoding", "utf-8"),
            )
            text_vectorization.set_vocabulary(vocab)

            if self.debugg_mode =='Y':
                print(f'Imported tokenizer for {model_name_string}')
            return text_vectorization

        elif model_name_string in ['gloveEmotion', 'gloveSentiment']:
           if self.debugg_mode =='Y':
            print(f'Imported tokenizer for {model_name_string}')

            config, vocab = load_config_and_vocab(self.tokenizer_config_dir, self.tokenizer_vocab_dir)
            tokenizer_layer = tf.keras.layers.TextVectorization(
                max_tokens=config.get("max_tokens"),
                standardize=config.get("standardize", "lower_and_strip_punctuation"),
                split=config.get("split", "whitespace"),
                ngrams=config.get("ngrams"),
                output_mode=config.get("output_mode", "int"),
                output_sequence_length=config.get("output_sequence_length", 100),
                pad_to_max_tokens=config.get("pad_to_max_tokens", False),
            )
            
            tokenizer_layer.set_vocabulary(config.get("vocabulary", vocab))
            
            return tokenizer_layer

        elif  model_name_string == 'bertEmotion' or model_name_string == 'bertSentiment':
            print(f'Imported tokenizer for {model_name_string}')

            #glove tokenizer

            raise NotImplementedError("Transformer tokenizer logic is not implemented.")

        elif  model_name_string == 'LSTMEmotion' or model_name_string == 'LSTMSentiment':
            print(f'Imported tokenizer for {model_name_string}')

            raise NotImplementedError("Transformer tokenizer logic is not implemented.")

    def get_model(self):

        # TESTING
        def get_positional_encoding(seq_length, d_model):
            # Calculate positional encoding
            positions = np.arange(seq_length)[:, np.newaxis]
            depths = np.arange(d_model)[np.newaxis, :] // 2 * 2  # Integer division

            # Create angle rates
            angle_rates = 1 / np.power(10000, (2 * (depths // 2)) / np.float32(d_model))
            angle_rads = positions * angle_rates

            # Apply sin/cos to even/odd indices
            pos_encoding = np.zeros(angle_rads.shape)
            pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
            pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

            return tf.cast(pos_encoding, dtype=tf.float32)

        #TESTING

        # Create ifs for different model types


        model_name_string, _ = self.model_name.split('_', 1)

        if model_name_string == 'Binary2gram':
            if self.debugg_mode =='Y':
                print(f'Imported model for {model_name_string}')


            # Load the Binary2gram model
            model = load_model(self.model_dir)
            
            return model

        elif model_name_string == 'transformerEmotion' or model_name_string == 'transformerSentiment':
            
            print(f'Importing model for {model_name_string}')
            
            model = load_model(self.model_dir)
            print(model)
            
            if self.debugg_mode =='Y':
                print(f'Imported model for {model_name_string}')

            return model

        elif model_name_string in ['gloveEmotion', 'gloveSentiment']:
            if self.debugg_mode =='Y':
                print(f'Imported tokenizer for {model_name_string}')

            import tensorflow as tf
            from tensorflow.keras import initializers

            # Define the custom embedding initializer
            class CustomEmbeddingInitializer(initializers.Initializer):
                def __init__(self, embedding_matrix):
                    self.embedding_matrix = embedding_matrix

                def __call__(self, shape, dtype=None):
                    return self.embedding_matrix

            # Custom NotEqual Layer (if needed in your model)
            class NotEqual(tf.keras.layers.Layer):
                def call(self, inputs):
                    return tf.not_equal(inputs[0], inputs[1])

            # Function to create the embedding layer
            def create_embedding_layer(embedding_matrix, max_tokens, embedding_dim):
                return layers.Embedding(
                    input_dim=max_tokens,
                    output_dim=embedding_dim,
                    embeddings_initializer=CustomEmbeddingInitializer(embedding_matrix),
                    trainable=False,
                    mask_zero=True,
                )
            # Load the pre-trained Keras model file
            model_path = self.model_dir  # Ensure this points to the correct saved model file
            try:
                # Register custom objects if needed
                # Load the model with custom objects
                model = load_model(
                    model_path,
                    custom_objects={
                        "CustomEmbeddingInitializer": CustomEmbeddingInitializer,
                        "NotEqual": NotEqual
                    }
                )
                print(f"GloVe model successfully loaded from {model_path}.")
            except Exception as e:
                raise ValueError(f"Error loading GloVe model from {model_path}: {e}")

            return model

            
        elif  model_name_string == 'bertEmotion' or model_name_string == 'bertSentiment':
            if self.debugg_mode =='Y':
                print(f'Imported tokenizer for {model_name_string}')

            #glove tokenizer

            raise NotImplementedError("Transformer tokenizer logic is not implemented.")

        elif  model_name_string == 'LSTMEmotion' or model_name_string == 'LSTMSentiment':
            print(f'Imported tokenizer for {model_name_string}')

            #glove tokenizer

            raise NotImplementedError("Transformer tokenizer logic is not implemented.")






    def predict_emotion(self, untokenized_string):
        """
        Predicts the classification for an untokenized input string.
        """
        # Load the model and TextVectorization layer
        text_vectorization = self.get_tokenizer()

        model = self.get_model()

        # Define the input layer
        inputs = keras.Input(shape=(1,), dtype="string")
        processed_inputs = text_vectorization(inputs)
        outputs = model(processed_inputs)
        inference_model = keras.Model(inputs, outputs)

        # Convert the input string to a tensor

        raw_text_data = tf.convert_to_tensor([untokenized_string])
        
        # Get prediction
        predictions = inference_model(raw_text_data)

        #print(f'Predictions: {predictions}')

        emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
            "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
            "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
            "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
        ]

        emotion_results = {label: float(prob) * 100 for label, prob in zip(emotion_labels, predictions[0])}
    
    
        # Display results
        if self.debugg_mode == 'Y':
            print(f'''===== Emotion results ====== ''')
            print(f'''Input: {raw_text_data} ''')
            for label, prob in zip(emotion_labels, predictions[0]):
                print(f"{label:15}: {float(prob) * 100:.2f}%")

        untokenized_string = None

        return emotion_results


    def predict_sentiment(self, untokenized_string):
        """
        Predicts the sentiment for an untokenized input string.
        """
        # Load the model and TextVectorization layer
        text_vectorization = self.get_tokenizer()
        model = self.get_model()
        # Define the input layer

        inputs = keras.Input(shape=(1,), dtype="string")
        processed_inputs = text_vectorization(inputs)
        outputs = model(processed_inputs)
        inference_model = keras.Model(inputs, outputs)

        # Convert the input string to a tensor
        raw_text_data = tf.convert_to_tensor([untokenized_string])
        
        #print(f'Raw text: {raw_text_data}')

        # Get prediction
        predictions = inference_model(raw_text_data)
        
        if self.debugg_mode == 'Y':
            print(f"{float(predictions[0] * 100):.2f} percent positive")

        untokenized_string = None
        
        positive_rate = int(predictions.numpy()[0] * 100) / 100.0

        return positive_rate


def create_model_instances(model_type = 'Emotion', model_name_list = ['Binary2gram'], debugg_mode = 'N'):    
    model_id = []
    instances = []

    # Create working directories
    for model_name in model_name_list:

        #model_directory = f'{os.getcwd()}/{model_type}_Model/{model_name}/{model_name}.keras'
        model_directory = f'{os.getcwd()}/{model_type}_Model/{model_name}/{model_name}.h5'
        tokenizer_config_directory = f'{os.getcwd()}/{model_type}_Model/{model_name}/{model_name}_config.json'
        tokenizer_vocab_directory = f'{os.getcwd()}/{model_type}_Model/{model_name}/{model_name}_vocab.json'

        model_name_type = f'{model_name}_{model_type}'

        model_id.append([model_name_type, model_directory, tokenizer_config_directory, tokenizer_vocab_directory])

    print(f'''===========
Stage 1 done! Created {len(model_id)} directories! Model instances can be created.
Created directories : {model_id}
===========
          ''')
    
    # Create instances for each model in model_id
    for model in model_id:
        # Pass both config and vocab files to ModelApplication
        instance = ModelApplication(
            model_directory=model[1],
            model_name=model[0],
            tokenizer_config_dir=model[2],
            tokenizer_vocab_dir=model[3],
            debugg_mode = debugg_mode
        )
        instances.append(instance)
        
    instance_dict = {model[0]: instance for model, instance in zip(model_id, instances)}
    print(f'''===========
Stage 2 done! Created {len(instances)} model application instances! Tokenzing and predicting can start!
===========
''')
    return instance_dict


def deploy_model(instance, audio_summary, model_name, unvectorized_text = 'I hate this film. I am really upset' ):
        
    # Split model_name for further usage
    model_name_string, model_type = model_name.split('_', 1)
    #print(f'Model Name: {model_name_string}, Model Category: {model_type}') DEBUGGING

    #model = instance.get_model() 
    #text_vectorizer = instance.get_tokenizer()

    if model_type == "Emotion":
    
        emotion_result_dict = instance.predict_emotion(unvectorized_text)

        #rf.store_emotion_results(emotion_result_dict, audio_summary, model_name)

        return emotion_result_dict
        
    elif model_type == 'Sentiment':

        sentiment_result = instance.predict_sentiment(unvectorized_text)
        
        return sentiment_result


def full_deployment(instances_emotion, transcribed_texts, audio_summary):

    print(f'Running classification tasks')    

    emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
            "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
            "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
            "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
        ]

    #results_df = pd.DataFrame(columns=['iterator', 'text'] + emotion_labels).set_index('iterator')

    for model_name, instance in instances_emotion.items():

        print(f'Running model: {model_name}')

        model_name_string, model_type = model_name.split('_', 1)

        #print(f'Model Name: {model_name_string}, Model Category: {model_type}') DEBUGGING

        if model_type == "Emotion":

            results_df = pd.DataFrame(columns=['iterator', 'text'] + emotion_labels).set_index('iterator')

            for iterator, texts in enumerate(transcribed_texts):
                
                # Deploy model and store results in 
                emotion = deploy_model(instance=instance, audio_summary = audio_summary, model_name = model_name, unvectorized_text = texts)
                
                new_row = pd.DataFrame([[iterator + 1, texts] + [emotion[label] for label in emotion_labels]], 
                                    columns=['iterator', 'text'] + emotion_labels)
                
                new_row = new_row.set_index('iterator')
                
                results_df = pd.concat([results_df, new_row])
                
                tf.keras.backend.clear_session()
                gc.collect()
            
            rf.store_model_results(model_results = results_df,model_name=model_name)

        elif model_type == 'Sentiment':

            #print(f'Model Name: {model_name_string}, Model Category: {model_type}') DEBUGGING

            results_df = pd.DataFrame(columns=['iterator', 'text', 'positive_rate']).set_index('iterator')

            for iterator, texts in enumerate(transcribed_texts):
                
                # Deploy model and store results in 
                sentiment_result = deploy_model(instance=instance, unvectorized_text = texts, audio_summary = audio_summary, model_name = model_name)
                
                new_row = pd.DataFrame([[iterator + 1, texts, sentiment_result]], 
                                    columns=['iterator', 'text', 'positive_rate'])
                
                new_row = new_row.set_index('iterator')
                
                results_df = pd.concat([results_df, new_row])
                
                tf.keras.backend.clear_session()
                
                gc.collect()
                
            rf.store_model_results(model_results = results_df,model_name=model_name)

    print('Model run completed.')



    
        
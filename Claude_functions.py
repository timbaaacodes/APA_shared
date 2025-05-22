import os
import json
import re
from typing import Dict, Any, List, Optional, Union

# LangChain imports
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.callbacks import get_openai_callback, CallbackManager
from langchain_core.tracers import ConsoleTracer
from langchain_core.pydantic_v1 import BaseModel, Field

# Cost tracking
class CostTracker:
    """Track API costs across multiple calls"""
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        
    def update(self, tokens_used, cost, prompt_tokens, completion_tokens):
        self.total_tokens += tokens_used
        self.total_cost += cost
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        
    def get_summary(self):
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_cost": self.total_cost,
            "calls": self.calls,
            "avg_cost_per_call": self.total_cost / max(1, self.calls)
        }

# Output parsers
class EmotionAnalysis(BaseModel):
    """Schema for emotion analysis output"""
    admiration: float = Field(description="Admiration score from 0-100")
    amusement: float = Field(description="Amusement score from 0-100")
    anger: float = Field(description="Anger score from 0-100")
    annoyance: float = Field(description="Annoyance score from 0-100")
    approval: float = Field(description="Approval score from 0-100")
    caring: float = Field(description="Caring score from 0-100")
    confusion: float = Field(description="Confusion score from 0-100")
    curiosity: float = Field(description="Curiosity score from 0-100")
    desire: float = Field(description="Desire score from 0-100")
    disappointment: float = Field(description="Disappointment score from 0-100")
    disapproval: float = Field(description="Disapproval score from 0-100")
    disgust: float = Field(description="Disgust score from 0-100")
    embarrassment: float = Field(description="Embarrassment score from 0-100")
    excitement: float = Field(description="Excitement score from 0-100")
    fear: float = Field(description="Fear score from 0-100")
    gratitude: float = Field(description="Gratitude score from 0-100")
    grief: float = Field(description="Grief score from 0-100")
    joy: float = Field(description="Joy score from 0-100")
    love: float = Field(description="Love score from 0-100")
    nervousness: float = Field(description="Nervousness score from 0-100")
    optimism: float = Field(description="Optimism score from 0-100")
    pride: float = Field(description="Pride score from 0-100")
    realization: float = Field(description="Realization score from 0-100")
    relief: float = Field(description="Relief score from 0-100")
    remorse: float = Field(description="Remorse score from 0-100")
    sadness: float = Field(description="Sadness score from 0-100")
    surprise: float = Field(description="Surprise score from 0-100")
    neutral: float = Field(description="Neutral score from 0-100")

class ClaudeAnalyzer:
    """
    Class for performing emotion and sentiment analysis using Claude via LangChain.
    
    This class provides methods to analyze text for emotional content and sentiment,
    with detailed tracking of API usage and costs.
    """
    
    def __init__(self, api_key=None, model_name="claude-3-opus-20240229", debug_mode='N'):
        """
        Initialize the Claude Analyzer with LangChain.
        
        Parameters:
        -----------
        api_key : str, optional
            Anthropic API key. If None, attempts to get from environment.
        model_name : str, optional
            The Claude model to use. Default is claude-3-opus-20240229.
        debug_mode : str, optional
            Whether to print debug information ('Y' or 'N'). Default is 'N'.
        """
        # Set API key
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set ANTHROPIC_API_KEY environment variable or pass api_key.")
        
        self.model_name = model_name
        self.debug_mode = debug_mode
        
        # Setup LangChain components
        self.llm = ChatAnthropic(
            model=self.model_name,
            anthropic_api_key=self.api_key,
            temperature=0
        )
        
        # Initialize cost tracker
        self.cost_tracker = CostTracker()
        
        # Setup emotion prompt template
        self.emotion_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an expert in emotion analysis. Provide accurate numeric ratings for emotions in text."),
            HumanMessage(content="""
            Analyze the emotional content in this text. Rate each emotion on a scale from 0 to 100, 
            where 0 means the emotion is not present and 100 means the emotion is very strongly present.
            
            Text: "{text}"
            
            Rate each of these emotions:
            {emotion_list}
            
            Provide your response in a JSON format with emotion names as keys and numeric scores as values.
            For example: {{"joy": 85.5, "sadness": 0.0, ...}}
            """)
        ])
        
        # Setup sentiment prompt template
        self.sentiment_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an expert in sentiment analysis. Provide a single numeric rating."),
            HumanMessage(content="""
            Analyze the sentiment in this text on a scale from 0 to 100, where:
            - 0 is extremely negative
            - 50 is neutral
            - 100 is extremely positive
            
            Text: "{text}"
            
            Provide ONLY a single number between 0 and 100 as your response.
            """)
        ])
        
        # Setup output parsers
        self.json_parser = JsonOutputParser(pydantic_object=EmotionAnalysis)
        self.str_parser = StrOutputParser()
        
        # Create chains
        self.emotion_chain = self.emotion_template | self.llm | self.json_parser
        self.sentiment_chain = self.sentiment_template | self.llm | self.str_parser
        
        # List of emotions to analyze
        self.emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
            "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
            "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
            "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
        ]
        
        # Document the prompt designs and their purposes
        self.prompt_documentation = {
            "emotion_prompt": {
                "purpose": "Analyze text for 28 distinct emotions with fine-grained scores",
                "design_choices": [
                    "Uses scale from 0-100 for more granular emotion detection",
                    "Requests structured JSON output for easy parsing",
                    "Lists all emotions to ensure comprehensive analysis",
                    "Uses system message to prime Claude for expert emotion analysis"
                ],
                "output_format": "JSON with emotion names as keys and scores as values"
            },
            "sentiment_prompt": {
                "purpose": "Determine overall sentiment polarity and intensity",
                "design_choices": [
                    "Uses 0-100 scale instead of -1 to 1 for consistency with emotion scores",
                    "Defines exact meaning of scale endpoints (0=extremely negative, 100=extremely positive)",
                    "Explicitly requests only a number to avoid extra text in response",
                    "Uses system message to prime Claude for expert sentiment analysis"
                ],
                "output_format": "Single numeric value between 0 and 100"
            }
        }
        
    def predict_emotion(self, text: str) -> Dict[str, float]:
        """
        Analyze the emotional content of the given text.
        
        Parameters:
        -----------
        text : str
            The text to analyze.
            
        Returns:
        --------
        dict
            Dictionary with emotion names as keys and scores (0-100) as values.
        """
        # Check for empty or invalid text
        if not text or not isinstance(text, str):
            if self.debug_mode == 'Y':
                print("Warning: Empty or invalid text provided to predict_emotion")
            return {emotion: 0.0 for emotion in self.emotion_labels}
        
        try:
            # Use callback manager to track token usage and cost
            with get_openai_callback() as cb:
                # Run the emotion chain
                result = self.emotion_chain.invoke({
                    "text": text,
                    "emotion_list": ", ".join(self.emotion_labels)
                })
                
                # Update cost tracker
                self.cost_tracker.update(
                    tokens_used=cb.total_tokens,
                    cost=cb.total_cost,
                    prompt_tokens=cb.prompt_tokens,
                    completion_tokens=cb.completion_tokens
                )
                
                # Convert to dictionary
                emotion_results = {k: float(v) for k, v in result.dict().items()}
                
                if self.debug_mode == 'Y':
                    print(f"===== Emotion results ======")
                    print(f"Input: {text}")
                    for label, prob in emotion_results.items():
                        print(f"{label:15}: {prob:.2f}%")
                    print(f"Tokens used: {cb.total_tokens}, Cost: ${cb.total_cost:.6f}")
                
                return emotion_results
                
        except Exception as e:
            if self.debug_mode == 'Y':
                print(f"Error in predict_emotion: {e}")
            # Return default values if analysis fails
            return {emotion: 0.0 for emotion in self.emotion_labels}
    
    def predict_sentiment(self, text: str) -> float:
        """
        Analyze the sentiment of the given text.
        
        Parameters:
        -----------
        text : str
            The text to analyze.
            
        Returns:
        --------
        float
            Sentiment score between 0 (very negative) and 100 (very positive).
        """
        # Check for empty or invalid text
        if not text or not isinstance(text, str):
            if self.debug_mode == 'Y':
                print("Warning: Empty or invalid text provided to predict_sentiment")
            return 50.0  # Neutral sentiment for invalid input
        
        try:
            # Use callback manager to track token usage and cost
            with get_openai_callback() as cb:
                # Run the sentiment chain
                result = self.sentiment_chain.invoke({"text": text})
                
                # Update cost tracker
                self.cost_tracker.update(
                    tokens_used=cb.total_tokens,
                    cost=cb.total_cost,
                    prompt_tokens=cb.prompt_tokens,
                    completion_tokens=cb.completion_tokens
                )
                
                # Extract the numeric sentiment value using regex
                match = re.search(r'(\d+(\.\d+)?)', result)
                if match:
                    sentiment_score = float(match.group(1))
                else:
                    sentiment_score = 50.0  # Default to neutral if parsing fails
                
                if self.debug_mode == 'Y':
                    print(f"{sentiment_score:.2f} percent positive")
                    print(f"Tokens used: {cb.total_tokens}, Cost: ${cb.total_cost:.6f}")
                
                return sentiment_score
                
        except Exception as e:
            if self.debug_mode == 'Y':
                print(f"Error in predict_sentiment: {e}")
            return 50.0  # Default to neutral sentiment if analysis fails
    
    def get_cost_summary(self):
        """
        Get a summary of API usage and costs.
        
        Returns:
        --------
        dict
            Dictionary with usage statistics.
        """
        return self.cost_tracker.get_summary()
    
    def get_prompt_documentation(self):
        """
        Get documentation about prompt designs and their effectiveness.
        
        Returns:
        --------
        dict
            Dictionary with prompt documentation.
        """
        return self.prompt_documentation

def create_claude_instance(api_key=None, debugg_mode='N'):
    """
    Create and return a Claude analyzer instance.
    
    Parameters:
    -----------
    api_key : str, optional
        Anthropic API key. If None, attempts to get from environment.
    debugg_mode : str, optional
        Whether to print debug information ('Y' or 'N'). Default is 'N'.
        
    Returns:
    --------
    ClaudeAnalyzer
        Initialized Claude analyzer instance.
    """
    return ClaudeAnalyzer(api_key=api_key, debug_mode=debugg_mode)
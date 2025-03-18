"""OpenAI model implementation."""
from typing import Any, Dict, List, Optional, Union

import openai
from pydantic import BaseModel

from lm_evals.core.model import BaseModel as LMModel

class OpenAIConfig(BaseModel):
    """Configuration for OpenAI models."""
    model: str = "gpt-4"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class OpenAIModel(LMModel):
    """OpenAI model implementation."""
    
    def __init__(self, **kwargs):
        """Initialize OpenAI model.
        
        Args:
            **kwargs: Configuration parameters
        """
        super().__init__(**kwargs)
        self.config = OpenAIConfig(**kwargs)
        
        if self.config.api_key:
            openai.api_key = self.config.api_key
            
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
    ) -> str:
        """Generate text using OpenAI model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequence(s)
            
        Returns:
            Generated text
        """
        response = openai.ChatCompletion.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty
        )
        return response.choices[0].message.content.strip()
    
    def get_logprobs(
        self,
        text: str,
        tokens: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Get token log probabilities.
        
        Args:
            text: Input text
            tokens: Optional specific tokens to get probabilities for
            
        Returns:
            Dictionary mapping tokens to log probabilities
        """
        # Note: OpenAI API doesn't provide direct logprob access
        # This is a placeholder implementation
        raise NotImplementedError(
            "Log probabilities not available through OpenAI API"
        )
    
    @property
    def model_name(self) -> str:
        return f"openai-{self.config.model}" 

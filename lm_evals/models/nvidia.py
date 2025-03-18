"""NVIDIA NIM model implementation."""
from typing import Any, Dict, List, Optional, Union

import numpy as np
from nvidia_nim import NimClient, NimConfig
from pydantic import BaseModel

from lm_evals.core.model import BaseModel as LMModel

class NIMConfig(BaseModel):
    """Configuration for NVIDIA NIM models."""
    model_name: str
    endpoint_url: str
    api_key: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 1.0
    stop_sequences: List[str] = []
    timeout: float = 30.0

class NIMModel(LMModel):
    """NVIDIA NIM model implementation."""
    
    def __init__(self, **kwargs):
        """Initialize NVIDIA NIM model.
        
        Args:
            **kwargs: Configuration parameters including:
                - model_name: Name of the deployed model
                - endpoint_url: URL of the NIM endpoint
                - api_key: Optional API key for authentication
                - max_tokens: Maximum tokens to generate
                - temperature: Sampling temperature
                - top_p: Nucleus sampling parameter
                - stop_sequences: List of stop sequences
                - timeout: Request timeout in seconds
        """
        super().__init__(**kwargs)
        self.config = NIMConfig(**kwargs)
        
        # Initialize NIM client
        nim_config = NimConfig(
            endpoint_url=self.config.endpoint_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout
        )
        self.client = NimClient(nim_config)
        
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
    ) -> str:
        """Generate text using NVIDIA NIM model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequence(s)
            
        Returns:
            Generated text
        """
        # Prepare request parameters
        request_params = {
            "prompt": prompt,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_sequences": (
                [stop] if isinstance(stop, str)
                else stop if stop
                else self.config.stop_sequences
            )
        }
        
        # Make request to NIM endpoint
        response = self.client.generate(
            model_name=self.config.model_name,
            **request_params
        )
        
        # Extract and return generated text
        return response.text.strip()
    
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
        # Make request to NIM endpoint with logprobs enabled
        response = self.client.generate(
            model_name=self.config.model_name,
            prompt=text,
            max_tokens=0,  # We only want logprobs of input
            logprobs=True,
            tokens=tokens
        )
        
        # Extract and return token logprobs
        return {
            token: logprob
            for token, logprob in zip(
                response.token_logprobs.tokens,
                response.token_logprobs.logprobs
            )
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        response = self.client.tokenize(
            model_name=self.config.model_name,
            text=text
        )
        return response.tokens
    
    @property
    def model_name(self) -> str:
        return f"nvidia-nim-{self.config.model_name}" 

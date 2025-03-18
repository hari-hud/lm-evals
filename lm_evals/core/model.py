"""Base model interface for language model evaluation."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

class BaseModel(ABC):
    """Abstract base class for all language models."""
    
    def __init__(self, **kwargs):
        """Initialize the model with optional configuration."""
        self.config = kwargs
        
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
    ) -> str:
        """Generate text completion for the given prompt.
        
        Args:
            prompt: Input text to complete
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            stop: Stop sequence(s) to end generation
            
        Returns:
            Generated text completion
        """
        pass
    
    @abstractmethod
    def get_logprobs(
        self,
        text: str,
        tokens: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Get log probabilities for tokens in the text.
        
        Args:
            text: Input text to analyze
            tokens: Optional specific tokens to get probabilities for
            
        Returns:
            Dictionary mapping tokens to their log probabilities
        """
        pass
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        raise NotImplementedError(
            "Tokenization not implemented for this model"
        )
    
    @property
    def model_name(self) -> str:
        """Get the name/identifier of this model."""
        return self.__class__.__name__ 

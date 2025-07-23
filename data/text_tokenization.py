from transformers import AutoTokenizer

import numpy as np


class BertTokenizer:
    def __init__(self, max_length: int) -> None:
        r"""
        Args:
            max_length: pad or truncate sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

    def __call__(self, x: str) -> np.ndarray:
        r"""Convert texts to token IDs. 
        
        Args:
            x: str, e.g., "Hello world."

        Outputs:
            x: ndarray, e.g., [101, 8667, 1362,  119,  102, 0, 0]
        """
        
        x = self.tokenizer.encode(
            text=x, 
            padding="max_length",
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="np",
        )[0]  
        # shape: (max_length,). E.g., [101, 8667, 1362,  119,  102, 0, 0]
        
        return x

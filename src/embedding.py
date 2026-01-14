from sentence_transformers import SentenceTransformer
from PIL import Image
from typing import List, Union
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """
    Handles generation of vector embeddings for images and text using CLIP.
    """
    
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """
        Initialize the EmbeddingEngine with a specific Sentence Transformer model.
        
        Args:
            model_name (str): Name of the model to load. Default is 'clip-ViT-B-32'.
        """
        try:
            logger.info(f"Loading embedding model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def encode_images(self, images: List[Image.Image]) -> List[List[float]]:
        """
        Batches images and returns normalized vector embeddings.
        
        Args:
            images (List[Image.Image]): List of PIL images.
            
        Returns:
            List[List[float]]: List of vector embeddings (floats).
        """
        if not images:
            return []
            
        try:
            # sentence-transformers encode method handles batching automatically
            # normalize_embeddings=True ensures vectors are unit length (good for cosine similarity)
            embeddings = self.model.encode(images, batch_size=32, normalize_embeddings=True)
            
            # Convert numpy array to list of lists
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error encoding images: {e}")
            return []

    def encode_text(self, text_list: List[str]) -> List[List[float]]:
        """
        Converts text queries or transcripts into vector embeddings.
        
        Args:
            text_list (List[str]): List of text strings.
            
        Returns:
            List[List[float]]: List of vector embeddings (floats).
        """
        if not text_list:
            return []
            
        try:
            embeddings = self.model.encode(text_list, batch_size=32, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return []

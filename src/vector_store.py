from typing import List, Dict, Any, Optional, Union
import uuid
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
# Import EmbeddingEngine for type hinting if needed, but avoid circular imports at runtime
# from .embedding import EmbeddingEngine 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages interactions with the Qdrant Vector Database.
    """
    
    def __init__(self, collection_name: str = "neuro_stream", embedding_engine = None, host: str = "localhost", port: int = 6333, path: str = None):
        """
        Initialize the Qdrant client connection and setup the collection.
        
        Args:
            collection_name (str): Name of the Qdrant collection.
            embedding_engine: Instance of EmbeddingEngine (needed for search).
            host (str): Qdrant host (used if path is None).
            port (int): Qdrant port (used if path is None).
            path (str): Path for local Qdrant storage (overrides host/port).
        """
        self.collection_name = collection_name
        self.embedding_engine = embedding_engine
        
        try:
            if path:
                self.client = QdrantClient(path=path)
                logger.info(f"Connected to Qdrant (Local) at {path}")
            else:
                self.client = QdrantClient(host=host, port=port)
                logger.info(f"Connected to Qdrant (Server) at {host}:{port}")
            
            self._ensure_collection_exists()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def _ensure_collection_exists(self):
        """Creates the collection if it doesn't represent."""
        try:
            collections = self.client.get_collections()
            exists = any(c.name == self.collection_name for c in collections.collections)
            
            if not exists:
                logger.info(f"Creating collection '{self.collection_name}'...")
                # CLIP ViT-B-32 output dimension is 512
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
                )
                logger.info("Collection created.")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Error checking/creating collection: {e}")

    def index_video(self, video_id: str, visual_vectors: List[Dict[str, Any]], audio_vectors: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """
        Uploads vectors to the Qdrant collection.
        
        Args:
            video_id (str): Unique identifier for the video.
            visual_vectors (List[Dict]): List of dicts containing 'vector' and 'timestamp'.
            audio_vectors (List[Dict]): List of dicts containing 'vector', 'start', 'end', 'text'.
            metadata (Dict): Video-level metadata (e.g., filename, duration).
        """
        points = []
        
        # Process visual vectors
        for item in visual_vectors:
            vector = item.get("vector")
            timestamp = item.get("timestamp")
            
            if vector:
                payload = {
                    "video_id": video_id,
                    "type": "visual",
                    "timestamp": timestamp,
                    **metadata
                }
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                ))
        
        # Process audio vectors
        for item in audio_vectors:
            vector = item.get("vector")
            
            if vector:
                payload = {
                    "video_id": video_id,
                    "type": "audio",
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "text": item.get("text"),
                    # Use start time as the primary timestamp for unified search
                    "timestamp": item.get("start"),
                    **metadata
                }
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                ))
        
        if points:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Indexed {len(points)} points for video {video_id}")
            except Exception as e:
                logger.error(f"Failed to upsert points: {e}")
        else:
            logger.warning("No valid points to index.")

    def search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Converts query to vector, searches Qdrant, and returns matching timestamps.
        
        Args:
            query_text (str): The search query.
            top_k (int): Number of results to return.
            
        Returns:
            List[Dict]: Search results containing timestamps and metadata.
        """
        if not self.embedding_engine:
            raise ValueError("EmbeddingEngine not initialized in VectorStore. Cannot encode query.")

        try:
            # 1. Convert text to vector
            query_vector_batch = self.embedding_engine.encode_text([query_text])
            if not query_vector_batch:
                return []
            query_vector = query_vector_batch[0]
            
            # 2. Search Qdrant
            # search() is deprecated/removed in newer qdrant-client versions. using query_points.
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            ).points
            
            # 3. Format results
            results = []
            for hit in search_result:
                results.append({
                    "score": hit.score,
                    "timestamp": hit.payload.get("timestamp"),
                    "video_id": hit.payload.get("video_id"),
                    "type": hit.payload.get("type"),
                    "text": hit.payload.get("text", None) # Include text if it's an audio match
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
import shutil
import os
import uuid
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from src.processor import VideoProcessor
from src.embedding import EmbeddingEngine
from src.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Global instances
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    logger.info("Loading models...")
    app_state["processor"] = VideoProcessor(whisper_model_size="base")
    app_state["embedding_engine"] = EmbeddingEngine() # Loads CLIP
    # Use local path for simple setup without Docker
    app_state["vector_store"] = VectorStore(
        embedding_engine=app_state["embedding_engine"], 
        path="./qdrant_local_store"
    )
    logger.info("Models loaded and dependencies initialized.")
    yield
    # Shutdown: Clean up if necessary
    app_state.clear()

app = FastAPI(title="Neuro-Stream API", lifespan=lifespan)

class SearchResult(BaseModel):
    timestamp: float
    score: float
    type: str # 'visual' or 'audio'
    text: Optional[str] = None
    video_id: str

class IngestResponse(BaseModel):
    video_id: str
    message: str
    frames_processed: int
    audio_segments_processed: int

def process_video_task(file_path: str, video_id: str):
    """
    Background task to process the video so the API doesn't block.
    """
    try:
        processor: VideoProcessor = app_state["processor"]
        embedding_engine: EmbeddingEngine = app_state["embedding_engine"]
        vector_store: VectorStore = app_state["vector_store"]

        logger.info(f"Starting processing for {video_id}...")

        # 1. Extract
        frames = processor.extract_frames(file_path)
        audio_segments = processor.extract_audio_segments(file_path)

        # 2. Embed Visual
        visual_vectors_data = []
        if frames:
            pil_images = [f['image'] for f in frames]
            visual_embeddings = embedding_engine.encode_images(pil_images)
            for i, frame in enumerate(frames):
                visual_vectors_data.append({
                    "vector": visual_embeddings[i],
                    "timestamp": frame['timestamp']
                })

        # 3. Embed Audio
        audio_vectors_data = []
        if audio_segments:
            texts = [s['text'] for s in audio_segments]
            audio_embeddings = embedding_engine.encode_text(texts)
            for i, segment in enumerate(audio_segments):
                audio_vectors_data.append({
                    "vector": audio_embeddings[i],
                    "start": segment['start'],
                    "end": segment['end'],
                    "text": segment['text']
                })

        # 4. Index
        vector_store.index_video(
            video_id=video_id,
            visual_vectors=visual_vectors_data,
            audio_vectors=audio_vectors_data,
            metadata={"filename": os.path.basename(file_path)}
        )
        logger.info(f"Completed processing for {video_id}")

        # Cleanup uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a video file to be processed in the background.
    """
    video_id = str(uuid.uuid4())
    file_location = f"temp_{video_id}_{file.filename}"
    
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    # Add processing to background tasks
    background_tasks.add_task(process_video_task, file_location, video_id)

    return IngestResponse(
        video_id=video_id,
        message="Video accepted for background processing.",
        frames_processed=0, # Placeholder as it's async
        audio_segments_processed=0
    )

@app.get("/search", response_model=List[SearchResult])
async def search_video(q: str = Query(..., description="The search query"), top_k: int = 5):
    """
    Search for moments in indexed videos matching the query.
    """
    vector_store: VectorStore = app_state.get("vector_store")
    if not vector_store:
        raise HTTPException(status_code=503, detail="System initializing, please try again later.")

    results = vector_store.search(q, top_k=top_k)
    
    response = []
    for res in results:
        response.append(SearchResult(
            timestamp=res['timestamp'],
            score=res['score'],
            type=res['type'],
            text=res['text'],
            video_id=res['video_id']
        ))
    
    return response

@app.get("/")
def health_check():
    return {"status": "running", "models_loaded": "processor" in app_state}

import os
from src.processor import VideoProcessor
from src.embedding import EmbeddingEngine
from src.vector_store import VectorStore

def main():
    # Configuration
    VIDEO_PATH = "sample_video_2.mp4" # Replace with your video path
    VIDEO_ID = "vid_002"
    
    # Check if a video file exists to avoid immediate crash in demo
    if not os.path.exists(VIDEO_PATH):
        print(f"Please place a video file at '{VIDEO_PATH}' to run the pipeline.")
        return

    print("--- Initializing Modules ---")
    processor = VideoProcessor(whisper_model_size="base")
    embedding_engine = EmbeddingEngine() # Loads CLIP
    
    # Use local file-based Qdrant to run without Docker
    # Change path=None and ensure Docker is running to use server mode
    vector_store = VectorStore(embedding_engine=embedding_engine, path="./qdrant_local_store")

    print(f"\n--- Processing Video: {VIDEO_PATH} ---")
    
    # 1. Extract Content
    print("Extracting frames...")
    frames = processor.extract_frames(VIDEO_PATH, interval=2)
    
    print("Extracting audio...")
    audio_segments = processor.extract_audio_segments(VIDEO_PATH)

    # 2. Generate Embeddings
    print("\n--- Generating Embeddings ---")
    
    visual_vectors_data = []
    if frames:
        print(f"Encoding {len(frames)} visual frames...")
        # Extract just the PIL images for encoding
        pil_images = [f['image'] for f in frames]
        visual_embeddings = embedding_engine.encode_images(pil_images)
        
        # Combine vectors with timestamps
        for i, frame in enumerate(frames):
            visual_vectors_data.append({
                "vector": visual_embeddings[i],
                "timestamp": frame['timestamp']
            })

    audio_vectors_data = []
    if audio_segments:
        print(f"Encoding {len(audio_segments)} audio segments...")
        # Extract just the text for encoding
        texts = [s['text'] for s in audio_segments]
        audio_embeddings = embedding_engine.encode_text(texts)
        
        # Combine vectors with segment data
        for i, segment in enumerate(audio_segments):
            audio_vectors_data.append({
                "vector": audio_embeddings[i],
                "start": segment['start'],
                "end": segment['end'],
                "text": segment['text']
            })

    # 3. Index into Qdrant
    print("\n--- Indexing to Qdrant ---")
    vector_store.index_video(
        video_id=VIDEO_ID,
        visual_vectors=visual_vectors_data,
        audio_vectors=audio_vectors_data,
        metadata={"filename": VIDEO_PATH}
    )

    # 4. Search and Retrieval Demo
    print("\n--- Testing Search ---")
    query = "people running in a park"
    print(f"Query: '{query}'")
    
    results = vector_store.search(query, top_k=3)
    
    print("\nSearch Results:")
    for res in results:
        ts = res['timestamp']
        score = res['score']
        type_Str = res['type']
        print(f"[{type_Str.upper()}] Time: {ts:.2f}s | Score: {score:.4f}")
        if res.get('text'):
             print(f"   Context: \"{res['text']}\"")

if __name__ == "__main__":
    main()

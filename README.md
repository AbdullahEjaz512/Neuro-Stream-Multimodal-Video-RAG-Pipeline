# Neuro-Stream-Multimodal-Video-RAG-Pipeline

## Overview

**Neuro-Stream** is a Multimodal Video Retrieval-Augmented Generation (RAG) pipeline that enables semantic search over video content using both visual and audio modalities. It leverages CLIP for visual embeddings, Whisper for audio transcription, and Qdrant as a vector database for efficient similarity search.

---

## Features

- **Video Ingestion**: Extracts frames and audio segments from videos.
- **Multimodal Embedding**: Uses CLIP for visual frame embeddings and Whisper for audio transcription and embedding.
- **Vector Database**: Stores embeddings and metadata in Qdrant for fast retrieval.
- **Semantic Search**: Enables text-based search over both visuals and audio.
- **Streamlit UI**: Simple web interface for uploading videos and searching.

---

## Architecture

```
[Video File]
    |
    v
[Frame Extraction] <---+ 
    |                  |
    v                  |
[CLIP Embedding]       |
    |                  |
    v                  |
[Qdrant Vector DB] <---+--- [Whisper Transcription & Embedding] <--- [Audio Extraction]
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/AbdullahEjaz512/Neuro-Stream-Multimodal-Video-RAG-Pipeline.git
cd Neuro-Stream-Multimodal-Video-RAG-Pipeline
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Qdrant (Vector DB)

- **Option 1: Local (file-based, no Docker needed)**
  - The code will use a local folder (`qdrant_local_store/`) by default.
- **Option 2: Docker**
  ```bash
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
  ```

### 4. Run the Backend API

```bash
uvicorn src.api:app --reload
```

### 5. Run the Streamlit UI

```bash
streamlit run ui.py
```

---

## Usage

- **Upload a Video**: Use the sidebar in the UI to upload a video file (`.mp4`, `.mov`, `.avi`).
- **Semantic Search**: Enter a text query (e.g., "people running", "car horn", "red car") to search across both visuals and audio.
- **Results**: The UI displays matching video timestamps, type (visual/audio), and context.

---

## File Structure

```
.
├── src/
│   ├── api.py           # FastAPI backend
│   ├── embedding.py     # CLIP embedding logic
│   ├── processor.py     # Frame/audio extraction
│   └── vector_store.py  # Qdrant interaction
├── ui.py                # Streamlit UI
├── requirements.txt
├── README.md
└── ...
```

---

## Models Used

- **CLIP**: For visual (frame) embeddings ([OpenAI CLIP](https://github.com/openai/CLIP))
- **Whisper**: For audio transcription ([OpenAI Whisper](https://github.com/openai/whisper))
- **Qdrant**: Vector database ([Qdrant](https://qdrant.tech/))

---

## Notes

- The default setup uses local Qdrant storage for easy experimentation.
- For production, consider running Qdrant in Docker or cloud.
- The pipeline can be extended to support more modalities or downstream tasks.

---

## License

MIT License
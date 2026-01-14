import streamlit as st
import requests
import time

# Configuration
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Neuro-Stream", layout="wide")

st.title("🧠 Neuro-Stream: Multimodal Video Search")

st.markdown("""
This tool uses **CLIP** (Visuals) and **Whisper** (Audio) to allow semantic search through videos.
""")

# --- Sidebar for Upload ---
with st.sidebar:
    st.header("📤 Video Ingestion")
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        if st.button("Ingest Video"):
            with st.spinner("Uploading and starting background processing..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                try:
                    response = requests.post(f"{API_URL}/ingest", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Processing started! ID: `{data['video_id']}`")
                        st.info("The server is processing frames and audio in the background. Please wait a moment before searching.")
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to API. Is the backend running?")

# --- Main Search Area ---
st.header("🔍 Semantic Search")

query = st.text_input("Enter your search query (e.g., 'people running', 'yelling', 'red car')", "")
top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)

if st.button("Search") or query:
    if not query:
        st.warning("Please enter a query.")
    else:
        try:
            with st.spinner("Searching Vector Database..."):
                start_time = time.time()
                response = requests.get(f"{API_URL}/search", params={"q": query, "top_k": top_k})
                duration = time.time() - start_time
                
            if response.status_code == 200:
                results = response.json()
                st.caption(f"Found {len(results)} results in {duration:.2f}s")
                
                if not results:
                    st.info("No matches found.")
                
                for res in results:
                    score = res['score']
                    timestamp = res['timestamp']
                    res_type = res['type'].upper()
                    video_id = res['video_id']
                    text_context = res.get('text')

                    # Formatting the result card
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        
                        with col1:
                            st.metric(label="Score", value=f"{score:.3f}")
                            st.caption(f"{res_type}")
                        
                        with col2:
                            st.subheader(f"⏱️ Timestamp: {timestamp:.2f}s")
                            st.text(f"Video ID: {video_id}")
                            
                            if text_context:
                                st.markdown(f"**🗣️ Audio Context:** *\"{text_context}\"*")
                            else:
                                st.markdown(f"**🖼️ Visual Match**")
                        
                        st.divider()

            else:
                st.error(f"Error searching: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to API. Please ensure `main.py` or `api.py` is running.")

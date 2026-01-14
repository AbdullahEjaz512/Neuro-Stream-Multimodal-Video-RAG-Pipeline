import cv2
import whisper
import os
import tempfile
import logging
import sys
from typing import List, Dict, Union, Tuple
from PIL import Image
from moviepy import VideoFileClip

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure current environment's Scripts folder is in PATH so subprocess can find ffmpeg.exe
os.environ["PATH"] += os.pathsep + os.path.dirname(sys.executable)

class VideoProcessor:
    """
    Handles video ingestion, frame extraction, and audio transcription.
    """
    
    def __init__(self, whisper_model_size: str = "base"):
        """
        Initialize the VideoProcessor with a specific Whisper model.
        
        Args:
            whisper_model_size (str): Size of the Whisper model to load (e.g., "tiny", "base", "medium").
        """
        try:
            self.whisper_model = whisper.load_model(whisper_model_size)
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def extract_frames(self, video_path: str, interval: int = 2) -> List[Dict[str, Union[Image.Image, float]]]:
        """
        Extracts one frame every `interval` seconds from the video.
        
        Args:
            video_path (str): Path to the input video file.
            interval (int): Time interval in seconds between frames.
            
        Returns:
            List[Dict]: A list of dictionaries containing the PIL Image and its timestamp.
            Example: [{'image': <PIL.Image>, 'timestamp': 2.0}, ...]
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return []

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                logger.error("Error: Video FPS is 0. File might be corrupt.")
                return []

            frame_interval = int(fps * interval)
            current_frame = 0
            
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                if current_frame % frame_interval == 0:
                    # Convert BGR (OpenCV) to RGB (PIL)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    frames.append({
                        "image": pil_image,
                        "timestamp": timestamp
                    })
                
                current_frame += 1
                
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
        finally:
            cap.release()
            
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames

    def extract_audio_segments(self, video_path: str) -> List[Dict[str, Union[float, str]]]:
        """
        Uses Whisper to transcribe the video's audio track.
        
        Args:
            video_path (str): Path to the input video file.
            
        Returns:
            List[Dict]: A list of text segments with start time, end time, and text.
            Example: [{'start': 0.0, 'end': 2.0, 'text': 'Hello world'}, ...]
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        temp_audio_path = None
        segments_data = []

        try:
            # Extract audio using moviepy
            video = VideoFileClip(video_path)
            if video.audio is None:
                logger.warning(f"No audio track found in {video_path}")
                video.close()
                return []
            
            # Create a temporary file for the audio
            fd, temp_audio_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd) # Close file descriptor, let moviepy open it by name
            
            video.audio.write_audiofile(temp_audio_path, logger=None)
            video.close()
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(temp_audio_path)
            
            for segment in result.get("segments", []):
                segments_data.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip()
                })
                
        except Exception as e:
            logger.error(f"Error extracting/transcribing audio: {e}")
            if hasattr(e, 'message'):
                 logger.error(str(e))
        finally:
            # Cleanup temporary file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except OSError:
                    pass

        logger.info(f"Extracted {len(segments_data)} audio segments from {video_path}")
        return segments_data

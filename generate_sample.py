from moviepy import ColorClip, TextClip, CompositeVideoClip, AudioFileClip
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np

def create_sample_video(filename="sample_video.mp4", duration=5, color=(255, 0, 0)):
    print(f"Generating sample video: {filename}...")
    
    # Create a background with specified color
    clip = ColorClip(size=(640, 480), color=color, duration=duration)
    
    # Generate simple audio (440Hz sine wave)
    # 44100 Hz, stereo
    rate = 44100
    t = np.linspace(0, duration, duration*rate)
    # Sine wave
    audio_data = np.sin(2 * np.pi * 440 * t)
    # Stereo: stack data for 2 channels
    audio_data = np.stack([audio_data, audio_data], axis=1)
    
    audio = AudioArrayClip(audio_data, fps=rate)
    
    clip.audio = audio
    
    # Write file
    clip.write_videofile(filename, fps=24)
    print("Done.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Simple arg parsing for demo: python generate_sample.py filename r g b
        fname = sys.argv[1]
        col = (255, 0, 0)
        if len(sys.argv) > 4:
            col = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
        create_sample_video(fname, color=col)
    else:
        create_sample_video()

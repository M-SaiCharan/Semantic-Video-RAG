import whisper
import os
from moviepy.editor import VideoFileClip
import torch

class AudioProcessor:
    def __init__(self, model_size="base", device=None):
        # Whisper runs safely on CPU for Mac demos (MPS can be buggy with Whisper)
        self.device = "cpu" 
        print(f"Loading Whisper Audio Model ({model_size}) on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)

    def extract_audio(self, video_path, output_audio_path="temp_audio.mp3"):
        """Extracts audio from video file."""
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)
            
        print(f"Extracting audio from {video_path}...")
        try:
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(output_audio_path, verbose=False, logger=None)
            return output_audio_path
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None

    def transcribe(self, audio_path):
        """Transcribes audio and returns segments with timestamps."""
        print("Transcribing audio (listening)...")
        result = self.model.transcribe(audio_path)
        
        # We only care about segments for the timeline
        segments = []
        for segment in result["segments"]:
            start = segment["start"]
            text = segment["text"].strip()
            # Format: [00:12] (Audio): Hello world
            formatted = f"[{int(start)//60:02d}:{int(start)%60:02d}] (Audio): {text}"
            segments.append(formatted)
            print(formatted)
            
        return segments
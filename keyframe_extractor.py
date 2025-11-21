import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

class SemanticKeyframeExtractor:
    def __init__(self, model_id="openai/clip-vit-base-patch32", device=None):
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading CLIP model on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def extract_frames(self, video_path, sample_rate=1):
        """Extracts raw frames from video at a specific sample rate (fps)."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        timestamps = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * sample_rate) if sample_rate < 1 else int(fps / sample_rate)
        
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            count += 1
        cap.release()
        return frames, timestamps

    def get_embeddings(self, frames, batch_size=32):
        """Generates CLIP embeddings for all frames."""
        embeddings = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                embeds = self.model.get_image_features(**inputs)
                # Normalize embeddings for cosine similarity
                embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                embeddings.append(embeds.cpu().numpy())
        return np.vstack(embeddings)

    def cluster_and_select(self, frames, timestamps, n_clusters=15):
        """
        The Research Logic:
        1. Embed all frames.
        2. Use K-Means to find 'n_clusters' distinct scenes.
        3. Pick the frame closest to the center of each cluster (Keyframe).
        """
        if not frames:
            return [], []
        
        print("Generating embeddings for semantic clustering...")
        embeddings = self.get_embeddings(frames)
        
        # Dynamic cluster adjustment if video is short
        n_clusters = min(n_clusters, len(frames))
        
        print(f"Clustering {len(frames)} frames into {n_clusters} distinct semantic scenes...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        
        # Find the frame closest to each cluster center
        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
        
        # Sort keyframes by time so the story makes sense
        selected_indices = sorted(closest_indices)
        
        keyframes = [frames[i] for i in selected_indices]
        key_timestamps = [timestamps[i] for i in selected_indices]
        
        return keyframes, key_timestamps
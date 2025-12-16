import torch
import clip
from PIL import Image
import cv2
import requests
from io import BytesIO
import os
import json
from pytube import YouTube
import numpy as np
from typing import List, Dict, Any, Optional
import tempfile

class MultiModalProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP model for image understanding
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    def download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            return img
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return None

    def download_video(self, url: str, output_path: str = None) -> Optional[str]:
        """Download video from YouTube or direct URL."""
        try:
            if "youtube.com" in url or "youtu.be" in url:
                yt = YouTube(url)
                stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
                if output_path is None:
                    output_path = tempfile.mktemp(suffix='.mp4')
                stream.download(filename=output_path)
                return output_path
            else:
                # Direct video URL
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                if output_path is None:
                    output_path = tempfile.mktemp(suffix='.mp4')
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return output_path
        except Exception as e:
            print(f"Error downloading video from {url}: {e}")
            return None

    def extract_video_frames(self, video_path: str, num_frames: int = 10) -> List[Image.Image]:
        """Extract frames from video."""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                return frames

            # Extract frames evenly distributed
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    frames.append(img)

            cap.release()
            return frames
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            return []

    def generate_image_description(self, image: Image.Image) -> str:
        """Generate description of image using CLIP."""
        try:
            # Preprocess image
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

            # Text prompts for description
            text_prompts = [
                "a photo of",
                "a diagram of",
                "an illustration of",
                "a picture showing",
                "this image depicts"
            ]

            text_inputs = clip.tokenize(text_prompts).to(self.device)

            # Get features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)

                # Compute similarity
                similarity = (image_features @ text_features.T).softmax(dim=-1)

                # Get most similar prompt
                best_prompt_idx = similarity.argmax().item()
                best_prompt = text_prompts[best_prompt_idx]

            return f"This appears to be {best_prompt} [objects/concepts in the image]"

        except Exception as e:
            print(f"Error generating image description: {e}")
            return "An image that could not be described"

    def process_image(self, url: str) -> None:
        """Process and add image to training data."""
        img = self.download_image(url)
        if img:
            description = self.generate_image_description(img)

            data = {
                "text": f"Image from {url}: {description}",
                "source": url,
                "type": "image",
                "description": description
            }

            self._save_data(data)
            print(f"Processed and added image from: {url}")

    def process_video(self, url: str, num_frames: int = 5) -> None:
        """Process and add video to training data."""
        video_path = self.download_video(url)
        if video_path:
            frames = self.extract_video_frames(video_path, num_frames)

            descriptions = []
            for i, frame in enumerate(frames):
                desc = self.generate_image_description(frame)
                descriptions.append(f"Frame {i+1}: {desc}")

            video_description = " ".join(descriptions)

            data = {
                "text": f"Video from {url}: {video_description}",
                "source": url,
                "type": "video",
                "frames": len(frames),
                "description": video_description
            }

            self._save_data(data)
            print(f"Processed and added video from: {url}")

            # Clean up
            try:
                os.remove(video_path)
            except:
                pass

    def _save_data(self, data: Dict[str, Any]) -> None:
        """Save data to JSONL file."""
        filename = f"{self.data_dir}/training_data.jsonl"
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

    def analyze_image_with_question(self, image_url: str, question: str) -> str:
        """Ask a question about an image and generate an answer."""
        img = self.download_image(image_url)
        if not img:
            return "Could not load the image to analyze."

        description = self.generate_image_description(img)

        # Use the description to answer the question
        prompt = f"Based on this image description: '{description}', answer the question: {question}"

        # This would integrate with the main model for answering
        # For now, return a simple response
        return f"Looking at the image, which shows {description}, {question.lower().replace('what', 'it shows').replace('?', '.')}"
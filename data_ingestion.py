import requests
from bs4 import BeautifulSoup
import json
import os
from typing import List, Dict, Any

class DataIngestion:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def scrape_webpage(self, url: str) -> str:
        """Scrape text content from a webpage."""
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text content
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            return text
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return ""

    def add_user_text(self, text: str, source: str = "user_input") -> None:
        """Add user-provided text to the dataset."""
        data = {
            "text": text,
            "source": source,
            "type": "user_input"
        }
        self._save_data(data)

    def add_scraped_content(self, url: str) -> None:
        """Scrape and add webpage content to the dataset."""
        content = self.scrape_webpage(url)
        if content:
            data = {
                "text": content,
                "source": url,
                "type": "web_scraped"
            }
            self._save_data(data)

    def add_image(self, url: str, caption: str) -> None:
        """Download image and add it with a caption to the dataset."""
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()

            # Ensure the images directory exists
            images_dir = os.path.join(self.data_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            # Save the image
            image_filename = os.path.join(images_dir, url.split("/")[-1])
            with open(image_filename, 'wb') as f:
                f.write(response.content)

            data = {
                "text": caption,
                "image": image_filename,
                "source": url,
                "type": "image_text"
            }
            self._save_data(data)
            print(f"Added image {image_filename} with caption: {caption}")

        except Exception as e:
            print(f"Error adding image from {url}: {e}")

    def _save_data(self, data: Dict[str, Any]) -> None:
        """Save data to JSON file."""
        filename = f"{self.data_dir}/training_data.jsonl"
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

    def load_training_data(self) -> List[Dict[str, Any]]:
        """Load all training data."""
        filename = f"{self.data_dir}/training_data.jsonl"
        if not os.path.exists(filename):
            return []

        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

if __name__ == "__main__":
    ingestion = DataIngestion()

    # Example usage
    ingestion.add_user_text("The Earth orbits around the Sun once every 365 days.")
    ingestion.add_scraped_content("https://en.wikipedia.org/wiki/Solar_System")
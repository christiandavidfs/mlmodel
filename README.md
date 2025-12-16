# Self-Learning AI Model

A child-like learning AI that can self-chat, reason, learn from internet data and external models, and run in LMStudio.

## Features

- **Child-like Learning**: Self-chat mechanism where the model generates questions and answers to simulate curiosity-driven learning
- **Data Ingestion**: Add text from users, scrape web content, add images with captions, or get insights from external AI models via OpenRouter.
- **Multi-modal Learning**: Train the model on both text and images.
- **Distributed Training**: Fine-tune larger models faster using multiple GPUs.
- **External Learning**: Integrate insights from external AI models via OpenRouter
- **Reasoning**: Chain-of-thought prompting and self-reflection
- **LMStudio Compatible**: Export to GGUF format for local inference

## Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up OpenRouter API (optional for external learning):
```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

### Docker Setup (Alternative)

For CPU:
```bash
docker-compose up mlmodel
```

For GPU:
```bash
docker-compose up mlmodel_gpu
```

### Add Training Data

```bash
# Add custom text
python main_interface.py --add-text "The Earth is round and orbits the Sun."

# Scrape web content
python main_interface.py --add-web "https://en.wikipedia.org/wiki/Solar_System"

# Add image with caption
python main_interface.py --add-image "https://www.nasa.gov/sites/default/files/thumbnails/image/j2m-shareable.jpg" "A picture of the Earth from space."

# Get insight from external AI model
python main_interface.py --add-external "anthropic/claude-3-haiku" "quantum physics"
```

### Run Learning Sessions

```bash
# Basic self-chat
python main_interface.py --self-chat "Why does the sky look blue?"

# Advanced self-chat with verification
python main_interface.py --self-chat "photosynthesis" --turns 10 --verify

# Full self-learning pipeline (external insight + self-chat)
python main_interface.py --self-learn "machine learning" --turns 5
```

### Train the Model

```bash
# Fine-tune on single GPU
python main_interface.py --fine-tune

# Distributed training on multiple GPUs
python main_interface.py --distributed-train --world-size 2 --epochs 3
```

### Export for LMStudio

```bash
python main_interface.py --export
```

Then use llama.cpp to convert to GGUF:
```bash
python llama.cpp/convert.py fine_tuned_model_merged/ --outtype f16 --outfile model.gguf
```

## Complete Command Reference

### Data Ingestion Commands

#### Add Text
```bash
python main_interface.py --add-text "Your custom text here"
```

#### Add Web Content
```bash
python main_interface.py --add-web "https://example.com"
```

#### Add Image
```bash
python main_interface.py --add-image <image_url> "Your caption here"
```

#### Add External Insight
Get insights from external AI models via OpenRouter:
```bash
python main_interface.py --add-external <model_name> <topic>
# Example:
python main_interface.py --add-external "anthropic/claude-3-haiku" "quantum physics"
```

Available models include:
- `anthropic/claude-3-haiku`
- `minimax/minimax-m2`
- `openai/gpt-3.5-turbo`
- And many more (see [OpenRouter docs](https://openrouter.ai/docs))

### Self-Chat Commands

#### Basic Self-Chat
```bash
python main_interface.py --self-chat "topic" [options]
```

Options:
- `--turns <int>`: Number of conversation turns (default: 5)
- `--questions-per-turn <int>`: Questions to generate per turn (default: 1)
- `--q-temp <float>`: Temperature for question generation (default: 0.8)
- `--q-top-p <float>`: Top-p for question generation (default: 0.9)
- `--a-temp <float>`: Temperature for answer generation (default: 0.6)
- `--a-top-p <float>`: Top-p for answer generation (default: 0.9)
- `--novelty-threshold <float>`: Similarity threshold for novel questions (default: 0.8)
- `--max-total-pairs <int>`: Maximum Q/A pairs to generate
- `--stop-on-consecutive-no-new <int>`: Stop after consecutive non-novel turns (default: 3)
- `--verify`: Enable answer verification before saving
- `--verify-method <local|openrouter>`: Verification method (default: local)
- `--use-external-for-self-chat <model>`: Use external model for self-chat
- `--use-qa-context`: Build questions on previous Q/A context
- `--qa-context-depth <int>`: Context depth when using QA context (default: 2)
- `--qa-context-include-reflection`: Include reflections in QA context

#### Full Self-Learning Pipeline
Combines external insight fetching and self-chat:
```bash
python main_interface.py --self-learn "topic" [options]
```

Options:
- `--turns <int>`: Number of self-chat turns (default: 5)
- `--self-learn-model <model>`: Model for external insights (default: anthropic/claude-3-haiku)
- `--questions-per-turn <int>`: Questions per turn (default: 1)
- `--q-temp <float>`: Question temperature (default: 0.8)
- `--q-top-p <float>`: Question top-p (default: 0.9)
- `--a-temp <float>`: Answer temperature (default: 0.6)
- `--a-top-p <float>`: Answer top-p (default: 0.9)
- `--novelty-threshold <float>`: Novelty threshold (default: 0.8)
- `--verify`: Enable verification (default: True)
- `--use-qa-context`: Use QA context
- `--qa-context-depth <int>`: Context depth (default: 2)
- `--qa-context-include-reflection`: Include reflections

### Training Commands

#### Single GPU Fine-Tuning
```bash
python main_interface.py --fine-tune
```

#### Distributed Training
```bash
python main_interface.py --distributed-train [options]
```

Options:
- `--world-size <int>`: Number of processes/GPUs (default: 2)
- `--epochs <int>`: Number of training epochs (default: 3)
- `--model_name <str>`: Model to fine-tune (default: distilgpt2)
- `--data_dir <str>`: Training data directory (default: data)
- `--output_dir <str>`: Output directory (default: distributed_model)

### Model Management

#### Export Model
Merge and export model for GGUF conversion:
```bash
python main_interface.py --export
```

#### Show Data Statistics
```bash
python main_interface.py --stats
```

### Distributed Inference

Run inference across multiple GPUs:
```bash
python distributed_inference.py --topic "your topic" --world_size 2
```

### Testing

Run the test suite:
```bash
pytest
```

## Architecture

- `main_interface.py`: CLI interface for all operations
- `data_ingestion.py`: Handles data collection from various sources
- `self_chat_engine.py`: Implements the curiosity-driven self-dialogue
- `multi_modal_model.py`: A multi-modal model combining language model with vision encoder
- `fine_tune.py`: Manages single-GPU model training with LoRA
- `distributed_fine_tuner.py`: Manages distributed model training
- `distributed_inference.py`: Handles distributed inference
- `multi_modal_processor.py`: Processes multi-modal data

## Requirements

- Python 3.8+
- PyTorch (with CUDA for GPU support)
- Hugging Face Transformers
- OpenRouter API key (optional, for external learning)
- `accelerate`, `peft`, `datasets` for training
- `beautifulsoup4`, `requests` for web scraping
- `Pillow`, `opencv-python-headless`, `pytube` for multi-modal capabilities
- `llama-cpp-python` for GGUF conversion

## Environment Variables

- `OPENROUTER_API_KEY`: API key for external AI model access
- `DEBUG`: Enable debug mode for error details

## Docker Usage

### CPU Version
```bash
# Build and run
docker-compose up mlmodel

# Or run specific commands
docker-compose run --rm mlmodel --add-text "Hello world"
```

### GPU Version
```bash
# Build and run
docker-compose up mlmodel_gpu

# Or run specific commands
docker-compose run --rm mlmodel_gpu --fine-tune
```

## LMStudio Setup

1. Export the model using `--export`
2. Convert to GGUF using llama.cpp:
   ```bash
   python llama.cpp/convert.py fine_tuned_model_merged/ --outtype f16 --outfile model.gguf
   ```
3. Load the GGUF file in LMStudio
4. Use chat templates for interactive learning sessions

## Future Improvements

- Reinforcement learning for better question generation
- Web interface for easier interaction
- Support for more multi-modal data types (e.g. video, audio)
- More sophisticated multi-modal fusion techniques
- Enhanced verification methods
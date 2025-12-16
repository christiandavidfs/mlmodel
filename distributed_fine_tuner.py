from multi_modal_model import MultiModalModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, AutoProcessor
from datasets import Dataset
import torch
import json
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from typing import List, Dict, Any, Optional
import argparse
from PIL import Image

class DistributedFineTuner:
    def __init__(self, model_name: str = "gpt2", data_dir: str = "data", output_dir: str = "distributed_model"):
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def setup_distributed(self, rank: int, world_size: int):
        """Setup distributed training."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # Initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup_distributed(self):
        """Clean up distributed training."""
        dist.destroy_process_group()

    def load_training_data(self) -> Dataset:
        """Load training data from JSONL file."""
        filename = f"{self.data_dir}/training_data.jsonl"

        if not os.path.exists(filename):
            print("No training data found. Please add some data first.")
            return None

        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if "image" in item and os.path.exists(item["image"]):
                    try:
                        image = Image.open(item["image"])
                        item["image"] = self.vision_processor(images=image, return_tensors="pt")["pixel_values"]
                        data.append(item)
                    except Exception as e:
                        print(f"Could not load image {item['image']}: {e}")
                else:
                    data.append({"text": item["text"]})

        return Dataset.from_list(data)

    def tokenize_function(self, examples, tokenizer):
        """Tokenize the text data."""
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    def fine_tune(self, rank: int, world_size: int, num_epochs: int = 3):
        """Main distributed training function."""
        self.setup_distributed(rank, world_size)

        # Load model and tokenizer
        model = MultiModalModel(model_name=self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare model for LoRA
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        model = model.to(rank)

        # Wrap model for distributed training
        model = DDP(model, device_ids=[rank])

        # Load and prepare dataset
        dataset = self.load_training_data()
        if dataset is None:
            self.cleanup_distributed()
            return

        tokenized_dataset = dataset.map(
            lambda x: self.tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=list(dataset.column_names)
        )

        # Data collator for multi-modal data
        def multi_modal_collator(features):
            pixel_values = torch.cat([f["pixel_values"] for f in features if "pixel_values" in f]) if any("pixel_values" in f for f in features) else None
            input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(f["input_ids"]) for f in features], batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(f["attention_mask"]) for f in features], batch_first=True, padding_value=0)
            labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(f["input_ids"]) for f in features], batch_first=True, padding_value=-100)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "labels": labels,
            }

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=multi_modal_collator,
            tokenizer=tokenizer,
        )

        if rank == 0:
            print("Starting distributed training...")

        trainer.train()

        if rank == 0:
            # Save the fine-tuned LoRA model
            trainer.save_model(self.output_dir)
            tokenizer.save_pretrained(self.output_dir)
            print(f"Distributed training completed. LoRA model saved to {self.output_dir}")

            # Merge and save the full model
            self.merge_and_save_full_model()

        self.cleanup_distributed()

    def merge_and_save_full_model(self):
        """Merge LoRA weights and save the full model for GGUF conversion."""
        print("Loading base model for merging...")
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        print("Loading fine-tuned LoRA model...")
        lora_model = PeftModel.from_pretrained(base_model, self.output_dir)

        print("Merging LoRA weights...")
        merged_model = lora_model.merge_and_unload()

        merged_output_dir = f"{self.output_dir}_merged"
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)

        print(f"Merged model saved to {merged_output_dir}")
        return merged_output_dir

def run_distributed_finetuning(world_size: int, num_epochs: int = 3, model_name: str = "gpt2", data_dir: str = "data", output_dir: str = "distributed_model"):
    """Launch distributed training across multiple processes."""
    tuner = DistributedFineTuner(model_name=model_name, data_dir=data_dir, output_dir=output_dir)
    mp.spawn(
        tuner.fine_tune,
        args=(world_size, num_epochs),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Fine-Tuning")
    parser.add_argument("--world_size", type=int, default=2, help="Number of processes/machines")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Name of the model to fine-tune")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with training data")
    parser.add_argument("--output_dir", type=str, default="distributed_model", help="Directory to save the fine-tuned model")

    args = parser.parse_args()

    run_distributed_finetuning(
        world_size=args.world_size,
        num_epochs=args.epochs,
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
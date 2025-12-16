import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import os
import json
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
from typing import List, Dict, Any, Optional
import argparse

class DistributedTrainer:
    def __init__(self, model_name: str = "gpt2", data_dir: str = "data", output_dir: str = "distributed_model"):
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.accelerator = Accelerator()

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
                data.append({"text": item["text"]})

        return Dataset.from_list(data)

    def tokenize_function(self, examples, tokenizer):
        """Tokenize the text data."""
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    def train_distributed(self, rank: int, world_size: int, num_epochs: int = 3):
        """Main distributed training function."""
        self.setup_distributed(rank, world_size)

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
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
            remove_columns=["text"]
        )

        # Split data across processes
        sampler = torch.utils.data.distributed.DistributedSampler(
            tokenized_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}_rank_{rank}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_dir=f"./logs_rank_{rank}",
            logging_steps=10,
            save_steps=500,
            dataloader_num_workers=0,  # Important for distributed training
            dataloader_pin_memory=False,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=lambda data: {
                'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in data]),
                'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in data])
            }
        )

        if rank == 0:
            print("Starting distributed training...")

        trainer.train()

        # Save model (only rank 0 saves the full model)
        if rank == 0:
            trainer.save_model(self.output_dir)
            tokenizer.save_pretrained(self.output_dir)
            print(f"Distributed training completed. Model saved to {self.output_dir}")

        self.cleanup_distributed()

    def merge_distributed_models(self, world_size: int):
        """Merge models from different ranks."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Average the LoRA weights from all ranks
        merged_state_dict = {}

        for rank in range(world_size):
            rank_model_path = f"{self.output_dir}_rank_{rank}"
            if os.path.exists(rank_model_path):
                rank_model = AutoModelForCausalLM.from_pretrained(rank_model_path)
                rank_state_dict = rank_model.state_dict()

                for key, value in rank_state_dict.items():
                    if key not in merged_state_dict:
                        merged_state_dict[key] = value.clone()
                    else:
                        merged_state_dict[key] += value

        # Average the weights
        for key in merged_state_dict:
            merged_state_dict[key] /= world_size

        # Load averaged weights into base model
        base_model.load_state_dict(merged_state_dict)

        # Save merged model
        base_model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        print(f"Merged distributed model saved to {self.output_dir}")

def run_distributed_training(world_size: int, num_epochs: int = 3):
    """Launch distributed training across multiple processes."""
    mp.spawn(
        DistributedTrainer().train_distributed,
        args=(world_size, num_epochs),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Training for Self-Learning AI")
    parser.add_argument("--world_size", type=int, default=2, help="Number of processes/machines")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")

    args = parser.parse_args()

    trainer = DistributedTrainer()
    run_distributed_training(args.world_size, args.epochs)

    # Merge the distributed models
    trainer.merge_distributed_models(args.world_size)
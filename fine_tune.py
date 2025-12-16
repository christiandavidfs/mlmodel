from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, AutoProcessor
from datasets import Dataset
import torch
import json
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import List, Dict, Any
from multi_modal_model import MultiModalModel
from PIL import Image
from training_config import TRAINING_CONFIG # Import auto-detected configuration

class FineTuner:
    def __init__(self, model_name: str = "distilgpt2", data_dir: str = "data", output_dir: str = "fine_tuned_model"):
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = torch.device(TRAINING_CONFIG["device"]) # Use configured device
        self.vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=TRAINING_CONFIG["max_length"])

    def fine_tune(self): # Removed num_epochs and learning_rate from signature
        """Fine-tune the multi-modal model."""
        print("Loading multi-modal model and tokenizer...")
        model = MultiModalModel(model_name=self.model_name).to(self.device) # Move model to configured device
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare model for LoRA
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj"],  # For GPT-2
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)

        # Load and prepare dataset
        dataset = self.load_training_data()
        if dataset is None:
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
            num_train_epochs=TRAINING_CONFIG["num_epochs"],
            per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
            gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=multi_modal_collator,
            tokenizer=tokenizer,
        )

        print("Starting fine-tuning...")
        trainer.train()

        # Save the fine-tuned model
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        print(f"Fine-tuned model saved to {self.output_dir}")

    def merge_and_save_full_model(self):
        """Merge LoRA weights and save the full model for GGUF conversion."""
        from peft import PeftModel

        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        print("Loading fine-tuned LoRA model...")
        model = PeftModel.from_pretrained(base_model, self.output_dir)

        print("Merging LoRA weights...")
        merged_model = model.merge_and_unload()

        merged_output_dir = f"{self.output_dir}_merged"
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)

        print(f"Merged model saved to {merged_output_dir}")
        return merged_output_dir

if __name__ == "__main__":
    # First, ensure training_config.py is generated
    from setup_config import SetupConfig
    setup = SetupConfig()
    config = setup.detect_optimal_config()
    setup.generate_training_config_file(config)

    tuner = FineTuner()

    # Fine-tune the model
    tuner.fine_tune() # Call without parameters, as they are now in TRAINING_CONFIG

    # Merge and save full model for GGUF conversion
    merged_dir = tuner.merge_and_save_full_model()
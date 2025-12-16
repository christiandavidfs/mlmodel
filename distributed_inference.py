import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
from typing import List, Dict, Any
import argparse

class DistributedInference:
    def __init__(self, model_path: str = "fine_tuned_model_merged"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_distributed(self, rank: int, world_size: int):
        """Setup distributed inference."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'  # Different port from training

        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup_distributed(self):
        """Clean up distributed inference."""
        dist.destroy_process_group()

    def load_model(self, rank: int):
        """Load model for distributed inference."""
        model = AutoModelForCausalLM.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Shard the model across processes
        model = DDP(model, device_ids=[rank])

        return model, tokenizer

    def generate_response(self, model, tokenizer, prompt: str, max_length: int = 100) -> str:
        """Generate response using distributed model."""
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        return response

    def collaborative_reasoning(self, rank: int, world_size: int, topic: str):
        """Perform collaborative reasoning across distributed processes."""
        self.setup_distributed(rank, world_size)
        model, tokenizer = self.load_model(rank)

        # Each process explores a different aspect of the topic
        aspects = [
            f"What is {topic}?",
            f"How does {topic} work?",
            f"Why is {topic} important?",
            f"What are examples of {topic}?",
            f"How can we learn more about {topic}?"
        ]

        # Assign aspect based on rank
        aspect = aspects[rank % len(aspects)]

        # Generate initial response
        response = self.generate_response(model, tokenizer, aspect, max_length=150)

        # Share responses across all processes
        responses = [None] * world_size
        dist.all_gather_object(responses, response)

        # Each process synthesizes all responses
        synthesis_prompt = f"Topic: {topic}\n\nDifferent perspectives:\n"
        for i, resp in enumerate(responses):
            synthesis_prompt += f"View {i+1}: {resp}\n"

        synthesis_prompt += f"\nAs process {rank+1}, synthesize these views into a comprehensive understanding:"

        final_response = self.generate_response(model, tokenizer, synthesis_prompt, max_length=300)

        # Save individual process results
        result = {
            "rank": rank,
            "topic": topic,
            "aspect": aspect,
            "individual_response": response,
            "synthesized_response": final_response
        }

        with open(f"distributed_inference_rank_{rank}.json", 'w') as f:
            json.dump(result, f, indent=2)

        if rank == 0:
            print(f"Distributed inference completed for topic: {topic}")
            print("Results saved in distributed_inference_rank_*.json files")

        self.cleanup_distributed()

    def merge_inference_results(self, world_size: int) -> Dict[str, Any]:
        """Merge results from all distributed inference processes."""
        merged_results = {
            "topic": None,
            "individual_responses": [],
            "synthesized_responses": []
        }

        for rank in range(world_size):
            filename = f"distributed_inference_rank_{rank}.json"
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    result = json.load(f)

                if merged_results["topic"] is None:
                    merged_results["topic"] = result["topic"]

                merged_results["individual_responses"].append({
                    "rank": result["rank"],
                    "aspect": result["aspect"],
                    "response": result["individual_response"]
                })

                merged_results["synthesized_responses"].append({
                    "rank": result["rank"],
                    "response": result["synthesized_response"]
                })

        # Save merged results
        with open("distributed_inference_merged.json", 'w') as f:
            json.dump(merged_results, f, indent=2)

        return merged_results

def run_distributed_inference(rank: int, world_size: int, topic: str):
    """Launch distributed inference for a specific rank."""
    inference = DistributedInference()
    inference.collaborative_reasoning(rank, world_size, topic)

def launch_distributed_inference(topic: str, world_size: int):
    """Launch distributed inference across multiple processes."""
    mp.spawn(
        run_distributed_inference,
        args=(world_size, topic),
        nprocs=world_size,
        join=True
    )

    # Merge results
    inference = DistributedInference()
    merged = inference.merge_inference_results(world_size)

    print(f"\nCollaborative reasoning on '{topic}' completed!")
    print(f"Individual perspectives: {len(merged['individual_responses'])}")
    print(f"Synthesized understandings: {len(merged['synthesized_responses'])}")
    print("Results saved in distributed_inference_merged.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Inference for Self-Learning AI")
    parser.add_argument("--topic", required=True, help="Topic to reason about")
    parser.add_argument("--world_size", type=int, default=2, help="Number of processes")

    args = parser.parse_args()

    launch_distributed_inference(args.topic, args.world_size)
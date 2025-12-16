#!/usr/bin/env python3
import argparse
import os
import sys
from data_ingestion import DataIngestion
from self_chat_engine import SelfChatEngine
from fine_tune import FineTuner
from multi_modal_processor import MultiModalProcessor
from distributed_fine_tuner import run_distributed_finetuning
from distributed_inference import DistributedInference, launch_distributed_inference
import traceback
from openrouter import OpenRouter

class MainInterface:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.self_chat_engine = SelfChatEngine()
        self.fine_tuner = FineTuner()
        self.multi_modal = MultiModalProcessor()

        # Initialize OpenRouter client
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_key = openrouter_key
        if openrouter_key:
            self.openrouter_client = OpenRouter(api_key=openrouter_key)
        else:
            self.openrouter_client = None

    def add_text(self, text: str):
        """Add user-provided text to training data."""
        self.data_ingestion.add_user_text(text)
        print(f"Added text: {text[:50]}...")

    def add_web_content(self, url: str):
        """Scrape and add web content."""
        self.data_ingestion.add_scraped_content(url)
        print(f"Scraped and added content from: {url}")

    def add_external_insight(self, topic: str, model: str = "minimax/minimax-m2"):
        """Get insight from external AI model via OpenRouter and add to training data."""
        if not self.openrouter_key:
            print("OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable.")
            return

        try:
            prompt = f"Explain {topic} in simple terms that a child could understand, with reasoning steps."
            print(f"Sending request to {model}...")
            
            with OpenRouter(api_key=self.openrouter_key) as client:
                response = client.chat.send(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful teacher explaining complex topics to children."},
                        {"role": "user", "content": prompt}
                    ]
                )
            insight = response.choices[0].message.content

            self.data_ingestion.add_user_text(insight, source=f"openrouter_{model}")
            print(f"\n✓ Successfully added insight from {model} on '{topic}':")
            print(f"\n{insight}")
        except Exception as e:
            error_msg = str(e)
            print(f"\n✗ Error getting insight from {model}: {error_msg}")
            
            if "cookie" in error_msg.lower() or "401" in error_msg or "authentication" in error_msg.lower():
                print(f"\n  ⚠ Authentication Issue Detected:")
                print(f"    - Your API key appears to be invalid or has been revoked")
                print(f"    - Get a new key from: https://openrouter.ai/keys")
                print(f"    - Make sure you're logged in to OpenRouter")
                print(f"    - Set the new key: export OPENROUTER_API_KEY=\"your-new-key\"")
            elif "model" in error_msg.lower():
                print(f"\n  ⚠ Model Issue:")
                print(f"    - Check the model name: {model}")
                print(f"    - Available models: https://openrouter.ai/docs")
            
            if os.getenv("DEBUG"):
                traceback.print_exc()

    def run_self_chat(
        self,
        topic: str,
        turns: int = 5,
        questions_per_turn=1,
        q_temp: float = 0.8,
        q_top_p: float = 0.9,
        a_temp: float = 0.6,
        a_top_p: float = 0.9,
        novelty_threshold: float = 0.8,
        max_total_pairs=None,
        stop_on_consecutive_no_new: int = 3,
        verify: bool = False,
        verify_method: str = "local",
        external_model: str = None,
        use_qa_context: bool = False,
        qa_context_depth: int = 2,
        qa_context_include_reflection: bool = False,
    ):
        """Run a self-chat learning session with configurable generation parameters."""
        if external_model:
            # Reinitialize engine with external model
            self.self_chat_engine = SelfChatEngine(external_model=external_model)
        
        print(f"Starting self-chat session on topic: {topic}")
        conversation = self.self_chat_engine.self_chat_session(
            initial_topic=topic,
            num_turns=turns,
            questions_per_turn=questions_per_turn,
            q_temp=q_temp,
            q_top_p=q_top_p,
            a_temp=a_temp,
            a_top_p=a_top_p,
            novelty_threshold=novelty_threshold,
            max_total_pairs=max_total_pairs,
            stop_on_consecutive_no_new=stop_on_consecutive_no_new,
            verify=verify,
            verify_method=verify_method,
            use_qa_context=use_qa_context,
            qa_context_depth=qa_context_depth,
            include_reflection=qa_context_include_reflection,
        )

        print("\nSelf-Chat Conversation:")
        for turn in conversation:
            print(f"\nTurn {turn['turn']}:")
            print(f"Question: {turn['question']}")
            print(f"Answer: {turn['answer']}")
            print(f"Reflection: {turn['reflection']}")

        self.self_chat_engine.save_conversation(conversation)
        print(f"\nConversation saved to training data.")

    def fine_tune_model(self):
        """Fine-tune the model on collected data."""
        print("Starting fine-tuning process...")
        self.fine_tuner.fine_tune()
        print("Fine-tuning completed.")

    def export_model(self):
        """Export model for GGUF conversion."""
        print("Merging and exporting model...")
        merged_dir = self.fine_tuner.merge_and_save_full_model()
        print(f"Model exported to: {merged_dir}")
        print("Use llama.cpp to convert to GGUF format for LMStudio.")

    def show_stats(self):
        """Show current data statistics."""
        data = self.data_ingestion.load_training_data()
        sources = {}
        types = {}

        for item in data:
            source = item.get('source', 'unknown')
            item_type = item.get('type', 'unknown')
            sources[source] = sources.get(source, 0) + 1
            types[item_type] = types.get(item_type, 0) + 1

        print(f"Total training samples: {len(data)}")
        print("Sources:", sources)
        print("Types:", types)

    def self_learn(self, topic: str, turns: int = 5, model: str = "anthropic/claude-3-haiku", 
                   questions_per_turn: int = 1, q_temp: float = 0.8, q_top_p: float = 0.9,
                   a_temp: float = 0.6, a_top_p: float = 0.9, novelty_threshold: float = 0.8,
                   verify: bool = True, use_qa_context: bool = False, qa_context_depth: int = 2, qa_context_include_reflection: bool = False):
        """Run full self-learning pipeline: fetch external insight, then self-chat on topic."""
        print(f"\n{'='*60}")
        print(f"🧠 Starting Self-Learning Pipeline on Topic: {topic}")
        print(f"{'='*60}\n")
        
        # Step 1: Fetch external insight
        print(f"📚 Step 1: Fetching external insight from {model}...")
        self.add_external_insight(topic, model)
        
        # Step 2: Run self-chat
        print(f"\n💬 Step 2: Running self-chat session with verification...")
        self.run_self_chat(
            topic,
            turns=turns,
            questions_per_turn=questions_per_turn,
            q_temp=q_temp,
            q_top_p=q_top_p,
            a_temp=a_temp,
            a_top_p=a_top_p,
            novelty_threshold=novelty_threshold,
            verify=verify,
            verify_method="local",
            external_model=model,  # Use same model for self-chat
            use_qa_context=use_qa_context,
            qa_context_depth=qa_context_depth,
            qa_context_include_reflection=qa_context_include_reflection
        )
        
        print(f"\n✅ Self-Learning Pipeline Complete!")
        print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Self-Learning AI Model Interface")
    parser.add_argument("--add-text", help="Add custom text to training data")
    parser.add_argument("--add-web", help="Scrape and add web content")
    parser.add_argument("--add-external", nargs='+', help="Get insight from external AI model via OpenRouter (format: model_name topic...)")
    parser.add_argument("--model", type=str, help="Model to use with --add-external (default: minimax/minimax-m2)")
    parser.add_argument("--add-image", nargs=2, help="Process and add image from URL with a caption (format: url caption)")
    parser.add_argument("--self-chat", help="Run self-chat learning session")
    parser.add_argument("--turns", type=int, default=5, help="Number of self-chat turns")
    parser.add_argument("--questions-per-turn", type=int, default=1, help="Number of candidate questions to sample per turn")
    parser.add_argument("--q-temp", type=float, default=0.8, help="Temperature for question generation")
    parser.add_argument("--q-top-p", type=float, default=0.9, help="Top-p for question generation")
    parser.add_argument("--a-temp", type=float, default=0.6, help="Temperature for answer generation")
    parser.add_argument("--a-top-p", type=float, default=0.9, help="Top-p for answer generation")
    parser.add_argument("--novelty-threshold", type=float, default=0.8, help="Similarity threshold to consider a question novel (0-1)")
    parser.add_argument("--max-total-pairs", type=int, default=None, help="Maximum total Q/A pairs to generate")
    parser.add_argument("--stop-on-consecutive-no-new", type=int, default=3, help="Stop after this many consecutive turns with no novel questions")
    parser.add_argument("--fine-tune", action="store_true", help="Fine-tune the model")
    parser.add_argument("--distributed-train", action="store_true", help="Run distributed training")
    parser.add_argument("--world-size", type=int, default=2, help="Number of processes for distributed training/inference")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="Name of the model to fine-tune")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with training data")
    parser.add_argument("--output_dir", type=str, default="distributed_model", help="Directory to save the fine-tuned model")
    parser.add_argument("--verify", action="store_true", help="Enable verification of generated Q/A before saving")
    parser.add_argument("--verify-method", type=str, choices=["local", "openrouter"], default="local", help="Verification method: local rule-based or openrouter remote")
    parser.add_argument("--use-external-for-self-chat", type=str, default=None, help="Use external OpenRouter model for self-chat (e.g., anthropic/claude-3-haiku)")
    parser.add_argument("--use-qa-context", action="store_true", help="Make each generated question build on the previous question+answer context")
    parser.add_argument("--qa-context-depth", type=int, default=2, help="How many previous Q/A pairs to include when --use-qa-context is set (default 2)")
    parser.add_argument("--qa-context-include-reflection", action="store_true", help="Include the previous turn's reflection in the QA context when using --use-qa-context")
    parser.add_argument("--self-learn", help="Run full self-learning pipeline: fetch external insight, then self-chat on topic")
    parser.add_argument("--self-learn-model", type=str, default="anthropic/claude-3-haiku", help="Model to use for self-learn external insights (default: anthropic/claude-3-haiku)")
    parser.add_argument("--export", action="store_true", help="Export model for LMStudio")
    parser.add_argument("--stats", action="store_true", help="Show data statistics")

    args = parser.parse_args()

    interface = MainInterface()

    if args.add_text:
        interface.add_text(args.add_text)
    elif args.add_web:
        interface.add_web_content(args.add_web)
    elif args.add_external:
        # Support both --add-external model topic and --add-external topic --model model
        if len(args.add_external) >= 2:
            # Format: --add-external model_name topic topic...
            model = args.add_external[0]
            topic = ' '.join(args.add_external[1:])
        else:
            # Format: --add-external topic --model model_name
            model = args.model if args.model else "minimax/minimax-m2"
            topic = args.add_external[0]
        interface.add_external_insight(topic, model)
    elif args.add_image:
        url, caption = args.add_image
        interface.data_ingestion.add_image(url, caption)
    elif args.self_chat:
        interface.run_self_chat(
            args.self_chat,
            turns=args.turns,
            questions_per_turn=args.questions_per_turn,
            q_temp=args.q_temp,
            q_top_p=args.q_top_p,
            a_temp=args.a_temp,
            a_top_p=args.a_top_p,
            novelty_threshold=args.novelty_threshold,
            max_total_pairs=args.max_total_pairs,
            stop_on_consecutive_no_new=args.stop_on_consecutive_no_new,
            verify=args.verify,
            verify_method=args.verify_method,
            external_model=args.use_external_for_self_chat,
            use_qa_context=args.use_qa_context,
            qa_context_depth=args.qa_context_depth,
            qa_context_include_reflection=args.qa_context_include_reflection
        )
    elif args.self_learn:
        interface.self_learn(
            args.self_learn,
            turns=args.turns,
            model=args.self_learn_model,
            questions_per_turn=args.questions_per_turn,
            q_temp=args.q_temp,
            q_top_p=args.q_top_p,
            a_temp=args.a_temp,
            a_top_p=args.a_top_p,
            novelty_threshold=args.novelty_threshold,
            verify=args.verify,
            use_qa_context=args.use_qa_context,
            qa_context_depth=args.qa_context_depth,
            qa_context_include_reflection=args.qa_context_include_reflection
        )
    elif args.fine_tune:
        interface.fine_tune_model()
    elif args.distributed_train:
        run_distributed_finetuning(
            world_size=args.world_size,
            num_epochs=args.epochs,
            model_name=args.model_name,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
    elif args.export:
        interface.export_model()
    elif args.stats:
        interface.show_stats()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
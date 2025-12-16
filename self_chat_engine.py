from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
from typing import List, Dict, Any, Optional

class SelfChatEngine:
    def __init__(self, model_name: str = "distilgpt2", data_dir: str = "data", external_model: str = None):
        self.model_name = model_name
        self.data_dir = data_dir
        self.external_model = external_model
        self.use_external = external_model is not None
        self.tokenizer = None
        self.model = None
        self.openrouter_client = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.use_external:
            print(f"Using external model for self-chat: {external_model}")
            self._init_openrouter()
        else:
            print(f"Using local model for self-chat: {model_name}")

    def _init_openrouter(self):
        """Initialize OpenRouter client for external model."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set. Please set it to use external models.")
        try:
            from openrouter import OpenRouter
            self.openrouter_client = OpenRouter(api_key=api_key)
            print(f"✓ OpenRouter client initialized for model: {self.external_model}")
        except ImportError:
            raise ImportError("openrouter package not installed. Run: pip install openrouter")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenRouter client: {e}")

    def load_model(self):
        """Load the language model."""
        if self.use_external:
            # External model is already initialized via OpenRouter
            return
        
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate a response from the model."""
        if self.model is None:
            self.load_model()
        return self.generate_from_model(prompt, max_new_tokens=max_new_tokens)

    def _generate_from_external(self, prompt: str, max_new_tokens: int = 150, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate text using external OpenRouter model."""
        try:
            response = self.openrouter_client.chat.send(
                model=self.external_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n⚠ Warning: External model call failed ({e}). Falling back to local model.")
            self.use_external = False
            self.load_model()
            return self._generate_from_local(prompt, max_new_tokens, temperature, top_p)

    def _generate_from_local(self, prompt: str, max_new_tokens: int = 150, temperature: float = 0.7, top_p: float = 0.9, num_return_sequences: int = 1) -> str:
        """Generate text using local model."""
        if self.model is None:
            self.load_model()

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=top_p
            )

        # If multiple return sequences, join them by newline; otherwise return single
        results = []
        for i in range(num_return_sequences):
            out = outputs[i]
            gen = self.tokenizer.decode(out[len(inputs["input_ids"][0]):], skip_special_tokens=True)
            results.append(gen.strip())

        if num_return_sequences == 1:
            return results[0]
        return "\n".join(results)

    def generate_from_model(self, prompt: str, max_new_tokens: int = 150, temperature: float = 0.7, top_p: float = 0.9, num_return_sequences: int = 1) -> str:
        """Generate text using either external or local model."""
        if self.use_external:
            return self._generate_from_external(prompt, max_new_tokens, temperature, top_p)
        else:
            return self._generate_from_local(prompt, max_new_tokens, temperature, top_p, num_return_sequences)

    def validate_turn(self, turn: Dict[str, str]) -> tuple:
        """Validate a single Q/A turn. Returns (bool, List[str]) where bool is pass/fail and list is rejection reasons."""
        from difflib import SequenceMatcher
        from collections import Counter
        
        reasons = []
        
        # Extract fields
        question = turn.get("question", "").strip()
        answer = turn.get("answer", "").strip()
        
        # Check for non-empty answer and question
        if not answer:
            reasons.append("Empty answer")
            return (False, reasons)
        if not question:
            reasons.append("Empty question")
            return (False, reasons)
        
        # Check question length and repetition
        question_tokens = question.split()
        if len(question_tokens) < 3:
            reasons.append(f"Question too short ({len(question_tokens)} tokens, min 3)")
        if len(set(question_tokens)) < len(question_tokens) * 0.4:  # < 40% unique tokens = repetitive
            reasons.append(f"Question is too repetitive (only {len(set(question_tokens))}/{len(question_tokens)} unique tokens)")
        
        # Check answer length (min/max)
        answer_tokens = answer.split()
        if len(answer_tokens) < 5:
            reasons.append(f"Answer too short ({len(answer_tokens)} tokens, min 5)")
        if len(answer_tokens) > 500:
            reasons.append(f"Answer too long ({len(answer_tokens)} tokens, max 500)")
        
        # Check for excessive repetition in answer
        if answer_tokens and len(answer_tokens) > 5:
            # Flag if any single token appears > 30% of the time
            token_counts = Counter(answer_tokens)
            max_token_ratio = max(token_counts.values()) / len(answer_tokens)
            if max_token_ratio > 0.3:
                reasons.append(f"Answer is overly repetitive (token '{token_counts.most_common(1)[0][0]}' appears {max_token_ratio*100:.0f}% of the time)")
        
        # Check for hedging/uncertainty phrases
        hedging_phrases = ["maybe", "i think", "not sure", "probably", "i'm not sure", "might be", "could be", "possibly"]
        answer_lower = answer.lower()
        found_hedging = [p for p in hedging_phrases if p in answer_lower]
        if found_hedging:
            reasons.append(f"Hedging phrases detected: {', '.join(found_hedging)}")
        
        # Check for echo/repetition (answer is too similar to question)
        if question and answer:
            sim_ratio = SequenceMatcher(None, question.lower(), answer.lower()).ratio()
            if sim_ratio > 0.7:
                reasons.append(f"Answer echoes question (similarity {sim_ratio:.2f}, max 0.7)")
        
        # Check for trivial answers
        trivial_answers = {"yes", "no", "true", "false", "ok", "sure", "i don't know", "unknown"}
        if answer.lower() in trivial_answers:
            reasons.append(f"Trivial answer: '{answer}'")
        
        # If any reasons, fail
        if reasons:
            return (False, reasons)
        
        return (True, [])

    def self_chat_session(
        self,
        initial_topic: str,
        num_turns: int = 5,
        questions_per_turn: int = 1,
        q_temp: float = 0.8,
        q_top_p: float = 0.9,
        a_temp: float = 0.6,
        a_top_p: float = 0.9,
        novelty_threshold: float = 0.8,
        max_total_pairs: Optional[int] = None,
        stop_on_consecutive_no_new: int = 3,
        verify: bool = False,
        verify_method: str = "local",
        use_qa_context: bool = False,
        qa_context_depth: int = 2,
        include_reflection: bool = False
    ) -> List[Dict[str, str]]:
        """Conduct a self-chat session with configurable sampling and novelty filtering.

        Novelty filtering uses a simple sequence-similarity heuristic (difflib).
        """
        from difflib import SequenceMatcher

        conversation: List[Dict[str, str]] = []
        seen_texts: List[str] = []

        current_topic = initial_topic
        consecutive_no_new = 0

        for turn in range(num_turns):
            if max_total_pairs and len(conversation) >= max_total_pairs:
                break

            # Generate multiple candidate questions. Optionally include the previous Q/A to produce follow-up questions.
            if use_qa_context and conversation:
                # include up to qa_context_depth most recent Q/A pairs (oldest->newest)
                depth = max(1, int(qa_context_depth))
                last_n = conversation[-depth:]
                context_parts = []
                for i, t in enumerate(last_n, start=1):
                    q_text = t.get('question', '').strip()
                    a_text = t.get('answer', '').strip()
                    part = f"Q{i}: {q_text}\nA{i}: {a_text}"
                    if include_reflection:
                        refl = t.get('reflection', '').strip()
                        if refl:
                            part += f"\nR{i}: {refl}"
                    context_parts.append(part)
                context_block = "\n".join(context_parts)
                question_prompt = (
                    f"Based on the topic '{current_topic}' and the recent conversation:\n"
                    f"{context_block}\n"
                    "What is a good follow-up question a curious child would ask next? Question:"
                )
            else:
                question_prompt = f"Based on '{current_topic}', what question would a curious child ask? Question:"
            candidates_raw = self.generate_from_model(question_prompt, max_new_tokens=64, temperature=q_temp, top_p=q_top_p, num_return_sequences=questions_per_turn)

            # split candidates if multiple sequences returned joined by newline
            if questions_per_turn == 1:
                candidates = [candidates_raw.strip()]
            else:
                candidates = [c.strip() for c in candidates_raw.split('\n') if c.strip()]

            # Normalize and ensure they look like questions
            processed_questions: List[str] = []
            for q in candidates:
                if q.startswith("Question:"):
                    q = q.split("Question:", 1)[1].strip()
                q = q.strip()
                if not q.endswith('?'):
                    q = q + '?'
                processed_questions.append(q)

            # Deduplicate and novelty filter
            new_questions: List[str] = []
            for q in processed_questions:
                is_similar = False
                for seen in seen_texts:
                    sim = SequenceMatcher(None, q.lower(), seen.lower()).ratio()
                    if sim >= novelty_threshold:
                        is_similar = True
                        break
                if not is_similar:
                    new_questions.append(q)

            if not new_questions:
                consecutive_no_new += 1
            else:
                consecutive_no_new = 0

            # Stop condition
            if stop_on_consecutive_no_new and consecutive_no_new >= stop_on_consecutive_no_new:
                break

            # For each new question, generate answer and reflection and append
            for question in (new_questions if new_questions else processed_questions[:1]):
                # Answer
                answer_prompt = f"Topic: {current_topic}\nQuestion: {question}\nAnswer step by step like a child learning:"
                answer = self.generate_from_model(answer_prompt, max_new_tokens=256, temperature=a_temp, top_p=a_top_p)

                # Reflection
                reflection_prompt = f"I asked: {question}\nI answered: {answer}\nWhat did I learn from this? Why is this important?"
                reflection = self.generate_from_model(reflection_prompt, max_new_tokens=120, temperature=0.6, top_p=0.9)

                turn_data = {
                    "turn": len(conversation) + 1,
                    "topic": current_topic,
                    "question": question,
                    "answer": answer.strip(),
                    "reflection": reflection.strip()
                }

                # Verify turn if requested
                if verify and verify_method == "local":
                    is_valid, reasons = self.validate_turn(turn_data)
                    turn_data["validated"] = is_valid
                    turn_data["rejection_reasons"] = reasons if not is_valid else []
                    if not is_valid:
                        print(f"\n  ✗ Turn {turn_data['turn']} rejected:")
                        for reason in reasons:
                            print(f"    - {reason}")
                        # Skip appending invalid turn
                        continue
                else:
                    turn_data["validated"] = True
                    turn_data["rejection_reasons"] = []

                conversation.append(turn_data)
                seen_texts.append(question)

                if max_total_pairs and len(conversation) >= max_total_pairs:
                    break

            # Update topic for next turn: include questions to drive curiosity
            if new_questions:
                current_topic = f"{current_topic} - {new_questions[0]}"
            else:
                # fallback: use the top processed question
                current_topic = f"{current_topic} - {processed_questions[0]}"

        return conversation

    def save_conversation(self, conversation: List[Dict[str, str]], validated_only: bool = True) -> None:
        """Save self-chat conversation to training data. 
        Validated turns go to data/training_data.jsonl.
        Rejected turns (if validated_only=True and has rejection_reasons) go to data/rejected_training_data.jsonl.
        """
        filename = f"{self.data_dir}/training_data.jsonl"
        rejected_filename = f"{self.data_dir}/rejected_training_data.jsonl"

        for turn in conversation:
            is_validated = turn.get("validated", True)
            rejection_reasons = turn.get("rejection_reasons", [])
            
            data = {
                "text": f"Topic: {turn['topic']}\nQuestion: {turn['question']}\nAnswer: {turn['answer']}\nReflection: {turn['reflection']}",
                "source": "self_chat",
                "type": "self_learning",
                "turn": turn["turn"],
                "validated": is_validated,
                "rejection_reasons": rejection_reasons
            }

            if is_validated or not rejection_reasons:
                # Save to main training data
                with open(filename, 'a', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
                    f.write('\n')
            else:
                # Save to rejected data for review
                with open(rejected_filename, 'a', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
                    f.write('\n')

if __name__ == "__main__":
    engine = SelfChatEngine()

    # Example self-chat session
    conversation = engine.self_chat_session("Why do stars twinkle?", num_turns=3)
    engine.save_conversation(conversation)

    print("Self-chat session completed and saved.")
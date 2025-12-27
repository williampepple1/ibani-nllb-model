"""
Evaluation script for Ibani-English translation model.
Computes BLEU scores and other metrics.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sacrebleu import corpus_bleu
from tqdm import tqdm


class ModelEvaluator:
    """Evaluate translation model performance."""
    
    def __init__(self, model_path: str, base_model: str = "facebook/nllb-200-distilled-600M", use_lora: bool = True):
        self.model_path = model_path
        self.base_model = base_model
        self.use_lora = use_lora
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 Device: {self.device}")
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer."""
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"📥 Loading model from: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.use_lora:
            print("🔧 Loading with LoRA adapters...")
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model = self.model.merge_and_unload()
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
        
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded!\n")
    
    def translate(self, text: str, max_length: int = 200, num_beams: int = 5) -> str:
        """Translate a single text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def evaluate(self, test_data: List[Dict[str, str]], direction: str = "eng2iba") -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            test_data: List of dicts with 'ibani' and 'english' keys
            direction: 'eng2iba' or 'iba2eng'
        """
        print(f"📊 Evaluating {direction} translation...")
        print(f"📝 Test examples: {len(test_data)}\n")
        
        sources = []
        references = []
        hypotheses = []
        
        # Prepare data based on direction
        for item in test_data:
            if direction == "eng2iba":
                sources.append(item['english'])
                references.append(item['ibani'])
            else:  # iba2eng
                sources.append(item['ibani'])
                references.append(item['english'])
        
        # Translate all sources
        print("🔄 Translating...")
        for source in tqdm(sources):
            translation = self.translate(source)
            hypotheses.append(translation)
        
        # Compute BLEU score
        bleu = corpus_bleu(hypotheses, [references])
        
        # Compute additional metrics
        exact_matches = sum(1 for h, r in zip(hypotheses, references) if h.strip().lower() == r.strip().lower())
        exact_match_rate = exact_matches / len(test_data) * 100
        
        results = {
            "direction": direction,
            "num_examples": len(test_data),
            "bleu_score": bleu.score,
            "exact_matches": exact_matches,
            "exact_match_rate": exact_match_rate,
        }
        
        # Print results
        print("\n" + "=" * 60)
        print(f"📊 Evaluation Results ({direction})")
        print("=" * 60)
        print(f"BLEU Score: {bleu.score:.2f}")
        print(f"Exact Matches: {exact_matches}/{len(test_data)} ({exact_match_rate:.1f}%)")
        print("=" * 60 + "\n")
        
        # Show some examples
        print("📝 Sample Translations:")
        print("-" * 60)
        for i in range(min(5, len(sources))):
            print(f"\nSource:      {sources[i]}")
            print(f"Reference:   {references[i]}")
            print(f"Translation: {hypotheses[i]}")
            print("-" * 60)
        
        return results, hypotheses


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ibani-English translation model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/ibani-nllb",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="ibani_eng_training_data.json",
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="Base NLLB model"
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA loading"
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="both",
        choices=["eng2iba", "iba2eng", "both"],
        help="Translation direction to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Load test data
    print(f"📚 Loading test data from: {args.test_data}")
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"✓ Loaded {len(test_data)} test examples\n")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        base_model=args.base_model,
        use_lora=not args.no_lora
    )
    
    # Run evaluation
    all_results = {}
    
    if args.direction in ["eng2iba", "both"]:
        results, hypotheses = evaluator.evaluate(test_data, "eng2iba")
        all_results["eng2iba"] = results
        all_results["eng2iba_translations"] = hypotheses
    
    if args.direction in ["iba2eng", "both"]:
        results, hypotheses = evaluator.evaluate(test_data, "iba2eng")
        all_results["iba2eng"] = results
        all_results["iba2eng_translations"] = hypotheses
    
    # Save results if output file specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()

"""
Interactive CLI translator for Ibani-English translation.
"""

import os
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


class IbaniTranslator:
    """Ibani-English translator using fine-tuned NLLB model."""
    
    def __init__(self, model_path: str, base_model: str = "facebook/nllb-200-distilled-600M", use_lora: bool = True):
        self.model_path = model_path
        self.base_model = base_model
        self.use_lora = use_lora
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 Device: {self.device}")
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer."""
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            print(f"⚠️  Model not found at {self.model_path}")
            print(f"📥 Loading base model: {self.base_model}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
        else:
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
    
    def translate(
        self,
        text: str,
        source_lang: str = "eng",
        target_lang: str = "iba",
        max_length: int = 200,
        num_beams: int = 5
    ) -> str:
        """Translate text."""
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                do_sample=False,
            )
        
        # Decode
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation


def interactive_mode(translator: IbaniTranslator):
    """Run interactive translation mode."""
    print("=" * 60)
    print("🌍 Ibani-English Interactive Translator")
    print("=" * 60)
    print("Commands:")
    print("  - Type text to translate (English → Ibani by default)")
    print("  - 'swap' to switch translation direction")
    print("  - 'quit' or 'exit' to quit")
    print("=" * 60 + "\n")
    
    source_lang = "eng"
    target_lang = "iba"
    
    while True:
        direction = f"{source_lang.upper()} → {target_lang.upper()}"
        user_input = input(f"\n[{direction}] Enter text: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == "swap":
            source_lang, target_lang = target_lang, source_lang
            print(f"✓ Switched to {source_lang.upper()} → {target_lang.upper()}")
            continue
        
        try:
            translation = translator.translate(
                user_input,
                source_lang=source_lang,
                target_lang=target_lang
            )
            print(f"\n✨ Translation: {translation}")
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Ibani-English CLI Translator")
    parser.add_argument(
        "--model",
        type=str,
        default="models/ibani-nllb",
        help="Path to fine-tuned model"
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
        "--text",
        type=str,
        help="Text to translate (non-interactive mode)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="eng",
        choices=["eng", "iba"],
        help="Source language"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="iba",
        choices=["eng", "iba"],
        help="Target language"
    )
    
    args = parser.parse_args()
    
    # Initialize translator
    translator = IbaniTranslator(
        model_path=args.model,
        base_model=args.base_model,
        use_lora=not args.no_lora
    )
    
    # Run in appropriate mode
    if args.text:
        # Single translation mode
        translation = translator.translate(
            args.text,
            source_lang=args.source,
            target_lang=args.target
        )
        print(f"\n📝 Input: {args.text}")
        print(f"✨ Translation: {translation}\n")
    else:
        # Interactive mode
        interactive_mode(translator)


if __name__ == "__main__":
    main()

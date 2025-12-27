"""
Quick test script to verify the model works correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.translate import IbaniTranslator


def main():
    print("=" * 60)
    print("🧪 Quick Model Test")
    print("=" * 60)
    
    # Initialize translator
    translator = IbaniTranslator(
        model_path="models/ibani-nllb",
        base_model="facebook/nllb-200-distilled-600M",
        use_lora=True
    )
    
    # Test cases
    test_cases = [
        ("Hello", "eng", "iba"),
        ("How are you?", "eng", "iba"),
        ("Thank you", "eng", "iba"),
    ]
    
    print("\n📝 Running test translations:\n")
    
    for text, src, tgt in test_cases:
        translation = translator.translate(text, source_lang=src, target_lang=tgt)
        print(f"{src.upper()} → {tgt.upper()}: {text}")
        print(f"Translation: {translation}\n")
    
    print("=" * 60)
    print("✅ Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

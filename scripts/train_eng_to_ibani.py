"""
Training script for fine-tuning NLLB-200 on English-to-Ibani translation only.
This is a lighter version that only trains the English → Ibani direction.
Supports both full fine-tuning and LoRA for efficient training.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType


def load_training_data(data_path: str) -> List[Dict[str, str]]:
    """Load training data from JSON file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} training examples")
    return data


def prepare_dataset(data: List[Dict[str, str]], tokenizer, source_lang: str = 'eng', target_lang: str = 'iba'):
    """Prepare dataset for English → Ibani training only."""
    
    # Create unidirectional examples (English → Ibani only)
    examples = []
    
    for item in data:
        # Extract text fields (handle multiple formats)
        # Format 1: Nested translation object {"translation": {"en": "...", "ibani": "..."}}
        if 'translation' in item:
            ibani = item['translation'].get('ibani', '')
            english = item['translation'].get('en', '')
        # Format 2: Direct fields with _text suffix
        elif 'ibani_text' in item or 'english_text' in item:
            ibani = item.get('ibani_text', '')
            english = item.get('english_text', '')
        # Format 3: Simple direct fields
        else:
            ibani = item.get('ibani', '')
            english = item.get('english', '')
        
        # Skip empty entries
        if not ibani or not english:
            continue
        
        # English → Ibani only
        examples.append({
            'source': english,
            'target': ibani,
            'source_lang': 'eng_Latn',
            'target_lang': 'iba_Latn'  # NLLB language code for Ibani (custom)
        })
    
    print(f"✓ Created {len(examples)} English→Ibani training examples")
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(examples)
    
    def preprocess_function(examples):
        """Tokenize the examples."""
        # NLLB uses special tokens for language codes

        # or train with eng_Latn and fine-tune
        
        inputs = [ex for ex in examples['source']]
        targets = [ex for ex in examples['target']]
        
        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=128,
            truncation=True,
            padding='max_length'
        )
        
        # Tokenize targets
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding='max_length'
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def setup_lora_model(model, use_lora: bool = True):
    """Setup LoRA for efficient fine-tuning."""
    if not use_lora:
        return model
    
    print("✓ Setting up LoRA for efficient training...")
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def train(
    data_path: str,
    output_dir: str,
    model_name: str = "facebook/nllb-200-distilled-600M",
    use_lora: bool = True,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    eval_split: float = 0.1,
):
    """Main training function for English → Ibani only."""
    
    print("=" * 60)
    print("🚀 English→Ibani NLLB Training (Unidirectional)")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer and model
    print(f"\n📥 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Setup LoRA if requested
    if use_lora:
        model = setup_lora_model(model, use_lora)
    
    # Load and prepare data
    print(f"\n📚 Loading training data from: {data_path}")
    data = load_training_data(data_path)
    dataset = prepare_dataset(data, tokenizer, 'eng', 'iba')
    
    # Split into train/eval
    split_dataset = dataset.train_test_split(test_size=eval_split, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"✓ Train examples: {len(train_dataset)}")
    print(f"✓ Eval examples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to=["tensorboard"],
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("\n🏋️ Starting training...")
    print("=" * 60)
    trainer.train()
    
    # Save final model
    print(f"\n💾 Saving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    info = {
        "base_model": model_name,
        "direction": "eng_to_iba",
        "use_lora": use_lora,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_training_examples": len(data),
        "num_unidirectional_examples": len(dataset),
    }
    
    with open(f"{output_dir}/training_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"📁 Model saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train NLLB model for English→Ibani translation only")
    parser.add_argument(
        "--data",
        type=str,
        default="ibani_eng.json",
        help="Path to training data JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/ibani-nllb-eng2iba",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="Base NLLB model to fine-tune"
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (use full fine-tuning)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for evaluation"
    )
    
    args = parser.parse_args()
    
    train(
        data_path=args.data,
        output_dir=args.output,
        model_name=args.model,
        use_lora=not args.no_lora,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_split=args.eval_split,
    )


if __name__ == "__main__":
    main()

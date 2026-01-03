"""
FastAPI application for English-to-Ibani translation using the fine-tuned NLLB model.
This server specifically serves the unidirectional English→Ibani model.
"""

import os
from typing import Optional
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/my-eng2iba-model")
BASE_MODEL = os.getenv("BASE_MODEL", "facebook/nllb-200-distilled-600M")

# Initialize FastAPI app
app = FastAPI(
    title="English-to-Ibani Translator API",
    description="Translation API powered by fine-tuned NLLB-200 model (English→Ibani only)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class TranslationRequest(BaseModel):
    text: str = Field(..., description="English text to translate to Ibani", min_length=1)
    max_length: Optional[int] = Field(200, description="Maximum length of translation")
    num_beams: Optional[int] = Field(5, description="Number of beams for beam search")


class BatchTranslationRequest(BaseModel):
    texts: list[str] = Field(..., description="List of English texts to translate", min_items=1)
    max_length: Optional[int] = Field(200, description="Maximum length of translation")
    num_beams: Optional[int] = Field(5, description="Number of beams for beam search")


class TranslationResponse(BaseModel):
    translation: str = Field(..., description="Translated Ibani text")
    source_text: str = Field(..., description="Original English text")
    model: str = Field(..., description="Model used for translation")


class BatchTranslationResponse(BaseModel):
    translations: list[str] = Field(..., description="List of translated Ibani texts")
    source_texts: list[str] = Field(..., description="Original English texts")
    model: str = Field(..., description="Model used for translation")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    device: str
    direction: str


# Global model and tokenizer
tokenizer = None
model = None
device = None


def load_model():
    """Load the fine-tuned English→Ibani model and tokenizer."""
    global tokenizer, model, device
    
    print("=" * 60)
    print("🚀 Loading English→Ibani Translation Model")
    print("=" * 60)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Device: {device}")
    
    # Check if model exists
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Please train the model first using scripts/train_eng_to_ibani.py"
        )
    
    print(f"📥 Loading fine-tuned model from: {MODEL_PATH}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Check if this is a LoRA model
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        print("🔧 Loading with LoRA adapters...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            local_files_only=True,  # Use cached model, don't re-download
        )
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model = model.merge_and_unload()  # Merge LoRA weights for faster inference
    else:
        print("📦 Loading full fine-tuned model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    
    # Move to device                                                                                                                                                                                                    
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    print("==" * 60)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "English-to-Ibani Translation API",
        "version": "1.0.0",
        "direction": "eng → iba",
        "endpoints": {
            "translate": "/translate",
            "batch_translate": "/batch-translate",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path=MODEL_PATH,
        device=str(device),
        direction="eng → iba"
    )


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Translate English text to Ibani.
    
    Example:
    ```json
    {
        "text": "Hello, how are you?",
        "max_length": 200,
        "num_beams": 5
    }
    ```
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                num_beams=request.num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                do_sample=False,
            )
        
        # Decode output
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return TranslationResponse(
            translation=translation,
            source_text=request.text,
            model=MODEL_PATH
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


@app.post("/batch-translate", response_model=BatchTranslationResponse)
async def batch_translate(request: BatchTranslationRequest):
    """
    Translate multiple English texts to Ibani in a batch.
    
    Example:
    ```json
    {
        "texts": ["Hello", "How are you?", "Good morning"],
        "max_length": 200,
        "num_beams": 5
    }
    ```
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize all inputs
        inputs = tokenizer(
            request.texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        
        # Generate translations
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                num_beams=request.num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                do_sample=False,
            )
        
        # Decode all outputs
        translations = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return BatchTranslationResponse(
            translations=translations,
            source_texts=request.texts,
            model=MODEL_PATH
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch translation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    
    print("\n" + "=" * 60)
    print("🌐 Starting English→Ibani Translation API")
    print("=" * 60)
    print(f"📍 URL: http://localhost:{port}")
    print(f"📚 Docs: http://localhost:{port}/docs")
    print(f"🔄 Direction: English → Ibani only")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

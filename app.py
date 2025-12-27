"""
FastAPI application for Ibani-English translation using fine-tuned NLLB model.
"""

import os
from typing import Optional, Literal
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/ibani-nllb")
BASE_MODEL = os.getenv("BASE_MODEL", "facebook/nllb-200-distilled-600M")
USE_LORA = os.getenv("USE_LORA", "true").lower() == "true"

# Language codes
LANG_CODES = {
    "iba": "iba_Latn",  # Ibani (custom)
    "eng": "eng_Latn",  # English
}

# Initialize FastAPI app
app = FastAPI(
    title="Ibani-English Translator API",
    description="Translation API powered by fine-tuned NLLB-200 model",
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
    text: str = Field(..., description="Text to translate", min_length=1)
    source_lang: Literal["iba", "eng"] = Field(..., description="Source language code")
    target_lang: Literal["iba", "eng"] = Field(..., description="Target language code")
    max_length: Optional[int] = Field(200, description="Maximum length of translation")
    num_beams: Optional[int] = Field(5, description="Number of beams for beam search")


class TranslationResponse(BaseModel):
    translation: str = Field(..., description="Translated text")
    source_lang: str = Field(..., description="Source language code")
    target_lang: str = Field(..., description="Target language code")
    model: str = Field(..., description="Model used for translation")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    device: str


# Global model and tokenizer
tokenizer = None
model = None
device = None


def load_model():
    """Load the fine-tuned model and tokenizer."""
    global tokenizer, model, device
    
    print("=" * 60)
    print("🚀 Loading Ibani-English Translation Model")
    print("=" * 60)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Device: {device}")
    
    # Check if model exists
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"⚠️  Model not found at {MODEL_PATH}")
        print(f"📥 Loading base model: {BASE_MODEL}")
        print("⚠️  Note: This is the base model, not fine-tuned for Ibani!")
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    else:
        print(f"📥 Loading fine-tuned model from: {MODEL_PATH}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # Load model (with LoRA if applicable)
        if USE_LORA:
            print("🔧 Loading with LoRA adapters...")
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            model = PeftModel.from_pretrained(base_model, MODEL_PATH)
            model = model.merge_and_unload()  # Merge LoRA weights
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
    
    # Move to device
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    print("=" * 60)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Ibani-English Translation API",
        "version": "1.0.0",
        "endpoints": {
            "translate": "/translate",
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
        device=str(device)
    )


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Translate text between Ibani and English.
    
    Example:
    ```json
    {
        "text": "Hello, how are you?",
        "source_lang": "eng",
        "target_lang": "iba"
    }
    ```
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if request.source_lang == request.target_lang:
        raise HTTPException(
            status_code=400,
            detail="Source and target languages must be different"
        )
    
    try:
        # Get NLLB language codes
        src_lang = LANG_CODES.get(request.source_lang)
        tgt_lang = LANG_CODES.get(request.target_lang)
        
        # For NLLB, we use forced_bos_token_id to specify target language
        # Note: Since Ibani might not be in NLLB's original vocabulary,
        # we'll use the model's learned behavior from fine-tuning
        
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
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            model=MODEL_PATH if Path(MODEL_PATH).exists() else BASE_MODEL
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


@app.post("/batch-translate", response_model=list[TranslationResponse])
async def batch_translate(requests: list[TranslationRequest]):
    """
    Translate multiple texts in a batch.
    
    Example:
    ```json
    [
        {
            "text": "Hello",
            "source_lang": "eng",
            "target_lang": "iba"
        },
        {
            "text": "How are you?",
            "source_lang": "eng",
            "target_lang": "iba"
        }
    ]
    ```
    """
    results = []
    for req in requests:
        result = await translate(req)
        results.append(result)
    return results


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    
    print("\n" + "=" * 60)
    print("🌐 Starting Ibani-English Translation API")
    print("=" * 60)
    print(f"📍 URL: http://localhost:{port}")
    print(f"📚 Docs: http://localhost:{port}/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

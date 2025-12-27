# Ibani-English NLLB Translator

A state-of-the-art translation model for Ibani ↔ English using Meta's No Language Left Behind (NLLB-200) architecture.

## 🌟 Features

- **Bidirectional Translation**: Ibani → English and English → Ibani
- **NLLB-200 Based**: Leverages Meta's multilingual model fine-tuned for Ibani
- **Tonal Mark Support**: Properly handles Ibani special characters (á, ḅ, etc.)
- **FastAPI Backend**: Production-ready REST API
- **Google Colab Training**: Train on free GPU resources
- **Local Inference**: Run the model on your machine

## 📋 Requirements

- Python 3.10+ (3.11 recommended for training)
- 8GB+ RAM for inference
- GPU recommended for training (Colab/Kaggle provides free GPUs)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/williampepple1/ibani-nllb-model.git
cd ibani-nllb-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Training Data

Your `ibani_eng_training_data.json` file should already be in the root directory (Bible translation data):

```json
[
  {
    "translation": {
      "en": "This is the genealogy of Jesus the Messiah...",
      "ibani": "Mịị anịị diri bie anị fịnị ḅara Jizọs tádọ́apụ..."
    }
  }
]
```

The scripts automatically handle multiple formats:
- `{"translation": {"en": "...", "ibani": "..."}}` (your current format) ✅
- `{"ibani_text": "...", "english_text": "..."}` (Bible format with metadata)
- `{"ibani": "...", "english": "..."}` (simple format)

### 3. Train the Model

**Option A: Google Colab (Recommended)**
1. Open `notebooks/train_ibani_nllb.ipynb` in Google Colab
2. Upload your training data
3. Run all cells
4. Download the trained model

**Option B: Local Training**
```bash
python scripts/train.py --data ibani_eng_training_data.json --output models/ibani-nllb
```

### 4. Run the API

```bash
python app.py
```

The API will be available at `http://localhost:8080`

### 5. Test Translation

```bash
curl -X POST "http://localhost:8080/translate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?", "source_lang": "eng", "target_lang": "iba"}'
```

## 📁 Project Structure

```
ibani-nllb-model/
├── app.py                          # FastAPI application
├── scripts/
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Model evaluation
│   └── translate.py                # CLI translator
├── notebooks/
│   └── train_ibani_nllb.ipynb     # Colab training notebook
├── data/                           # Optional data directory
├── ibani_eng_training_data.json    # Training data (root)
├── models/                         # Trained models (gitignored)
├── requirements.txt
└── README.md
```

## 🔧 API Endpoints

The FastAPI server provides the following endpoints:

### 📍 **GET /** - Root Endpoint
Get API information and available endpoints.

```bash
curl http://localhost:8080/
```

**Response:**
```json
{
  "message": "Ibani-English Translation API",
  "version": "1.0.0",
  "endpoints": {
    "translate": "/translate",
    "health": "/health",
    "docs": "/docs"
  }
}
```

---

### 📍 **GET /health** - Health Check
Check if the model is loaded and API is healthy.

```bash
curl http://localhost:8080/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/ibani-nllb",
  "device": "cuda"
}
```

---

### 📍 **POST /translate** - Single Translation
Translate text between Ibani and English.

**Request:**
```bash
curl -X POST "http://localhost:8080/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_lang": "eng",
    "target_lang": "iba",
    "max_length": 200,
    "num_beams": 5
  }'
```

**Request Body:**
```json
{
  "text": "Hello, how are you?",
  "source_lang": "eng",
  "target_lang": "iba",
  "max_length": 200,
  "num_beams": 5
}
```

**Parameters:**
- `text` (required): Text to translate
- `source_lang` (required): Source language (`eng` or `iba`)
- `target_lang` (required): Target language (`eng` or `iba`)
- `max_length` (optional): Maximum translation length (default: 200)
- `num_beams` (optional): Number of beams for beam search (default: 5)

**Response:**
```json
{
  "translation": "Translated text here",
  "source_lang": "eng",
  "target_lang": "iba",
  "model": "models/ibani-nllb"
}
```

---

### 📍 **POST /batch-translate** - Batch Translation
Translate multiple texts in a single request.

**Request:**
```bash
curl -X POST "http://localhost:8080/batch-translate" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "text": "Hello",
      "source_lang": "eng",
      "target_lang": "iba"
    },
    {
      "text": "Thank you",
      "source_lang": "eng",
      "target_lang": "iba"
    }
  ]'
```

**Request Body:**
```json
[
  {
    "text": "Hello",
    "source_lang": "eng",
    "target_lang": "iba"
  },
  {
    "text": "Thank you",
    "source_lang": "eng",
    "target_lang": "iba"
  }
]
```

**Response:**
```json
[
  {
    "translation": "First translation",
    "source_lang": "eng",
    "target_lang": "iba",
    "model": "models/ibani-nllb"
  },
  {
    "translation": "Second translation",
    "source_lang": "eng",
    "target_lang": "iba",
    "model": "models/ibani-nllb"
  }
]
```

---

### 📚 **Interactive API Documentation**

Once the server is running, access interactive documentation:

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

These provide a web interface to test all endpoints directly in your browser!

---

### 🎯 **Language Codes**

- **English**: `eng`
- **Ibani**: `iba`

---

### 💡 **Python Client Example**

```python
import requests

# Single translation
response = requests.post(
    "http://localhost:8080/translate",
    json={
        "text": "Hello, how are you?",
        "source_lang": "eng",
        "target_lang": "iba"
    }
)
result = response.json()
print(f"Translation: {result['translation']}")

# Batch translation
response = requests.post(
    "http://localhost:8080/batch-translate",
    json=[
        {"text": "Hello", "source_lang": "eng", "target_lang": "iba"},
        {"text": "Thank you", "source_lang": "eng", "target_lang": "iba"}
    ]
)
results = response.json()
for item in results:
    print(f"Translation: {item['translation']}")
```

---

### ⚙️ **Environment Configuration**

Create a `.env` file to configure the API:

```bash
MODEL_PATH=models/ibani-nllb
BASE_MODEL=facebook/nllb-200-distilled-600M
USE_LORA=true
PORT=8080
```

## 🎯 Model Details

- **Base Model**: `facebook/nllb-200-distilled-600M`
- **Fine-tuned for**: Ibani (iba) ↔ English (eng)
- **Training Method**: Supervised fine-tuning with LoRA
- **Special Features**: Custom handling for Ibani tonal marks

## 📊 Performance

| Language Pair | BLEU Score | Training Examples |
|--------------|------------|-------------------|
| Ibani → English | TBD | TBD |
| English → Ibani | TBD | TBD |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License

## 🙏 Acknowledgments

- Meta AI for the NLLB-200 model
- Hugging Face for the transformers library
- The Ibani language community

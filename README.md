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

### POST `/translate`
Translate text between Ibani and English.

**Request:**
```json
{
  "text": "Hello world",
  "source_lang": "eng",
  "target_lang": "iba"
}
```

**Response:**
```json
{
  "translation": "Ndewo ụwa",
  "source_lang": "eng",
  "target_lang": "iba"
}
```

### GET `/health`
Check API health status.

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

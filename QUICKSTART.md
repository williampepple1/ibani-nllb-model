# Quick Start Guide

## 🎯 Goal
Train an Ibani-English translation model using Meta's NLLB-200 and deploy it locally.

## 📋 Prerequisites
- Python 3.10 or 3.11
- Your `ibani_eng.json` training data
- Google account (for Colab training)

## 🚀 Steps

### 1. Setup Local Environment

```bash
# Navigate to project
cd ibani-nllb-model

# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Your Training Data

Your `ibani_eng_training_data.json` file should already be in the root directory. Format:

```json
[
  {
    "ibani": "Ibani text with tonal marks",
    "english": "English translation"
  }
]
```

### 3. Train on Google Colab (Recommended)

**Why Colab?**
- Free GPU (T4)
- No local GPU required
- Faster training

**Steps:**
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `notebooks/train_ibani_nllb.ipynb`
3. Change runtime to GPU: Runtime → Change runtime type → GPU
4. Upload your `ibani_eng.json` when prompted
5. Run all cells
6. Download the trained model (will be a .zip file)

### 4. Use the Trained Model Locally

```bash
# Extract the downloaded model
# Place it in: models/ibani-nllb/

# Test the model
python test_model.py

# Start the API server
python app.py
```

### 5. Access the API

- **API**: http://localhost:8080
- **Docs**: http://localhost:8080/docs
- **Health**: http://localhost:8080/health

### 6. Test Translation

**Using the interactive docs:**
1. Go to http://localhost:8080/docs
2. Try the `/translate` endpoint
3. Example request:
```json
{
  "text": "Hello, how are you?",
  "source_lang": "eng",
  "target_lang": "iba"
}
```

**Using curl:**
```bash
curl -X POST "http://localhost:8080/translate" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Hello\", \"source_lang\": \"eng\", \"target_lang\": \"iba\"}"
```

**Using the CLI:**
```bash
# Interactive mode
python scripts/translate.py

# Single translation
python scripts/translate.py --text "Hello" --source eng --target iba
```

## 🔧 Troubleshooting

### Model not found
- Make sure you extracted the model to `models/ibani-nllb/`
- Check that the folder contains `config.json`, `adapter_config.json`, etc.

### Out of memory during training
- Reduce `BATCH_SIZE` in the Colab notebook (try 4 or 2)
- Use a smaller model: `facebook/nllb-200-distilled-600M` (default)

### Poor translation quality
- Add more training data
- Increase `NUM_EPOCHS` (try 15-20)
- Ensure your data has proper tonal marks

## 📊 Evaluation

To evaluate your model:

```bash
python scripts/evaluate.py --model models/ibani-nllb --test-data ibani_eng_training_data.json
```

This will show BLEU scores and sample translations.

## 🎓 Next Steps

1. **Improve the model**: Add more training data
2. **Deploy online**: Use Hugging Face Spaces or Railway
3. **Share**: Push to Hugging Face Hub
4. **Integrate**: Use the API in your applications

## 💡 Tips

- **More data = better model**: Aim for 1000+ sentence pairs
- **Quality over quantity**: Ensure translations are accurate
- **Tonal marks matter**: NLLB handles special characters well
- **Bidirectional training**: The model learns both directions automatically

## 🆘 Need Help?

Check the main README.md for detailed documentation.

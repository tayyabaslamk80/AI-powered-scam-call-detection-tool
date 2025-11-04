# AI-Powered Scam Call Identification Tool

Flask-based real-time call scam detection using Whisper and keyword analysis.

## Quick Start

**Activate environment and run:**
```bash
cd E:\FYP-ALI
venv\Scripts\activate
cd flask_sentinel
python app.py
```

**Open browser:** http://localhost:5000

**Login:** test@example.com / password123

## How to Run

1. Go to: `E:\FYP-ALI`
2. Activate venv: `venv\Scripts\activate`
3. Go to: `cd flask_sentinel`
4. Run: `python app.py`
5. Open: http://localhost:5000

## Features

✓ Real-time call audio analysis  
✓ Upload audio/video for scam detection  
✓ Whisper transcription (English & Hindi)  
✓ Dynamic keyword-based fraud detection  
✓ Contact management with risk levels  
✓ Call history tracking  

## Project Structure

```
E:\FYP-ALI\flask_sentinel\
├── app.py              # Flask application
├── services.py         # ML models integration
├── requirements.txt    # Dependencies
├── sentinel.db         # Database
├── templates/          # HTML pages
├── static/            # CSS & JS
└── uploads/           # Uploaded files

E:\FYP-ALI\Models\
├── Whisper_model/     # Transcription model
└── Fraud_model/       # Fraud detection
```

## Requirements

Python 3.9+, Flask, PyTorch, Transformers, Whisper, Librosa, NLTK

See `HOW_TO_RUN.txt` for detailed commands.


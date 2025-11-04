"""
Services Module - Integration with ML Models
Handles Whisper transcription, DistilBERT fraud detection, and feature extraction
"""

import os
import json
import logging
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    WhisperProcessor, 
    WhisperForConditionalGeneration
)
import librosa
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import subprocess
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data with specific download directory
import nltk
import os

# Set custom NLTK data directory
custom_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.environ['NLTK_DATA'] = custom_dir

# Download NLTK data to local directory
try:
    nltk.download('vader_lexicon', download_dir=custom_dir, quiet=True)
    nltk.download('punkt', download_dir=custom_dir, quiet=True)
    nltk.download('punkt_tab', download_dir=custom_dir, quiet=True)
    nltk.download('stopwords', download_dir=custom_dir, quiet=True)
except Exception as e:
    logger.warning(f"NLTK download failed: {e}")
    
# Add to nltk.data.path
nltk.data.path.insert(0, custom_dir)


class ModelService:
    """Service for loading and using ML models locally"""
    
    def __init__(self):
        # Models path - local directory
        # flask_sentinel is at E:\FYP-ALI\flask_sentinel, so Models is at E:\FYP-ALI\Models
        self.models_path = r"E:\FYP-ALI\Models"
        logger.info(f"Models path: {self.models_path}")
        
        self.fraud_model = None
        self.fraud_tokenizer = None
        self.whisper_model = None
        self.whisper_processor = None
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load only Whisper model (DistilBERT is optional but not working)
        self.load_whisper_model()
        # Note: Fraud model was not discriminating, using keyword-based detection only
        
        # Scam keywords - comprehensive list
        self.scam_keywords = {
            'authority': ['bank', 'government', 'police', 'court', 'irs', 'fbi', 'official', 'urgent', 'immediately', 'department', 'officer', 'agent', 'underwriting', 'security', 'compliance', 'audit', 'legal', 'attorney', 'federal', 'state', 'agency', 'bureau'],
            'urgency': ['urgent', 'immediately', 'now', 'asap', 'right now', 'hurry', 'quickly', 'emergency', 'before', 'deadline', 'expire', 'expired', 'expiring', 'last chance', 'final notice', 'only notice', 'limited time', 'act now', 'don\'t delay', 'today only', 'three days', 'business days', 'hours', 'minutes'],
            'threat': ['arrest', 'warrant', 'jail', 'prison', 'fine', 'penalty', 'suspended', 'blocked', 'frozen', 'charge', 'lawsuit', 'legal action', 'prosecute', 'investigation', 'violation', 'crime', 'fraud', 'illegal', 'criminal', 'debt', 'owing'],
            'bait': ['prize', 'winner', 'congratulations', 'free', 'gift', 'bonus', 'reward', 'lottery', 'claim', 'win', 'won', 'selected', 'eligible', 'interest rate reduction', 'special offer', 'exclusive', 'guaranteed', '100%'],
            'sensitivity': ['ssn', 'social security', 'credit card', 'bank account', 'account', 'debit', 'password', 'pin', 'personal', 'card number', 'cvv', 'expiry', 'balance', 'statement', 'routing', 'checking', 'saving', 'date of birth', 'dob', 'mother maiden', 'security question', 'verify', 'confirm', 'authentication', 'card member services', 'call back number'],
            'suspicious_patterns': ['limited time', 'fraudulent activity', 'unusual activity', 'verify account', 'confirm identity', 're-activate', 're-activation', 'compromise', 'breach', 'locked', 'shut down', 'terminate', 'close account']
        }
    
    def load_fraud_model(self):
        """Load DistilBERT fraud detection model from local directory"""
        try:
            model_path = os.path.join(self.models_path, 'Fraud_model')
            
            if not os.path.exists(model_path):
                logger.error(f"Model path does not exist: {model_path}")
                return False
            
            logger.info(f"Loading Fraud model from: {model_path}")
            self.fraud_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.fraud_model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            self.fraud_model.eval()
            logger.info("✓ Fraud detection model loaded from local directory")
            return True
        except Exception as e:
            logger.error(f"Failed to load fraud model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def load_whisper_model(self):
        """Load Whisper model for transcription from local directory"""
        try:
            model_path = os.path.join(self.models_path, 'Whisper_model')
            
            if not os.path.exists(model_path):
                logger.error(f"Model path does not exist: {model_path}")
                return False
            
            logger.info(f"Loading Whisper from: {model_path}")
            self.whisper_processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
            self.whisper_model.eval()
            logger.info("✓ Whisper model loaded from local directory")
            return True
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def convert_audio_to_wav(self, audio_path):
        """Convert audio to WAV format using FFmpeg"""
        try:
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav.close()
            
            subprocess.run([
                'ffmpeg', '-i', audio_path,
                '-ar', '16000',
                '-ac', '1',
                '-f', 'wav',
                '-y',
                temp_wav.name
            ], capture_output=True, timeout=30)
            
            return temp_wav.name
        except Exception as e:
            logger.error(f"FFmpeg conversion failed: {e}")
            return None
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio file using Whisper with proper format handling"""
        try:
            logger.info(f"Attempting to transcribe: {audio_path}")
            
            # Load local Whisper model first
            if not self.whisper_processor or not self.whisper_model:
                self.load_whisper_model()
            
            # Try to load audio with librosa - handle different formats
            audio = None
            sr = 16000
            
            try:
                # First try librosa
                audio, sr = librosa.load(audio_path, sr=16000, offset=0, duration=None)
                logger.info(f"Loaded with librosa: duration={len(audio)/sr:.2f}s, sr={sr}")
            except Exception as e:
                logger.warning(f"Librosa failed: {e}")
                
                # Try renaming .dat.unknown to .ogg (WhatsApp format)
                try:
                    if audio_path.endswith('.dat.unknown'):
                        temp_ogg = audio_path.replace('.dat.unknown', '.ogg')
                        import shutil
                        shutil.copy2(audio_path, temp_ogg)
                        
                        audio, sr = librosa.load(temp_ogg, sr=16000, offset=0, duration=None)
                        logger.info(f"Loaded after rename: duration={len(audio)/sr:.2f}s")
                        
                        # Clean up temp file
                        if os.path.exists(temp_ogg):
                            os.unlink(temp_ogg)
                    else:
                        return ""
                except Exception as e2:
                    logger.error(f"Renaming also failed: {e2}")
                    return ""
            
            # Process with local Whisper model - handle long audio by chunking
            if self.whisper_processor and self.whisper_model:
                # Chunk audio if longer than 30 seconds to avoid memory issues
                chunk_length = 30 * sr  # 30 seconds in samples
                transcriptions = []
                
                if len(audio) > chunk_length:
                    num_chunks = int(len(audio) / chunk_length) + 1
                    logger.info(f"Audio is {len(audio)/sr:.2f}s, splitting into {num_chunks} chunks of 30s")
                    
                    for chunk_idx in range(0, len(audio), chunk_length):
                        chunk = audio[chunk_idx:chunk_idx+chunk_length]
                        logger.info(f"Processing chunk {chunk_idx//chunk_length + 1}/{num_chunks}")
                        
                        input_features = self.whisper_processor(chunk, sampling_rate=sr, return_tensors="pt").input_features
                        
                        with torch.no_grad():
                            predicted_ids = self.whisper_model.generate(
                                input_features,
                                max_length=448,  # Longer sequence
                                num_beams=3,
                                early_stopping=False
                            )
                        
                        chunk_text = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                        if chunk_text:
                            logger.info(f"Chunk transcription: {chunk_text[:50]}")
                            transcriptions.append(chunk_text)
                
                else:
                    # Process entire audio
                    logger.info("Processing entire audio without chunking")
                    input_features = self.whisper_processor(audio, sampling_rate=sr, return_tensors="pt").input_features
                    
                    with torch.no_grad():
                        predicted_ids = self.whisper_model.generate(input_features)
                    
                    transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    transcriptions.append(transcription)
                
                full_transcription = " ".join(transcriptions)
                logger.info(f"✓ FULL Transcription ({len(full_transcription)} chars): {full_transcription[:200]}...")
                return full_transcription.strip()
            else:
                logger.error("Whisper model not loaded")
                return ""
                    
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def detect_fraud(self, text):
        """Detect fraud using DistilBERT model"""
        try:
            if not text.strip():
                return 0.0
            
            if not self.fraud_model or not self.fraud_tokenizer:
                if not self.load_fraud_model():
                    return 0.0
            
            # Tokenize
            inputs = self.fraud_tokenizer(text[:500], return_tensors="pt", truncation=True, max_length=512)
            
            # Predict
            with torch.no_grad():
                outputs = self.fraud_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Check model output shape
                logger.info(f"Model output shape: {logits.shape}, probabilities: {probs}")
                
                # Get probabilities for both classes
                if probs.shape[1] >= 2:
                    prob_class_0 = probs[0][0].item()
                    prob_class_1 = probs[0][1].item()
                    
                    logger.info(f"Class 0 prob: {prob_class_0:.6f}, Class 1 prob: {prob_class_1:.6f}")
                    
                    # Check if model is working or always extreme
                    if prob_class_0 < 0.01 and prob_class_1 > 0.99:
                        # Model always outputs extreme values - not discriminating
                        # Use the model output as-is: if class 1 is high = fraud
                        fraud_prob = prob_class_1
                        logger.info("Model output extreme - using class 1 as fraud indicator")
                    elif prob_class_1 > prob_class_0:
                        # Class 1 has higher probability - likely fraud
                        fraud_prob = prob_class_1
                    else:
                        # Class 0 has higher probability - likely not fraud
                        fraud_prob = 1.0 - prob_class_0  # Invert to get fraud probability
                else:
                    fraud_prob = probs[0][0].item()
                
                logger.info(f"Final fraud probability: {fraud_prob:.6f}")
                
                return fraud_prob
        except Exception as e:
            logger.error(f"Error detecting fraud: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.5
    
    def extract_linguistic_features(self, text):
        """Extract linguistic features from text"""
        try:
            text_lower = text.lower()
            words = word_tokenize(text_lower)
            sentences = sent_tokenize(text)
            
            features = {}
            
            # Check for ALL keywords in each category (case-insensitive)
            # Check if any keyword appears in the text (whole word or as substring)
            text_lower_full = text.lower()
            
            authority_words = sum(1 for kw in self.scam_keywords['authority'] if kw in text_lower_full)
            features['authority'] = 1.0 if authority_words > 0 else 0.0  # Binary: found or not
            
            urgency_words = sum(1 for kw in self.scam_keywords['urgency'] if kw in text_lower_full)
            features['urgency'] = 1.0 if urgency_words > 0 else 0.0
            
            threat_words = sum(1 for kw in self.scam_keywords['threat'] if kw in text_lower_full)
            features['threat'] = 1.0 if threat_words > 0 else 0.0
            
            bait_words = sum(1 for kw in self.scam_keywords['bait'] if kw in text_lower_full)
            features['bait'] = 1.0 if bait_words > 0 else 0.0
            
            sensitivity_words = sum(1 for kw in self.scam_keywords['sensitivity'] if kw in text_lower_full)
            features['sensitivity'] = 1.0 if sensitivity_words > 0 else 0.0
            
            suspicious_words = sum(1 for kw in self.scam_keywords['suspicious_patterns'] if kw in text_lower_full)
            features['suspicious_patterns'] = 1.0 if suspicious_words > 0 else 0.0
            
            logger.info(f"Detected: authority={authority_words}, urgency={urgency_words}, sensitivity={sensitivity_words}, suspicious={suspicious_words}")
            
            # Sentiment analysis
            sentiment = self.sia.polarity_scores(text)
            features['sentiment_negative'] = sentiment['neg']
            features['sentiment_compound'] = sentiment['compound']
            
            # Text statistics
            features['word_count'] = len(words)
            features['sentence_count'] = len(sentences)
            
            logger.info(f"Keyword matches - authority:{authority_words}, urgency:{urgency_words}, threat:{threat_words}, sensitivity:{sensitivity_words}")
            
            return features
        except Exception as e:
            logger.error(f"Error extracting linguistic features: {e}")
            return {}
    
    def analyze_text(self, text):
        """Full analysis pipeline: extract features + detect fraud"""
        # Extract linguistic features
        linguistic_features = self.extract_linguistic_features(text)
        
        logger.info(f"Linguistic features: authority={linguistic_features.get('authority', 0):.2f}, urgency={linguistic_features.get('urgency', 0):.2f}, sensitivity={linguistic_features.get('sensitivity', 0):.2f}")
        
        # Calculate fraud score based on ACTUAL content analysis
        # Weight features based on their scam indicators
        fraud_score = (
            linguistic_features.get('authority', 0) * 0.30 +      # Authority claims = high risk
            linguistic_features.get('sensitivity', 0) * 0.30 +    # Asking for sensitive info = high risk
            linguistic_features.get('urgency', 0) * 0.20 +         # Urgent pressure = risk
            linguistic_features.get('suspicious_patterns', 0) * 0.15 +  # Suspicious phrases
            linguistic_features.get('threat', 0) * 0.05             # Direct threats
        )
        
        logger.info(f"Calculated fraud score: {fraud_score:.4f} (based on content features)")
        
        # Combine features
        features = {
            **linguistic_features,
            'fraud_score_distilbert': 0.0  # Not using DistilBERT, using keyword-based detection
        }
        
        # Simple detection: SUSPICIOUS if fraud score > 30% OR authority + asking for sensitive info
        is_scam = (
            fraud_score > 0.3 or  # Any significant fraud indicators
            (linguistic_features.get('authority', 0) > 0 and linguistic_features.get('sensitivity', 0) > 0) or  # Authority + asking for info
            linguistic_features.get('suspicious_patterns', 0) > 0  # Any suspicious patterns
        )
        
        # Use calculated score (not broken model)
        scam_score = fraud_score
        
        logger.info(f"Final scam score: {scam_score:.4f}, is_scam: {is_scam}")
        
        # Determine risk level based on score and features
        # Thresholds: Safe < 30%, Fraud >= 30%
        if scam_score >= 0.3 or linguistic_features.get('sensitivity', 0) > 0 or linguistic_features.get('authority', 0) > 0:
            risk_level = "critical"
        else:
            risk_level = "safe"
        
        # Generate reasons
        reasons = []
        
        # Add fraud score reason
        reasons.append(f"🤖 AI Fraud Score: {(fraud_score * 100):.0f}%")
        
        # Find what specific keywords were detected in the text
        text_lower = text.lower()
        detected_keywords = {
            'authority': [kw for kw in self.scam_keywords['authority'] if kw in text_lower],
            'sensitivity': [kw for kw in self.scam_keywords['sensitivity'] if kw in text_lower],
            'urgency': [kw for kw in self.scam_keywords['urgency'] if kw in text_lower],
            'bait': [kw for kw in self.scam_keywords['bait'] if kw in text_lower],
            'threat': [kw for kw in self.scam_keywords['threat'] if kw in text_lower],
            'suspicious_patterns': [kw for kw in self.scam_keywords['suspicious_patterns'] if kw in text_lower]
        }
        
        # Generate specific cues based on what was actually detected
        cues = []
        
        if linguistic_features.get('authority', 0) > 0:
            reasons.append("🚨 Authority claims detected")
            if detected_keywords['authority']:
                cues.append(f"claiming authority: {', '.join(detected_keywords['authority'][:3])}")
        
        if linguistic_features.get('sensitivity', 0) > 0:
            reasons.append("🔐 Sensitive info requests detected")
            if detected_keywords['sensitivity']:
                cues.append(f"requesting info: {', '.join(detected_keywords['sensitivity'][:3])}")
        
        if linguistic_features.get('urgency', 0) > 0:
            reasons.append("⏰ Urgent/pressuring language used")
            if detected_keywords['urgency']:
                cues.append(f"urgency phrases: {', '.join(detected_keywords['urgency'][:3])}")
        
        if linguistic_features.get('bait', 0) > 0:
            reasons.append("🎁 Bait offers detected")
            if detected_keywords['bait']:
                cues.append(f"bait offers: {', '.join(detected_keywords['bait'][:2])}")
        
        if linguistic_features.get('threat', 0) > 0:
            reasons.append("⚖️ Threatening language detected")
            if detected_keywords['threat']:
                cues.append(f"threats: {', '.join(detected_keywords['threat'][:2])}")
        
        if linguistic_features.get('suspicious_patterns', 0) > 0:
            reasons.append("⚠️ Suspicious patterns detected")
            if detected_keywords['suspicious_patterns']:
                cues.append(f"patterns: {', '.join(detected_keywords['suspicious_patterns'][:2])}")
        
        # Add summarized cues with actual detected keywords
        if cues:
            reasons.append(f"✅ Cues: {', '.join(cues)}")
        elif fraud_score < 0.3:
            reasons.append("✅ Low risk indicators - no suspicious cues detected")
        
        return {
            'is_scam': is_scam,
            'scam_score': scam_score,
            'risk_level': risk_level,
            'reasons': reasons,
            'features': features
        }


# Global instance
model_service = ModelService()


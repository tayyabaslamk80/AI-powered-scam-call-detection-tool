# 🧠 AI-Powered Scam Call Detection Tool  

An intelligent, real-time **AI-powered system** designed to detect and prevent **scam and fraudulent phone calls** using advanced **speech recognition**, **machine learning**, and **natural language processing (NLP)** techniques.  

This tool can analyze both **live calls** and **recorded audio**, automatically classifying conversations as *safe* or *scam-related* based on linguistic and behavioral cues.  

---

## 🚀 Features  

- 🎙️ **Real-Time Call Analysis** – Detects scam activity during live phone calls.  
- 🔊 **Recorded Call Detection** – Analyzes uploaded or saved audio files for fraud indicators.  
- 🧩 **Speech-to-Text Conversion** – Uses **OpenAI Whisper** for accurate transcription of voice conversations.  
- 🤖 **Intelligent Classification** – Machine learning model trained to classify conversations as *Normal* or *Fraudulent*.  
- 💬 **NLP-Powered Understanding** – Detects scam keywords, tone, and suspicious dialogue patterns.  
- 🌐 **Web Interface** – Simple and responsive frontend built with **HTML**, **CSS**, and **Flask** backend integration.  
- 📊 **Results Dashboard** – Displays detection outcomes and model confidence levels.  

---

## 🏗️ System Architecture  

**Frontend:**  
- HTML  
- CSS  

**Backend:**  
- Flask (Python)  
- Whisper (Speech-to-Text Conversion)  
- Machine Learning Model (Classification Engine)  

**Workflow:**  
1. User initiates or uploads a call/audio file.  
2. Whisper converts the speech to text.  
3. The text is passed to the ML model for analysis.  
4. The model predicts whether the conversation is **safe** or **potentially fraudulent**.  
5. The result is displayed instantly on the frontend interface.  

---

## 🧠 Machine Learning Model  

The ML model was trained using a combination of **scam call datasets** and **normal conversation samples**.  
It extracts linguistic and contextual features such as:  
- Keyword frequency  
- Sentiment polarity  
- Tone and conversational intent  

**Output Classes:**  
- 🟢 Normal Call  
- 🔴 Scam/Fraudulent Call  

---

## ⚙️ Technologies Used  

| Component | Technology |
|------------|-------------|
| Frontend | HTML, CSS |
| Backend | Flask (Python) |
| Speech Recognition | OpenAI Whisper |
| ML Framework | Scikit-learn / TensorFlow / PyTorch |
| Dataset | Custom labeled dataset of scam vs normal calls |

---

## 🧩 Installation & Setup  

```bash
# Clone the repository
git clone https://github.com/<your-username>/AI-Powered-Scam-Call-Detection-Tool.git
cd AI-Powered-Scam-Call-Detection-Tool

# Create virtual environment
python -m venv venv
venv\Scripts\activate    # For Windows
# OR
source venv/bin/activate # For Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the Flask application
python app.py

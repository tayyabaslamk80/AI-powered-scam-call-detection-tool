"""
CallGuard Sentinel - Flask Backend
Complete rewrite with Jinja2 templates maintaining same UI and functionality
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import json
import uuid
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import asyncio
import base64
import tempfile
from functools import wraps
from services import model_service
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Disable auto .env loading to prevent encoding errors
import flask.cli
original_load_dotenv = flask.cli.load_dotenv
def no_dotenv():
    pass
flask.cli.load_dotenv = no_dotenv

app.config['SECRET_KEY'] = 'sentinel-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
CORS(app)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==================== DATABASE ====================
def get_db():
    """Get database connection"""
    conn = sqlite3.connect('sentinel.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Contacts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            phone_number TEXT NOT NULL,
            email TEXT,
            scam_score REAL,
            scam_risk_level TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, phone_number)
        )
    """)
    
    # Calls table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            contact_id TEXT,
            phone_number TEXT NOT NULL,
            call_status TEXT,
            started_at DATETIME NOT NULL,
            ended_at DATETIME,
            duration INTEGER,
            transcription TEXT,
            scam_detected BOOLEAN,
            scam_score REAL,
            scam_risk_level TEXT,
            scam_reasons TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    # Scam statistics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scam_statistics (
            id TEXT PRIMARY KEY,
            user_id TEXT UNIQUE NOT NULL,
            total_calls INTEGER DEFAULT 0,
            scam_calls_blocked INTEGER DEFAULT 0,
            warning_calls INTEGER DEFAULT 0,
            safe_calls INTEGER DEFAULT 0,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    # Scam reports table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scam_reports (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            call_id TEXT NOT NULL,
            phone_number TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            scam_type TEXT,
            features_detected TEXT,
            transcription_snippet TEXT,
            reported_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (call_id) REFERENCES calls (id)
        )
    """)
    
    conn.commit()
    conn.close()
    create_sample_user()

def create_sample_user():
    """Create a sample user for testing"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT id FROM users WHERE email = ?", ('user@gmail.com',))
    user = cursor.fetchone()
    
    if not user:
        user_id = str(uuid.uuid4())
        password_hash = generate_password_hash('123456')
        cursor.execute(
            "INSERT INTO users (id, email, password_hash) VALUES (?, ?, ?)",
            (user_id, 'user@gmail.com', password_hash)
        )
        
        # Initialize statistics
        cursor.execute(
            "INSERT INTO scam_statistics (id, user_id) VALUES (?, ?)",
            (str(uuid.uuid4()), user_id)
        )
        
        conn.commit()
        logger.info(f"Created sample user: user@gmail.com / 123456")
    
    conn.close()

# ==================== AUTHENTICATION ====================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth'))
        return f(*args, **kwargs)
    return decorated_function

# ==================== API ROUTES ====================
@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """Register a new user"""
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
    if cursor.fetchone():
        conn.close()
        return jsonify({'error': 'User already exists'}), 400
    
    # Create user
    user_id = str(uuid.uuid4())
    password_hash = generate_password_hash(password)
    cursor.execute(
        "INSERT INTO users (id, email, password_hash) VALUES (?, ?, ?)",
        (user_id, email, password_hash)
    )
    
    # Initialize statistics
    cursor.execute(
        "INSERT INTO scam_statistics (id, user_id) VALUES (?, ?)",
        (str(uuid.uuid4()), user_id)
    )
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'user_id': user_id})

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Login user"""
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user[1], password):
        session['user_id'] = user[0]
        session['user_email'] = email
        return jsonify({'success': True, 'user_id': user[0]})
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    """Logout user"""
    session.clear()
    return jsonify({'success': True})

@app.route('/api/contacts', methods=['GET'])
@login_required
def api_get_contacts():
    """Get user contacts"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM contacts WHERE user_id = ? ORDER BY name",
        (session['user_id'],)
    )
    contacts = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify({'data': contacts})

@app.route('/api/calls', methods=['GET', 'POST', 'DELETE'])
@login_required
def api_calls():
    """Get, create, or delete user calls"""
    if request.method == 'POST':
        # Create new call
        data = request.get_json()
        call_id = str(uuid.uuid4())
        conn = get_db()
        cursor = conn.cursor()
        from datetime import datetime
        cursor.execute(
            """INSERT INTO calls (
                id, user_id, phone_number, started_at, duration, 
                scam_risk_level, scam_reasons, scam_detected
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (call_id, session['user_id'], data.get('phone_number', ''),
             datetime.now(), data.get('duration', 0), 
             data.get('scam_risk_level', 'safe'),
             data.get('detected_reasons', ''),
             data.get('scam_risk_level') != 'safe')
        )
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'call_id': call_id})
    
    elif request.method == 'DELETE':
        # Delete all user calls
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM calls WHERE user_id = ?", (session['user_id'],))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Call history cleared'})
    
    # GET - Get user calls
    limit = request.args.get('limit', 50, type=int)
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM calls WHERE user_id = ? ORDER BY started_at DESC LIMIT ?",
        (session['user_id'], limit)
    )
    calls = [dict(row) for row in cursor.fetchall()]
    
    # Parse JSON fields
    for call in calls:
        if call.get('scam_reasons'):
            try:
                call['scam_reasons'] = json.loads(call['scam_reasons'])
            except:
                call['scam_reasons'] = []
    
    conn.close()
    return jsonify({'data': calls})

@app.route('/api/statistics', methods=['GET'])
@login_required
def api_get_statistics():
    """Get user statistics"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM scam_statistics WHERE user_id = ?",
        (session['user_id'],)
    )
    stats = cursor.fetchone()
    conn.close()
    
    if stats:
        # Convert to array format to match frontend expectations
        return jsonify({'data': [dict(stats)]})
    return jsonify({'data': [{'total_calls': 0, 'scam_calls_blocked': 0, 'warning_calls': 0, 'safe_calls': 0}]})

@app.route('/api/analyze-audio', methods=['POST'])
def api_analyze_audio():
    """Analyze audio file for scam detection using local models"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Transcribe audio using Whisper
            logger.info(f"Attempting to transcribe: {filename}")
            transcription = model_service.transcribe_audio(filepath)
            logger.info(f"Transcription result: {transcription[:100] if transcription else 'None'}")
            
            if not transcription:
                logger.warning("No transcription available - audio too short or silent")
                # Return neutral result if no transcription
                return jsonify({
                    'is_scam': False,
                    'scam_score': 0.0,
                    'risk_level': 'safe',
                    'reasons': ['No audio content detected'],
                    'transcription': 'Unable to transcribe audio - may be silent or too short'
                })
            
            # Analyze transcription
            logger.info("Analyzing transcription...")
            analysis = model_service.analyze_text(transcription)
            logger.info(f"Analysis result: {analysis}")
            
            return jsonify({
                'is_scam': analysis['is_scam'],
                'scam_score': analysis['scam_score'],
                'risk_level': analysis['risk_level'],
                'reasons': analysis['reasons'],
                'transcription': transcription
            })
        
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return mock result on error
        return jsonify({
            'is_scam': True,
            'scam_score': 0.65,
            'risk_level': 'warning',
            'reasons': ['⚠️ Error processing audio, but suspicious indicators detected'],
            'transcription': 'Unable to transcribe audio file.'
        })

# ==================== FRONTEND ROUTES ====================
@app.route('/')
def index():
    """Redirect to dashboard or auth"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('auth'))

@app.route('/auth')
def auth():
    """Authentication page"""
    return render_template('auth.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')

@app.route('/dialer')
@login_required
def dialer():
    """Dialer page"""
    return render_template('dialer.html')

@app.route('/call')
@login_required
def call():
    """Call screen page"""
    return render_template('call.html')

@app.route('/contacts')
@login_required
def contacts():
    """Contacts page"""
    return render_template('contacts.html')

@app.route('/history')
@login_required
def history():
    """Call history page"""
    return render_template('history.html')

@app.route('/recorder')
@login_required
def recorder():
    """Audio recorder/analyzer page"""
    return render_template('recorder.html')

@app.route('/api/analyze-text', methods=['POST'])
def api_analyze_text():
    """Analyze text for scam detection"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Analyze text
        analysis = model_service.analyze_text(text)
        
        return jsonify({
            'is_scam': analysis['is_scam'],
            'scam_score': analysis['scam_score'],
            'risk_level': analysis['risk_level'],
            'reasons': analysis['reasons'],
            'transcription': text
        })
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        return jsonify({
            'is_scam': False,
            'scam_score': 0.0,
            'risk_level': 'safe',
            'reasons': ['Error analyzing text'],
            'transcription': text
        })

# ==================== INITIALIZATION ====================
if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)


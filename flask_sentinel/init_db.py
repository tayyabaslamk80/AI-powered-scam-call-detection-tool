"""
Initialize the database for Flask Sentinel
Run this once to set up the database
"""

import sqlite3
import uuid
from werkzeug.security import generate_password_hash

def init_database():
    """Initialize database tables and create sample user"""
    conn = sqlite3.connect('sentinel.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
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
            UNIQUE(user_id, phone_number)
        )
    """)
    
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
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scam_statistics (
            id TEXT PRIMARY KEY,
            user_id TEXT UNIQUE NOT NULL,
            total_calls INTEGER DEFAULT 0,
            scam_calls_blocked INTEGER DEFAULT 0,
            warning_calls INTEGER DEFAULT 0,
            safe_calls INTEGER DEFAULT 0,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
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
            reported_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create sample user
    cursor.execute("SELECT id FROM users WHERE email = ?", ('user@gmail.com',))
    if not cursor.fetchone():
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
        
        print(f"✓ Created sample user: user@gmail.com / 123456")
    
    conn.commit()
    conn.close()
    print("✓ Database initialized successfully!")

if __name__ == '__main__':
    init_database()


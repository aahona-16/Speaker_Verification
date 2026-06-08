import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "vector_store.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS enrolled_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            embedding_json TEXT NOT NULL,
            file_count INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS file_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            filename TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Safely migrate existing databases
    try:
        cursor.execute('ALTER TABLE enrolled_users ADD COLUMN file_count INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        pass # Column already exists
        
    try:
        cursor.execute('ALTER TABLE file_history ADD COLUMN filepath TEXT DEFAULT ""')
    except sqlite3.OperationalError:
        pass # Column already exists
        
    conn.commit()
    conn.close()

def enroll_user(name: str, embedding_list: list, initial_count: int = 1):
    """Saves or updates a user's embedding in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    emb_str = json.dumps(embedding_list)
    
    # Insert or update based on name
    cursor.execute('''
        INSERT INTO enrolled_users (name, embedding_json, file_count)
        VALUES (?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET 
            embedding_json = excluded.embedding_json,
            file_count = enrolled_users.file_count + excluded.file_count
    ''', (name, emb_str, initial_count))
    
    conn.commit()
    conn.close()

def get_user_embedding(name: str) -> list:
    """Returns the embedding list for a specific user, or None if not found."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT embedding_json FROM enrolled_users WHERE name = ?', (name,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return json.loads(row[0])
    return None

def get_all_users() -> list:
    """Returns a list of all enrolled user names."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT name FROM enrolled_users ORDER BY name ASC')
    rows = cursor.fetchall()
    conn.close()
    
    return [row[0] for row in rows]

def delete_user(name: str):
    """Deletes a user from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM enrolled_users WHERE name = ?', (name,))
    conn.commit()
    conn.close()

def increment_file_count(name: str, count: int):
    """Adds to the file_count of an existing user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE enrolled_users SET file_count = file_count + ? WHERE name = ?', (count, name))
    conn.commit()
    conn.close()
    
def get_file_count(name: str) -> int:
    """Returns the total number of files assigned to a user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT file_count FROM enrolled_users WHERE name = ?', (name,))
    row = cursor.fetchone()
    conn.close()
    
    if row and row[0] is not None:
        return row[0]
    return 0

def log_file_to_user(name: str, filename: str, filepath: str = ""):
    """Logs a specific filename and its persistent filepath to a user's history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO file_history (user_name, filename, filepath) VALUES (?, ?, ?)', (name, filename, filepath))
    conn.commit()
    conn.close()

def get_historical_files(name: str) -> list:
    """Returns a list of (filename, filepath) tuples for a user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT filename, filepath FROM file_history WHERE user_name = ? ORDER BY timestamp DESC', (name,))
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        # Fallback if migration hasn't run yet
        cursor.execute('SELECT filename, "" FROM file_history WHERE user_name = ? ORDER BY timestamp DESC', (name,))
        rows = cursor.fetchall()
    conn.close()
    
    return rows

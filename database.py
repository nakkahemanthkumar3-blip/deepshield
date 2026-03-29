import sqlite3
from datetime import datetime

DB_PATH = "deepshield.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS scan_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT NOT NULL,
            file_type   TEXT NOT NULL,
            result      TEXT NOT NULL,
            confidence  REAL NOT NULL,
            scanned_at  TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_scan(filename, file_type, result, confidence):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO scan_history (filename, file_type, result, confidence, scanned_at) VALUES (?,?,?,?,?)',
        (filename, file_type, result, confidence,
         datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    )
    conn.commit()
    conn.close()

def get_all_scans():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        'SELECT id, filename, file_type, result, confidence, scanned_at FROM scan_history ORDER BY id DESC'
    ).fetchall()
    conn.close()
    return rows
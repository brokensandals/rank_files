import sqlite3
import os
from typing import Self, Optional
from datetime import datetime, timezone


class Cache:
    def __init__(self, path: str) -> None:
        self.path = path
        self.total_hits = 0
        self.db = sqlite3.connect(self.path)
        self.init_db()
    
    def __enter__(self) -> Self:
        return self
    
    def __exit__(self, type, value, traceback) -> None:
        self.close()
    
    def close(self) -> None:
        self.db.close()
    
    def init_db(self) -> None:
        cur = self.db.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS cache_entry (entry_key TEXT PRIMARY KEY, entry_value TEXT, timestamp INTEGER)")
    
    def put(self, key: str, val: str) -> None:
        cur = self.db.cursor()
        cur.execute("INSERT OR REPLACE INTO cache_entry (entry_key, entry_value, timestamp) VALUES (?, ?, ?)", (key, val, datetime.now(timezone.utc).timestamp()))
        self.db.commit()
    
    def fetch(self, key: str) -> Optional[str]:
        cur = self.db.cursor()
        results = cur.execute("SELECT entry_value FROM cache_entry WHERE entry_key = ?", (key,))
        result = results.fetchone()
        if result is None:
            return None
        self.total_hits += 1
        return result[0]


def default_cache() -> Cache:
    path = os.getenv("RANK_FILES_CACHE", "rank-files-cache.sqlite3")
    return Cache(path)

import os
import queue
import sqlite3
import threading
import time
from typing import List, Tuple

def init_db(db_path: str = "shared/metrics.db"):
    """
    Initializes the SQLite database and ensures directories and schema exist.
    """
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Create the metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            step INTEGER NOT NULL,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            node_id TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
    """)
    # Index to speed up retrieval
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_step_name ON metrics(step, name)")
    conn.commit()
    conn.close()

class AsyncMetricLogger:
    """
    Asynchronously write training metrics to SQLite to avoid I/O bottlenecks in the training loop.
    Uses a background worker thread and thread-safe queue.
    """
    def __init__(self, db_path: str = "shared/metrics.db", node_id: str = "node-0", flush_interval_secs: float = 1.0, batch_size: int = 50):
        self.db_path = db_path
        self.node_id = node_id
        self.flush_interval_secs = flush_interval_secs
        self.batch_size = batch_size
        
        # Initialize database schema
        init_db(self.db_path)
        
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def log(self, step: int, name: str, value: float):
        """
        Pushes a metric to the background queue. Non-blocking.
        """
        import math
        try:
            val_float = float(value)
            if math.isnan(val_float) or math.isinf(val_float):
                return
            self.queue.put((step, name, val_float, self.node_id, time.time()))
        except Exception:
            pass

    def _worker_loop(self):
        # Open connection in this background thread
        conn = sqlite3.connect(self.db_path)
        last_flush = time.time()
        batch: List[Tuple] = []

        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                # Wait for an item with a timeout to allow flushing and checking stop_event
                item = self.queue.get(timeout=0.1)
                batch.append(item)
                self.queue.task_done()
            except queue.Empty:
                pass

            # Flush condition: batch size reached or flush interval passed
            now = time.time()
            if len(batch) >= self.batch_size or (now - last_flush >= self.flush_interval_secs and batch):
                self._write_batch(conn, batch)
                batch.clear()
                last_flush = now

        # Final flush before exit
        if batch:
            self._write_batch(conn, batch)
        
        conn.close()

    def _write_batch(self, conn: sqlite3.Connection, batch: List[Tuple]):
        try:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT INTO metrics (step, name, value, node_id, timestamp) VALUES (?, ?, ?, ?, ?)",
                batch
            )
            conn.commit()
        except Exception as e:
            print(f"Error writing metrics to SQLite batch: {e}")

    def close(self):
        """
        Stops the worker thread and flushes remaining metrics.
        """
        self.stop_event.set()
        self.worker.join(timeout=5.0)

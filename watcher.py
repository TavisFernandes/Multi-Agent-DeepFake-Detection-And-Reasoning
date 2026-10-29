import sys
import time
import hashlib
import os
import shutil
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import tempfile

class TempFileWatcher(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.mp4'):
            self.process_file(event.src_path)
            
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.mp4'):
            self.process_file(event.src_path)
            
    def process_file(self, path):
        try:
            # Try to read the file
            size = os.path.getsize(path)
            if size == 0:
                return # Still being written
                
            # Compute hash
            hasher = hashlib.sha256()
            with open(path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            
            print(f"Captured {path}")
            print(f"Size: {size}")
            print(f"SHA-256: {hasher.hexdigest()}")
            print("--------------------------------------------------")
            sys.stdout.flush()
        except Exception as e:
            pass # File might be locked or deleted

if __name__ == "__main__":
    temp_dir = tempfile.gettempdir()
    print(f"Watching {temp_dir} for .mp4 files...")
    
    event_handler = TempFileWatcher()
    observer = Observer()
    observer.schedule(event_handler, temp_dir, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

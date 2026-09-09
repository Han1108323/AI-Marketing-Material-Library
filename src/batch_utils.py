import threading
import time
import os
import glob
import traceback
from PIL import Image

class BatchProcessor:
    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set() # Initially not paused
        
        # Shared state
        self.progress = {
            "total": 0,
            "processed": 0,
            "current_file": "",
            "status": "idle", # idle, running, paused, completed, stopped, error
            "logs": [],
            "success_count": 0,
            "fail_count": 0
        }
        
        self.pipeline = None
        self.agent = None
        self.auth_manager = None
        self.user_id = None
        self.img_dir = ""
        
    def start(self, pipeline, agent, auth_manager, user_id, img_dir):
        if self._thread is not None and self._thread.is_alive():
            # Check if it's actually finished but thread is still lingering?
            # No, is_alive() is accurate.
            return 
            
        self.pipeline = pipeline
        self.agent = agent
        self.auth_manager = auth_manager
        self.user_id = user_id
        self.img_dir = img_dir
        
        self._stop_event.clear()
        self._pause_event.set()
        
        # Reset progress
        self.progress["status"] = "running"
        self.progress["logs"] = []
        self.progress["processed"] = 0
        self.progress["total"] = 0
        self.progress["success_count"] = 0
        self.progress["fail_count"] = 0
        self.progress["current_file"] = ""
        
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
    def pause(self):
        self._pause_event.clear()
        self.progress["status"] = "paused"
        
    def resume(self):
        self._pause_event.set()
        self.progress["status"] = "running"
        
    def stop(self):
        self._stop_event.set()
        self.progress["status"] = "stopped"
        self._pause_event.set() # Unblock wait
        
    def _run_loop(self):
        try:
            if not os.path.exists(self.img_dir):
                self._log("Error: Directory not found")
                self.progress["status"] = "error"
                return

            all_files = glob.glob(os.path.join(self.img_dir, "*.*"))
            img_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            self.progress["total"] = len(img_files)
            self._log(f"Found {len(img_files)} images.")
            
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Use ThreadPool for speed (3 workers to avoid strict rate limits but speed up I/O)
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                
                # Submit all tasks
                for i, fpath in enumerate(img_files):
                    if self._stop_event.is_set():
                        break
                    
                    # Pause check (Global pause, blocks submission)
                    while not self._pause_event.is_set() and not self._stop_event.is_set():
                        self.progress["status"] = "paused"
                        time.sleep(0.5)
                    
                    if self.progress["status"] == "paused":
                         self.progress["status"] = "running"

                    if self._stop_event.is_set():
                        break

                    fname = os.path.basename(fpath)
                    future = executor.submit(self._process_single_image, fpath, fname)
                    futures[future] = fname
                    
                # Wait for completion and update progress
                for future in as_completed(futures):
                    if self._stop_event.is_set():
                         executor.shutdown(wait=False, cancel_futures=True)
                         break
                         
                    fname = futures[future]
                    self.progress["current_file"] = fname # Just to show something changing
                    
                    try:
                        res = future.result()
                        if res:
                             self.progress["success_count"] += 1
                        else:
                             self.progress["fail_count"] += 1
                    except Exception as e:
                        self._log(f"Error processing {fname}: {e}")
                        self.progress["fail_count"] += 1
                    
                    self.progress["processed"] += 1
                    
            if not self._stop_event.is_set():
                self.progress["status"] = "completed"
                self._log("Batch processing completed.")
            else:
                self.progress["status"] = "stopped"
                self._log("Batch processing stopped.")
            
        except Exception as e:
            self._log(f"Fatal error: {str(e)}")
            self.progress["status"] = "error"

    def _process_single_image(self, fpath, fname):
        try:
            # Use a fresh file handle to avoid issues with closed files
            # Image.open is lazy, so we load it to force read
            with Image.open(fpath) as img:
                img.load()
                # pipeline.process_image handles the logic. 
                # Note: We must ensure pipeline is thread-safe or has its own resources.
                # Assuming pipeline uses API calls which are generally thread-safe.
                
                res = self.pipeline.process_image(img, filename=fname, user_id=self.user_id)
            
            is_success = False
            if isinstance(res, tuple) and res[0] is not None:
                is_success = True
            elif res:
                is_success = True
                
            if is_success:
                if self.auth_manager and self.user_id:
                    # Simple lock for safety if needed, but dict updates are atomic-ish in GIL
                    self.auth_manager.increment_upload_count(self.user_id)
                return True
            return False
        except Exception as e:
            # Log full traceback to stdout for debugging
            print(f"❌ Error processing {fname}:")
            traceback.print_exc()
            raise e
            
    def _log(self, msg):
        # Keep only last 5 logs for UI
        self.progress["logs"].append(msg)
        if len(self.progress["logs"]) > 5:
            self.progress["logs"].pop(0)

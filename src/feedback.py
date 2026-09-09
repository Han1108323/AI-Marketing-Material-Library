import os
import json
import csv
from datetime import datetime
from typing import Dict, Any

class FeedbackSystem:
    """
    Handles user feedback collection and persistence.
    """
    def __init__(self, log_file: str = "user_feedback.csv"):
        self.log_file = log_file
        self._init_log_file()

    def _init_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "material_id", "action_type", "user_query", "metadata"])

    def record(self, material_id: str, action_type: str, user_query: str = "", metadata: Dict = None):
        """
        Record user feedback.
        action_type: "click", "like", "dislike", "copy", "download", "fission", "reuse"
        """
        if metadata is None:
            metadata = {}
            
        timestamp = datetime.now().isoformat()
        
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, 
                    material_id, 
                    action_type, 
                    user_query, 
                    json.dumps(metadata, ensure_ascii=False)
                ])
            print(f"📝 Feedback recorded: {action_type} for {material_id}")
            
            # TODO: Future implementation could trigger weight adjustment here
            
        except Exception as e:
            print(f"❌ Failed to record feedback: {e}")

    def get_stats(self) -> Dict[str, int]:
        """
        Get simple stats.
        """
        stats = {}
        try:
            if os.path.exists(self.log_file):
                import pandas as pd
                df = pd.read_csv(self.log_file)
                stats = df['action_type'].value_counts().to_dict()
        except:
            pass
        return stats

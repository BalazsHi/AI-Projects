import json
import os
from typing import Dict, List, Any
from datetime import datetime

class JsonUtils:
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str) -> None:
        """Save data to JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def create_checklist_item(requirement: str, reference: str, section: str) -> Dict[str, Any]:
        """Create a standardized checklist item."""
        return {
            "requirement": requirement,
            "reference": reference,
            "section": section,
            "compliance_status": None,
            "notes": "",
            "timestamp": datetime.now().isoformat()
        }

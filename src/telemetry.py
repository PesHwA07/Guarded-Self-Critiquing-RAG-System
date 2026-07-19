import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

class JsonFormatter(logging.Formatter):
    """Custom logging formatter that outputs structured JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Core standard fields
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        # Capture optional structured data passed via extra={'metadata': ...}
        if hasattr(record, 'metadata') and isinstance(record.metadata, dict):
            log_record.update(record.metadata)
            
        # Capture exception details
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
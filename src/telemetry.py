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
            
        return json.dumps(log_record)


def setup_telemetry(json_format: bool = True, level: int = logging.INFO) -> None:
    """Initialize telemetry and structured logging for the pipeline.
    
    Args:
        json_format: If True, outputs logs as JSON (best for production/telemetry).
                     If False, outputs simple text (best for local CLI interaction).
        level: The minimum logging level.
    """
    root_logger = logging.getLogger()
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    handler = logging.StreamHandler(sys.stdout)
    
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(message)s"))
        
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    
    # Optional LangSmith integration note:
    # LangChain/LangGraph will automatically trace to LangSmith if the 
    # LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY environment variables are set.
    # No additional code is required here for LangSmith tracing!

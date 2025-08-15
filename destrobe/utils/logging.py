"""
Logging utilities for destrobe.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional


class JSONLHandler(logging.Handler):
    """Custom handler that writes log records as JSON Lines."""
    
    def __init__(self, filename: Path) -> None:
        super().__init__()
        self.filename = filename
        self.file = open(filename, 'a', encoding='utf-8')
    
    def emit(self, record: logging.LogRecord) -> None:
        try:
            if hasattr(record, 'msg') and isinstance(record.msg, dict):
                # If the message is already a dict, use it directly
                log_entry = record.msg
            else:
                # Otherwise, create a structured entry
                log_entry = {
                    'timestamp': record.created,
                    'level': record.levelname,
                    'message': self.format(record),
                }
            
            json_line = json.dumps(log_entry, ensure_ascii=False)
            self.file.write(json_line + '\n')
            self.file.flush()
            
        except Exception:
            self.handleError(record)
    
    def close(self) -> None:
        if self.file:
            self.file.close()
        super().close()


def setup_logging(logfile: Optional[Path] = None) -> Optional[logging.Logger]:
    """Set up logging configuration."""
    if logfile is None:
        return None
    
    # Create logger
    logger = logging.getLogger('destrobe')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Add JSONL handler
    handler = JSONLHandler(logfile)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return logger


def log_metrics(logger: Optional[logging.Logger], metrics: Dict[str, Any]) -> None:
    """Log metrics data if logger is available."""
    if logger:
        logger.info(metrics)

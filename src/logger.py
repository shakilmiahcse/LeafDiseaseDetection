import logging
import logging.handlers
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_dir: str = "logs", level=logging.INFO) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler (detailed)
    timestamp = datetime.now().strftime("%Y%m%d")
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / f"app_{timestamp}.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (simple)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger


# Module-level logger instance
app_logger = setup_logger(__name__)

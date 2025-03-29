"""Utility functions for the LlamaIndex RAG system."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def load_dotenv(dotenv_path: Optional[Union[str, Path]] = None) -> bool:
    """Load environment variables from .env file.
    
    Args:
        dotenv_path: Path to .env file (default: look in current and parent directories)
        
    Returns:
        True if .env file was loaded, False otherwise
    """
    try:
        from dotenv import load_dotenv as _load_dotenv
        
        if dotenv_path is None:
            # Look for .env in current and parent directories
            potential_paths = [
                Path(".env"),
                Path("../.env"),
                Path.home() / ".env",
            ]
            
            for path in potential_paths:
                if path.exists():
                    dotenv_path = path
                    break
        
        if dotenv_path is not None and Path(dotenv_path).exists():
            _load_dotenv(dotenv_path=dotenv_path)
            logger.debug(f"Loaded environment variables from {dotenv_path}")
            return True
        else:
            logger.debug("No .env file found")
            return False
    except ImportError:
        logger.warning("python-dotenv not installed. Cannot load .env file.")
        return False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
) -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, log to console only)
        log_format: Log format string (if None, use default format)
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Set up default format if not provided
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # File handler if log_file is provided
    if log_file is not None:
        log_file = Path(log_file) if isinstance(log_file, str) else log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format=log_format,
    )
    
    # Also configure llamaindex logger with the same level
    logging.getLogger("llama_index").setLevel(numeric_level)


def get_available_devices() -> Dict[str, bool]:
    """Get available devices for running models.
    
    Returns:
        Dictionary mapping device names to availability
    """
    devices = {
        "cpu": True,  # CPU is always available
        "cuda": False,
        "mps": False,
    }
    
    try:
        import torch
        
        # Check CUDA availability
        if torch.cuda.is_available():
            devices["cuda"] = True
            logger.debug(f"CUDA available with {torch.cuda.device_count()} devices")
        
        # Check MPS availability (Apple Silicon)
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices["mps"] = True
            logger.debug("MPS (Apple Silicon) available")
    except ImportError:
        logger.debug("PyTorch not installed, cannot check CUDA/MPS availability")
    
    return devices


def get_optimal_device() -> str:
    """Get the optimal device for running models.
    
    Returns:
        Device name (cuda, mps, or cpu)
    """
    devices = get_available_devices()
    
    # Prefer CUDA, then MPS, then CPU
    if devices["cuda"]:
        return "cuda"
    elif devices["mps"]:
        return "mps"
    else:
        return "cpu"


def check_api_keys() -> Dict[str, bool]:
    """Check which API keys are available in the environment.
    
    Returns:
        Dictionary mapping API service names to availability
    """
    keys = {
        "openai": "OPENAI_API_KEY" in os.environ,
        "anthropic": "ANTHROPIC_API_KEY" in os.environ,
        "huggingface": "HUGGINGFACE_API_KEY" in os.environ or "HF_API_KEY" in os.environ,
        "cohere": "COHERE_API_KEY" in os.environ,
        "azure_openai": "AZURE_OPENAI_API_KEY" in os.environ,
    }
    
    return keys


def human_readable_size(size_bytes: int) -> str:
    """Convert size in bytes to a human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string (e.g., "4.2 MB")
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ("B", "KB", "MB", "GB", "TB")
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}" 
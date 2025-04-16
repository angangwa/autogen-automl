"""
Configuration settings for the AutoGen EDA application.
Loads settings from environment variables with sensible defaults.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Warn about missing API keys only if we're using a specific model provider
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "anthropic").lower()  # anthropic, openai, azure, google

if MODEL_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
    logging.warning("ANTHROPIC_API_KEY not set in environment variables.")
elif MODEL_PROVIDER == "openai" and not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY not set in environment variables.")
elif MODEL_PROVIDER == "azure" and not AZURE_OPENAI_API_KEY and not os.getenv("AZURE_AD_TOKEN"):
    logging.warning("Neither AZURE_OPENAI_API_KEY nor AZURE_AD_TOKEN set for Azure authentication.")
elif MODEL_PROVIDER == "google" and not GOOGLE_API_KEY:
    logging.warning("GOOGLE_API_KEY not set in environment variables.")

# Paths
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "data"))
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", str(BASE_DIR / "outputs"))
HISTORY_DIR = os.getenv("HISTORY_DIR", str(BASE_DIR / "history"))

# Ensure directories exist
Path(DATA_DIR).mkdir(exist_ok=True)
Path(OUTPUTS_DIR).mkdir(exist_ok=True)
Path(HISTORY_DIR).mkdir(exist_ok=True)

# Docker Settings
DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "python:3.11")
DOCKER_INIT_PACKAGES = os.getenv(
    "DOCKER_INIT_PACKAGES", 
    "pandas numpy scikit-learn matplotlib seaborn plotly kaleido"
).split()

# Model Settings
AI_MODEL = os.getenv("AI_MODEL", "claude-3-7-sonnet-20250219")

# Model Provider-specific settings
# Azure OpenAI settings
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-06-01")

# Application Settings
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
LOG_LEVEL = os.getenv("LOG_LEVEL", "ERROR")  # Default to ERROR unless set explicitly

# Console Settings (all can be overridden by environment variables)
CONSOLE_SHOW_TECHNICAL_DETAILS = os.getenv("CONSOLE_SHOW_TECHNICAL_DETAILS", "False").lower() in ("true", "1", "t")
CONSOLE_OUTPUT_STATS = os.getenv("CONSOLE_OUTPUT_STATS", "True").lower() in ("true", "1", "t")
CONSOLE_NO_INLINE_IMAGES = os.getenv("CONSOLE_NO_INLINE_IMAGES", "False").lower() in ("true", "1", "t")
CONSOLE_SILENT_LOGGING = os.getenv("CONSOLE_SILENT_LOGGING", "True").lower() in ("true", "1", "t")

# Standardized defaults for analysis behavior
INTERACTIVE = os.getenv("INTERACTIVE", "True").lower() in ("true", "1", "t")
DOCKER_WAIT_TIME = int(os.getenv("DOCKER_WAIT_TIME", 30))
MAX_TURNS = int(os.getenv("MAX_TURNS", 20))
SAVE_HISTORY = os.getenv("SAVE_HISTORY", "True").lower() in ("true", "1", "t")
CLEANUP_BEFORE_RUN = os.getenv("CLEANUP_BEFORE_RUN", "True").lower() in ("true", "1", "t")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

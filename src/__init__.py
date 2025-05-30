"""
AutoGen EDA: AI-powered Exploratory Data Analysis.
"""

import asyncio
import logging
import os
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging - This needs to happen before any other imports
# Set root logger to ERROR level
logging.basicConfig(level=logging.ERROR, force=True)

# Configure all known verbose loggers to ERROR level - add more if needed
VERBOSE_LOGGERS = [
    "autogen_core",
    "autogen_agentchat", 
    "openai",
    "anthropic",
    "httpx",
    "urllib3",
    "asyncio",
    "httpcore",
    "matplotlib",
    "PIL",
    "docker",
]

# Silence all the verbose loggers
for logger_name in VERBOSE_LOGGERS:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    # Remove any existing handlers to prevent duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

# For autogen events specifically
from autogen_core import EVENT_LOGGER_NAME
event_logger = logging.getLogger(EVENT_LOGGER_NAME)
event_logger.setLevel(logging.ERROR)
# Remove any existing handlers
for handler in event_logger.handlers[:]:
    event_logger.removeHandler(handler)

# Create our application logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep this at INFO so you see important app messages

from src.config import settings
from src.executors.docker import setup_docker_executor
from src.teams.data_exploration import DataExplorationTeam
from src.utils.helpers import format_data_files_info


def generate_run_id(prefix="automl_run_"):
    """
    Generate a unique run ID with timestamp.
    
    Args:
        prefix: Prefix for the run ID.
        
    Returns:
        str: A unique run ID.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}{timestamp}_{unique_id}"


def cleanup_directory(directory_path):
    """
    Clean up a directory by removing all its contents.
    
    Args:
        directory_path: Path to the directory to clean up.
    """
    directory = Path(directory_path)
    if directory.exists():
        for item in directory.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


async def run_analysis(
    user_intent: str,
    data_dir: Optional[str] = None,
    outputs_dir: Optional[str] = None,
    interactive: bool = settings.INTERACTIVE,
    docker_wait_time: int = settings.DOCKER_WAIT_TIME,
    max_turns: int = settings.MAX_TURNS,
    save_history: bool = settings.SAVE_HISTORY,
    cleanup_before_run: bool = settings.CLEANUP_BEFORE_RUN,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    model_provider: Optional[str] = None,
    reflect_on_tool_use: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run an exploratory data analysis based on user intent.
    
    Args:
        user_intent: The user's intent for the ML solution.
        data_dir: The directory containing the data files. If None, uses the value from settings.
        outputs_dir: The directory to store the output files. If None, uses the value from settings.
        interactive: Whether to allow interactive feedback from the user.
        docker_wait_time: The time to wait after starting the Docker container.
        max_turns: The maximum number of turns for the conversation.
        save_history: Whether to save the run history.
        cleanup_before_run: Whether to clean up the data and outputs directories before running.
        model: The model to use. If None, uses the value from settings.
        api_key: The API key to use. If None, uses the value from settings.
        model_provider: The model provider to use (anthropic, openai, azure, google).
                       If None, uses the value from settings.
        reflect_on_tool_use: Whether agents should reflect on tool use.
        
    Returns:
        Dict[str, Any]: The analysis results.
    """
    # Use default paths from settings if not provided
    data_dir = data_dir or settings.DATA_DIR
    outputs_dir = outputs_dir or settings.OUTPUTS_DIR
    
    # Clean up data and output directories if requested
    if cleanup_before_run:
        logger.info("Cleaning output directory before starting...")
        cleanup_directory(outputs_dir)

    # Set up the Docker executor
    docker_executor = await setup_docker_executor(
        data_dir=data_dir,
        outputs_dir=outputs_dir,
        wait_time=docker_wait_time
    )
    
    try:
        # Format data files information
        data_files_info = format_data_files_info(data_dir)
        
        # Set up the Data Exploration team
        team = DataExplorationTeam(
            docker_executor, 
            max_turns=max_turns,
            model=model,
            api_key=api_key,
            model_provider=model_provider,
            reflect_on_tool_use=reflect_on_tool_use
        )
        
        # Record the start time
        start_time = datetime.now()
        
        # Run the analysis
        results, team_state = await team.run_analysis(
            user_intent=user_intent,
            data_files=data_files_info,
            interactive=interactive
        )
        
        # Record the end time
        end_time = datetime.now()
        
        # Add run details to results
        results["start_time"] = start_time.isoformat()
        results["end_time"] = end_time.isoformat()
        results["duration"] = (end_time - start_time).total_seconds()
        results["user_intent"] = user_intent
        results["model_provider"] = model_provider or settings.MODEL_PROVIDER
        results["model"] = model or settings.AI_MODEL
        
        # Save run history if requested
        if save_history and results["completed"]:
            run_id = generate_run_id()
            results["run_id"] = run_id
            
            # Create run directory structure
            history_dir = Path(settings.HISTORY_DIR) / run_id
            history_data_dir = history_dir / "data"
            history_outputs_dir = history_dir / "outputs"
            
            # Create directories
            history_dir.mkdir(parents=True, exist_ok=True)
            history_data_dir.mkdir(parents=True, exist_ok=True)
            history_outputs_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy data files to history
            data_path = Path(data_dir)
            for item in data_path.glob("*"):
                if item.is_file():
                    shutil.copy2(item, history_data_dir / item.name)
            
            # Copy output files to history
            outputs_path = Path(outputs_dir)
            for item in outputs_path.glob("*"):
                if item.is_file():
                    shutil.copy2(item, history_outputs_dir / item.name)
            
            # Save run details
            run_details = {
                "id": run_id,
                "user_intent": user_intent,
                "interactive": interactive,
                "max_turns": max_turns,
                "docker_wait_time": docker_wait_time,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "completed": results["completed"],
                "stop_reason": results["stop_reason"],
                "model_provider": model_provider or settings.MODEL_PROVIDER,
                "model": model or settings.AI_MODEL,
                "team_state": team_state
            }
            
            with open(history_dir / "run_details.json", "w") as f:
                json.dump(run_details, f, indent=4)
            
            logger.info(f"Run history saved with ID: {run_id}")
        
        return results
    finally:
        # Stop the Docker container
        logger.info("Stopping Docker container...")
        await docker_executor.stop()
        logger.info("Docker container stopped")

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
from autogen_core import EVENT_LOGGER_NAME

from src.config import settings

# Configure autogen logging to reduce verbosity
logging.basicConfig(level=logging.WARNING)
autogen_logger = logging.getLogger(EVENT_LOGGER_NAME)
autogen_logger.addHandler(logging.StreamHandler())
autogen_logger.setLevel(logging.ERROR)

from src.executors.docker import setup_docker_executor
from src.teams.data_exploration import DataExplorationTeam
from src.utils.helpers import format_data_files_info


logger = logging.getLogger(__name__)


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
    interactive: bool = True,
    docker_wait_time: int = 10,
    max_turns: int = 20,
    save_history: bool = True,
    cleanup_before_run: bool = True,
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
        team = DataExplorationTeam(docker_executor, max_turns=max_turns)
        
        # Record the start time
        start_time = datetime.now()
        
        # Run the analysis
        results = await team.run_analysis(
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
            }
            
            with open(history_dir / "run_details.json", "w") as f:
                json.dump(run_details, f, indent=4)
            
            logger.info(f"Run history saved with ID: {run_id}")
        
        return results
    finally:
        # Stop the Docker container
        logger.info("Stopping Docker container...")
        cleanup_directory(outputs_dir)
        cleanup_directory(data_dir)
        await docker_executor.stop()
        logger.info("Docker container stopped")

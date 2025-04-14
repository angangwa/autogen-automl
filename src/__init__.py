"""
AutoGen EDA: AI-powered Exploratory Data Analysis.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from autogen_core import EVENT_LOGGER_NAME

from src.config import settings

# Configure autogen logging to reduce verbosity
autogen_logger = logging.getLogger(EVENT_LOGGER_NAME)
autogen_logger.addHandler(logging.StreamHandler())
autogen_logger.setLevel(logging.ERROR)

from src.executors.docker import setup_docker_executor
from src.teams.data_exploration import DataExplorationTeam
from src.utils.helpers import format_data_files_info


logger = logging.getLogger(__name__)


async def run_analysis(
    user_intent: str,
    data_dir: Optional[str] = None,
    outputs_dir: Optional[str] = None,
    interactive: bool = True,
    docker_wait_time: int = 10,
    max_turns: int = 20,
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
        
    Returns:
        Dict[str, Any]: The analysis results.
    """
    # Use default paths from settings if not provided
    data_dir = data_dir or settings.DATA_DIR
    outputs_dir = outputs_dir or settings.OUTPUTS_DIR
    
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
        
        # Run the analysis
        results = await team.run_analysis(
            user_intent=user_intent,
            data_files=data_files_info,
            interactive=interactive
        )
        
        return results
    finally:
        # Stop the Docker container
        logger.info("Stopping Docker container...")
        docker_executor.stop()
        logger.info("Docker container stopped")

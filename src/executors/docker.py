"""
Docker executor for running Python code in a containerized environment.
"""

import logging
import os
import time
from pathlib import Path

from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from src.config import settings

logger = logging.getLogger(__name__)

def create_docker_executor(data_dir: str = None, outputs_dir: str = None) -> DockerCommandLineCodeExecutor:
    """
    Create a Docker executor for running Python code.
    
    Args:
        data_dir: The path to the data directory to mount. If None, uses the value from settings.
        outputs_dir: The path to the outputs directory to mount. If None, uses the value from settings.
        
    Returns:
        DockerCommandLineCodeExecutor: The Docker executor.
    """
    # Use default paths from settings if not provided
    data_dir = data_dir or settings.DATA_DIR
    outputs_dir = outputs_dir or settings.OUTPUTS_DIR
    
    # Ensure paths exist
    Path(data_dir).mkdir(exist_ok=True, parents=True)
    Path(outputs_dir).mkdir(exist_ok=True, parents=True)
    
    # Get absolute paths
    data_dir_abs = str(Path(data_dir).absolute())
    outputs_dir_abs = str(Path(outputs_dir).absolute())
    
    logger.info(f"Creating Docker executor with data_dir={data_dir_abs}, outputs_dir={outputs_dir_abs}")
    
    # Create Docker executor
    executor = DockerCommandLineCodeExecutor(
        image=settings.DOCKER_IMAGE,
        extra_volumes={
            data_dir_abs: {"bind": "/mnt/data"},
            outputs_dir_abs: {"bind": "/mnt/outputs", "mode": "rw"}
        },
        init_command=f"pip install {' '.join(settings.DOCKER_INIT_PACKAGES)}",
    )
    
    return executor

async def setup_docker_executor(
    data_dir: str = None, 
    outputs_dir: str = None,
    wait_time: int = 10
) -> DockerCommandLineCodeExecutor:
    """
    Set up a Docker executor for running Python code.
    
    Args:
        data_dir: The path to the data directory to mount. If None, uses the value from settings.
        outputs_dir: The path to the outputs directory to mount. If None, uses the value from settings.
        wait_time: The time to wait after starting the Docker container.
        
    Returns:
        DockerCommandLineCodeExecutor: The Docker executor.
    """
    executor = create_docker_executor(data_dir, outputs_dir)
    
    try:
        # Start the Docker container
        logger.info("Starting Docker container...")
        await executor.start()
        
        # Wait for the container to initialize
        logger.info(f"Waiting {wait_time} seconds for container to initialize...")
        time.sleep(wait_time)
        
        logger.info("Docker container ready")
        return executor
    except Exception as e:
        logger.error(f"Failed to start Docker container: {e}")
        raise

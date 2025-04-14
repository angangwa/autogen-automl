"""
File-related tools for the AutoGen EDA application.
"""

import os
import logging
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

def write_text_file(relative_filename: str, file_content: str) -> str:
    """
    Write a text file in the output directory. Must provide both `relative_filename` and `file_content`.
    
    Args:
        relative_filename: The relative filename to write. E.g. "output.md" is written to "/mnt/outputs/output.md".
        file_content: The content to write to the file.
        
    Returns:
        str: Result of the write operation.
    """
    try:
        # Ensure the directory exists
        output_path = relative_filename.replace("/mnt/outputs/", "")
        
        output_path = Path(f"{settings.OUTPUTS_DIR}/{relative_filename}")
        # output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        with open(output_path, "w") as f:
            f.write(file_content)
            
        logger.info(f"Successfully wrote file {relative_filename}")
        return f"Successfully wrote file {relative_filename}"
    except Exception as e:
        logger.error(f"Failed to write file {relative_filename}: {e}")
        return f"Failed to write file {relative_filename}: {e}"

# Currently not in use
def list_files_in_directory(directory_path: str = "/mnt/data") -> list:
    """
    List all files in output directory.
    
    Args:
        directory_path: The path to the directory to list files from.
        
    Returns:
        list: A list of filenames in the directory.
    """
    try:
        return os.listdir(directory_path)
    except Exception as e:
        logger.error(f"Failed to list files in {directory_path}: {e}")
        return []

"""
Helper functions for the AutoGen EDA application.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def get_files_in_directory(directory: str) -> List[str]:
    """
    Get a list of files in a directory.
    
    Args:
        directory: The directory to get files from.
        
    Returns:
        List[str]: A list of files in the directory.
    """
    try:
        path = Path(directory)
        if not path.exists():
            logger.warning(f"Directory {directory} does not exist.")
            return []
        
        files = [f.name for f in path.iterdir() if f.is_file()]
        return files
    except Exception as e:
        logger.error(f"Failed to get files in {directory}: {e}")
        return []

def format_data_files_info(directory: str) -> str:
    """
    Format information about data files for user prompt.
    
    Args:
        directory: The directory containing the data files.
        
    Returns:
        str: Formatted information about the data files.
    """
    files = get_files_in_directory(directory)
    
    if not files:
        return "Data is in the folder"
    
    file_list = "\n".join([f"- {file}" for file in files])
    return f"Available files in the data directory:\n{file_list}"

def get_output_files(directory: str) -> Dict[str, List[str]]:
    """
    Get a dictionary of output files categorized by type.
    
    Args:
        directory: The directory to get files from.
        
    Returns:
        Dict[str, List[str]]: A dictionary of output files, categorized by type.
    """
    try:
        path = Path(directory)
        if not path.exists():
            logger.warning(f"Directory {directory} does not exist.")
            return {}
        
        files = [f for f in path.iterdir() if f.is_file()]
        
        # Categorize files by extension
        categorized = {
            "markdown": [],
            "python": [],
            "images": [],
            "html": [],
            "other": [],
        }
        
        for file in files:
            ext = file.suffix.lower()
            if ext in ['.md', '.markdown']:
                categorized["markdown"].append(file.name)
            elif ext == '.py':
                categorized["python"].append(file.name)
            elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
                categorized["images"].append(file.name)
            elif ext == '.html':
                categorized["html"].append(file.name)
            else:
                categorized["other"].append(file.name)
        
        return categorized
    except Exception as e:
        logger.error(f"Failed to get output files in {directory}: {e}")
        return {}

def read_file_content(file_path: str) -> Optional[str]:
    """
    Read the content of a file.
    
    Args:
        file_path: The path to the file.
        
    Returns:
        Optional[str]: The content of the file, or None if an error occurred.
    """
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return None

"""
Helper functions for the AutoGen EDA application.
"""

import os
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

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

def get_run_history() -> List[Dict[str, Any]]:
    """
    Get a list of all previous runs.
    
    Returns:
        List[Dict[str, Any]]: A list of run details, sorted by date (newest first).
    """
    from src.config import settings
    
    try:
        history_dir = Path(settings.HISTORY_DIR)
        if not history_dir.exists():
            return []
        
        runs = []
        
        # Get all directories in history_dir that start with "automl_run_"
        for run_dir in history_dir.glob("automl_run_*"):
            if not run_dir.is_dir():
                continue
            
            details_file = run_dir / "run_details.json"
            if not details_file.exists():
                continue
            
            try:
                with open(details_file, 'r') as f:
                    run_details = json.load(f)
                
                # Add some extra information
                run_details['path'] = str(run_dir)
                run_details['date_formatted'] = format_datetime(run_details.get('start_time'))
                
                runs.append(run_details)
            except Exception as e:
                logger.error(f"Failed to read run details from {details_file}: {e}")
        
        # Sort by start_time (newest first)
        runs.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        return runs
    except Exception as e:
        logger.error(f"Failed to get run history: {e}")
        return []

def get_run_details(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Get details for a specific run.
    
    Args:
        run_id: The ID of the run.
        
    Returns:
        Optional[Dict[str, Any]]: The run details, or None if not found.
    """
    from src.config import settings
    
    try:
        run_dir = Path(settings.HISTORY_DIR) / run_id
        details_file = run_dir / "run_details.json"
        
        if not details_file.exists():
            return None
        
        with open(details_file, 'r') as f:
            run_details = json.load(f)
        
        # Add path information
        run_details['path'] = str(run_dir)
        run_details['outputs_path'] = str(run_dir / "outputs")
        run_details['data_path'] = str(run_dir / "data")
        run_details['date_formatted'] = format_datetime(run_details.get('start_time'))
        
        return run_details
    except Exception as e:
        logger.error(f"Failed to get run details for {run_id}: {e}")
        return None

def load_run_to_current(run_id: str) -> bool:
    """
    Load data and outputs from a previous run to the current workspace.
    
    Args:
        run_id: The ID of the run to load.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    from src.config import settings
    import shutil
    
    try:
        # Get paths
        run_details = get_run_details(run_id)
        if not run_details:
            logger.error(f"Run {run_id} not found")
            return False
        
        run_dir = Path(run_details['path'])
        data_dir = Path(settings.DATA_DIR)
        outputs_dir = Path(settings.OUTPUTS_DIR)
        
        run_data_dir = run_dir / "data"
        run_outputs_dir = run_dir / "outputs"
        
        # Clean up current directories
        for item in data_dir.glob("*"):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
                
        for item in outputs_dir.glob("*"):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        # Copy files from the run to current directories
        if run_data_dir.exists():
            for item in run_data_dir.glob("*"):
                if item.is_file():
                    shutil.copy2(item, data_dir / item.name)
        
        if run_outputs_dir.exists():
            for item in run_outputs_dir.glob("*"):
                if item.is_file():
                    shutil.copy2(item, outputs_dir / item.name)
        
        return True
    except Exception as e:
        logger.error(f"Failed to load run {run_id}: {e}")
        return False

def format_datetime(iso_date: Optional[str]) -> str:
    """
    Format ISO datetime string to a more readable format.
    
    Args:
        iso_date: ISO format datetime string.
        
    Returns:
        str: Formatted datetime string.
    """
    if not iso_date:
        return "Unknown date"
    
    try:
        dt = datetime.fromisoformat(iso_date)
        return dt.strftime("%b %d, %Y at %I:%M %p")
    except:
        return iso_date

def get_available_examples() -> List[Dict[str, Any]]:
    """
    Get a list of available examples from the examples directory.
    
    Returns:
        List[Dict[str, Any]]: A list of example configurations.
    """
    from src.config import settings
    
    try:
        base_dir = Path(settings.BASE_DIR)
        examples_file = base_dir / "examples" / "examples.json"
        
        if not examples_file.exists():
            logger.warning(f"Examples file not found: {examples_file}")
            return []
        
        with open(examples_file, 'r') as f:
            examples = json.load(f)
            
        # Process relative paths in examples
        for example in examples:
            if "data" in example and example["data"]:
                # Convert relative path to absolute path
                data_path = example["data"]
                if not Path(data_path).is_absolute():
                    example["data_full_path"] = str(base_dir / "examples" / data_path)
                else:
                    example["data_full_path"] = data_path
        
        return examples
    except Exception as e:
        logger.error(f"Failed to get examples: {e}")
        return []

def load_example_to_current(example: Dict[str, Any]) -> bool:
    """
    Load an example dataset to the current data directory.
    
    Args:
        example: The example configuration to load.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    from src.config import settings
    
    try:
        data_dir = Path(settings.DATA_DIR)
        
        # Clean up current data directory
        for item in data_dir.glob("*"):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        # Get the source path
        source_path = Path(example.get("data_full_path", ""))
        if not source_path.exists():
            logger.error(f"Example data path not found: {source_path}")
            return False
        
        # Copy data from example to data directory
        if source_path.is_file():
            # If source is a file, copy it directly
            shutil.copy2(source_path, data_dir / source_path.name)
        elif source_path.is_dir():
            # If source is a directory, copy all its contents
            for item in source_path.glob("*"):
                if item.is_file():
                    shutil.copy2(item, data_dir / item.name)
                elif item.is_dir():
                    shutil.copytree(item, data_dir / item.name)
        
        logger.info(f"Loaded example data from {source_path} to {data_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to load example: {e}")
        return False

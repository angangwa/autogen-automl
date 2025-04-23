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

def read_text_file(relative_filename: str, directory: str = "outputs", max_lines: int = 1000) -> str:
    """
    Read a text file from the outputs or data directory, returning up to max_lines lines.
    Args:
        relative_filename: The relative filename to read.
        directory: Either 'outputs' or 'data'.
        max_lines: Maximum number of lines to read.
    Returns:
        str: The file content (up to max_lines lines) or error message.
    """
    try:
        if directory not in ("outputs", "data"):
            return "Invalid directory. Only 'outputs' or 'data' allowed."
        base_dir = settings.OUTPUTS_DIR if directory == "outputs" else settings.DATA_DIR
        file_path = Path(base_dir) / relative_filename
        if not file_path.exists():
            return f"File not found: {file_path}"
        lines = []
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line)
        return "".join(lines)
    except Exception as e:
        logger.error(f"Failed to read file {relative_filename}: {e}")
        return f"Failed to read file {relative_filename}: {e}"

def append_text_file(relative_filename: str, file_content: str, directory: str = "outputs") -> str:
    """
    Append text to a file in the outputs or data directory.
    Args:
        relative_filename: The relative filename to append to.
        file_content: The content to append.
        directory: Either 'outputs' or 'data'.
    Returns:
        str: Result of the append operation.
    """
    try:
        if directory not in ("outputs", "data"):
            return "Invalid directory. Only 'outputs' or 'data' allowed."
        base_dir = settings.OUTPUTS_DIR if directory == "outputs" else settings.DATA_DIR
        file_path = Path(base_dir) / relative_filename
        with open(file_path, "a") as f:
            f.write(file_content)
        logger.info(f"Successfully appended to file {relative_filename}")
        return f"Successfully appended to file {relative_filename}"
    except Exception as e:
        logger.error(f"Failed to append to file {relative_filename}: {e}")
        return f"Failed to append to file {relative_filename}: {e}"

def list_files_in_directory(directory: str = "outputs", recursive: bool = False) -> list:
    """
    List all files in the outputs or data directory, optionally recursively.
    Args:
        directory: Either 'outputs' or 'data'.
        recursive: Whether to list files recursively.
    Returns:
        list: A list of file paths (relative to the base directory).
    """
    try:
        if directory not in ("outputs", "data"):
            return "Invalid directory. Only 'outputs' or 'data' allowed."
        base_dir = settings.OUTPUTS_DIR if directory == "outputs" else settings.DATA_DIR
        base_path = Path(base_dir)
        if not recursive:
            return [str(p.name) for p in base_path.iterdir() if p.is_file()]
        else:
            return [str(p.relative_to(base_path)) for p in base_path.rglob("*") if p.is_file()]
    except Exception as e:
        logger.error(f"Failed to list files in {directory}: {e}")
        return []

def search_files(pattern: str, directory: str = "outputs") -> list:
    """
    Search for files matching a glob pattern in outputs or data directory.
    Args:
        pattern: Glob pattern (e.g., '*.md', 'subdir/*.json').
        directory: Either 'outputs' or 'data'.
    Returns:
        list: List of matching file paths (relative to base directory).
    """
    try:
        if directory not in ("outputs", "data"):
            return "Invalid directory. Only 'outputs' or 'data' allowed."
        base_dir = settings.OUTPUTS_DIR if directory == "outputs" else settings.DATA_DIR
        base_path = Path(base_dir)
        return [str(p.relative_to(base_path)) for p in base_path.rglob(pattern) if p.is_file()]
    except Exception as e:
        logger.error(f"Failed to search files in {directory} with pattern {pattern}: {e}")
        return []

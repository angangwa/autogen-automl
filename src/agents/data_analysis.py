"""
Data Analysis Agent for the AutoGen EDA application.
"""

import logging
from typing import Any, List, Optional

from autogen_core.tools import FunctionTool
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

from src.agents.base import BaseAgent
from src.tools.file_tools import (
    write_text_file,
    read_text_file,
    append_text_file,
    list_files_in_directory,
    search_files
)
from src.tools.image_tools import analyze_image
from src.prompts.data_analysis import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class DataAnalysisAgent(BaseAgent):
    """
    Data Analysis Agent for exploratory data analysis.
    """
    
    def __init__(
        self,
        docker_executor: DockerCommandLineCodeExecutor,
        name: str = "data_analysis_agent",
        system_message: str = SYSTEM_PROMPT,
        model: str = None,
        api_key: str = None,
        model_provider: str = None,
        reflect_on_tool_use: bool = False, # False for most models except some OpenAI models
    ):
        """
        Initialize the Data Analysis Agent.
        
        Args:
            docker_executor: The Docker executor for running Python code.
            name: The name of the agent.
            system_message: The system message for the agent.
            model: The model to use. If None, uses the value from settings.
            api_key: The API key to use. If None, uses the value from settings.
            model_provider: The model provider to use. If None, uses the value from settings.
            reflect_on_tool_use: Whether the agent should reflect on tool use.
        """
        self.docker_executor = docker_executor
        
        # Prepare tools
        tools = self.prepare_tools()
        
        # Initialize the base agent
        super().__init__(
            name=name,
            system_message=system_message,
            tools=tools,
            model=model,
            api_key=api_key,
            model_provider=model_provider,
            reflect_on_tool_use=reflect_on_tool_use
        )
        
        logger.info(f"Initialized {self.name} with {len(tools)} tools")
    
    def prepare_tools(self) -> List[Any]:
        """
        Prepare the tools for the Data Analysis Agent.
        
        Returns:
            List[Any]: The tools for the agent.
        """
        # Create the Python code execution tool
        coding_tool = PythonCodeExecutionTool(self.docker_executor)
        
        # File tools
        write_text_file_tool = FunctionTool(
            write_text_file, 
            description="Write a text file in the output directory."
        )
        read_text_file_tool = FunctionTool(
            read_text_file,
            description="Read up to 1000 lines from a text file in the outputs or data directory."
        )
        append_text_file_tool = FunctionTool(
            append_text_file,
            description="Append text to a file in the outputs or data directory."
        )
        list_files_tool = FunctionTool(
            list_files_in_directory,
            description="List files in the outputs or data directory. Supports recursive listing."
        )
        search_files_tool = FunctionTool(
            search_files,
            description="Search for files matching a glob pattern in outputs or data directory."
        )
        
        # Create the image analysis tool
        analyze_image_tool = FunctionTool(
            analyze_image, 
            description="Analyze an image using Claude's vision capabilities."
        )
        
        return [
            coding_tool,
            write_text_file_tool,
            read_text_file_tool,
            append_text_file_tool,
            list_files_tool,
            search_files_tool,
            analyze_image_tool
        ]

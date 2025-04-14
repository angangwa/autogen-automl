"""
Data Analysis Agent for the AutoGen EDA application.
"""

import logging
from typing import Any, List, Optional

from autogen_core.tools import FunctionTool
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

from src.agents.base import BaseAgent
from src.tools.file_tools import write_text_file
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
    ):
        """
        Initialize the Data Analysis Agent.
        
        Args:
            docker_executor: The Docker executor for running Python code.
            name: The name of the agent.
            system_message: The system message for the agent.
            model: The model to use. If None, uses the value from settings.
            api_key: The API key to use. If None, uses the value from settings.
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
            reflect_on_tool_use=False  # False for Anthropic models
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
        
        # Create the file writing tool
        write_text_file_tool = FunctionTool(
            write_text_file, 
            description="Write a text file in the output directory."
        )
        
        # Create the image analysis tool
        analyze_image_tool = FunctionTool(
            analyze_image, 
            description="Analyze an image using Claude's vision capabilities."
        )
        
        return [
            coding_tool,
            write_text_file_tool,
            analyze_image_tool
        ]

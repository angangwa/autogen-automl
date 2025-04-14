"""
Base agent class for the AutoGen EDA application.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_core.tools import FunctionTool

from src.config import settings

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Base class for all agents in the AutoGen EDA application.
    """
    
    def __init__(
        self,
        name: str,
        system_message: str,
        tools: List[Any],
        model: str = None,
        api_key: str = None,
        reflect_on_tool_use: bool = False,
    ):
        """
        Initialize the base agent.
        
        Args:
            name: The name of the agent.
            system_message: The system message for the agent.
            tools: The tools available to the agent.
            model: The model to use. If None, uses the value from settings.
            api_key: The API key to use. If None, uses the value from settings.
            reflect_on_tool_use: Whether the agent should reflect on tool use.
        """
        self.name = name
        self.system_message = system_message
        self.tools = tools
        self.model = model or settings.AI_MODEL
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.reflect_on_tool_use = reflect_on_tool_use
        
        # Create the model client
        self.model_client = self._create_model_client()
        
        # Create the agent
        self.agent = self._create_agent()
        
        logger.info(f"Initialized {self.name} agent with model {self.model}")
    
    def _create_model_client(self) -> AnthropicChatCompletionClient:
        """
        Create the model client.
        
        Returns:
            AnthropicChatCompletionClient: The model client.
        """
        return AnthropicChatCompletionClient(
            model=self.model,
            api_key=self.api_key
        )
    
    def _create_agent(self) -> AssistantAgent:
        """
        Create the agent.
        
        Returns:
            AssistantAgent: The agent.
        """
        return AssistantAgent(
            name=self.name,
            system_message=self.system_message,
            reflect_on_tool_use=self.reflect_on_tool_use,
            model_client=self.model_client,
            tools=self.tools
        )
    
    @abstractmethod
    def prepare_tools(self) -> List[Any]:
        """
        Prepare the tools for the agent.
        
        Returns:
            List[Any]: The tools for the agent.
        """
        pass

"""
Base agent class for the AutoGen EDA application.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, Union

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_core.tools import FunctionTool
from autogen_core.models import ChatCompletionClient
from autogen_core.models import ModelInfo

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
        model_provider: str = None,
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
            model_provider: The model provider to use (anthropic, openai, azure, google).
                           If None, uses the value from settings.
            reflect_on_tool_use: Whether the agent should reflect on tool use.
        """
        self.name = name
        self.system_message = system_message
        self.tools = tools
        self.model = model or settings.AI_MODEL
        self.api_key = api_key
        self.model_provider = model_provider or settings.MODEL_PROVIDER
        self.reflect_on_tool_use = reflect_on_tool_use
        
        # Create the model client
        self.model_client = self._create_model_client()
        
        # Create the agent
        self.agent = self._create_agent()
        
        logger.info(f"Initialized {self.name} agent with {self.model_provider} model {self.model}")
    
    def _create_model_client(self) -> ChatCompletionClient:
        """
        Create the appropriate model client based on the model provider.
        
        Returns:
            ChatCompletionClient: The model client for the specified provider.
        """
        if self.model_provider == "anthropic":
            api_key = self.api_key or settings.ANTHROPIC_API_KEY
            return AnthropicChatCompletionClient(
                model=self.model,
                api_key=api_key
            )
        elif self.model_provider == "openai":
            api_key = self.api_key or settings.OPENAI_API_KEY
            try:
                return OpenAIChatCompletionClient(
                    model=self.model,
                    api_key=api_key
                )
            except ValueError as e:
                logger.warning(f"Error creating OpenAI client: {e}. Trying to explicitly set model_info.")
                return OpenAIChatCompletionClient(
                    model=self.model,
                    api_key=api_key,
                    model_info=ModelInfo(
                        vision=True,
                        function_calling=True,
                        json_output=True,
                        family="unknown",
                        structured_output=True
                    )
                )
        elif self.model_provider == "azure":
            # Check if using token-based auth or API key
            if settings.AZURE_OPENAI_API_KEY:
                # API key based authentication
                return AzureOpenAIChatCompletionClient(
                    model=self.model,
                    azure_deployment=settings.AZURE_DEPLOYMENT,
                    azure_endpoint=settings.AZURE_ENDPOINT,
                    api_version=settings.AZURE_API_VERSION,
                    api_key=self.api_key or settings.AZURE_OPENAI_API_KEY
                )
            else:
                # Token-based authentication - requires DefaultAzureCredential
                from azure.identity import DefaultAzureCredential
                token_provider = AzureTokenProvider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
                return AzureOpenAIChatCompletionClient(
                    model=self.model,
                    azure_deployment=settings.AZURE_DEPLOYMENT,
                    azure_endpoint=settings.AZURE_ENDPOINT,
                    api_version=settings.AZURE_API_VERSION,
                    azure_ad_token_provider=token_provider
                )
        elif self.model_provider == "google":
            # Gemini uses OpenAI-compatible API, so we use the OpenAIChatCompletionClient
            # with the appropriate Gemini model name and Google's base URL
            api_key = self.api_key or settings.GOOGLE_API_KEY
                
            # Google's OpenAI-compatible API endpoint
            gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            
            # Check if this is a newer Gemini model that needs explicit capabilities
            return OpenAIChatCompletionClient(
                model=self.model,
                api_key=api_key,
                model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True),
                base_url=gemini_base_url
            )
        else:
            logger.error(f"Unsupported model provider: {self.model_provider}")
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
    
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

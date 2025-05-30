"""
Data Exploration team setup for the AutoGen EDA application.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from autogen_agentchat.teams import RoundRobinGroupChat, Swarm
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination, HandoffTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

from src.agents.data_analysis import DataAnalysisAgent
from src.agents.ideation import IdeationAgent

from src.prompts.data_analysis import format_user_prompt
from src.utils.console import CustomConsole
from src.config import settings

logger = logging.getLogger(__name__)

class DataExplorationTeam:
    """
    Team for data exploration tasks.
    """
    
    def __init__(
        self,
        docker_executor: DockerCommandLineCodeExecutor,
        max_turns: int = 20,
        model: str = None,
        api_key: str = None,
        model_provider: str = None,
        reflect_on_tool_use: bool = None,
    ):
        """
        Initialize the Data Exploration team.
        
        Args:
            docker_executor: The Docker executor for running Python code.
            max_turns: The maximum number of turns for the conversation.
            model: The model to use. If None, uses the value from settings.
            api_key: The API key to use. If None, uses the value from settings.
            model_provider: The model provider to use. If None, uses the value from settings.
            reflect_on_tool_use: Whether agents should reflect on tool use.
        """
        self.docker_executor = docker_executor
        self.max_turns = max_turns
        self.model = model or settings.AI_MODEL
        self.api_key = api_key
        self.model_provider = model_provider or settings.MODEL_PROVIDER
        
        # Set default reflect_on_tool_use based on model provider if not specified
        if reflect_on_tool_use is None:
            # Default to False for most models except some OpenAI models
            if self.model_provider == "openai" and "gpt-4" in (self.model or "").lower():
                reflect_on_tool_use = True
            else:
                reflect_on_tool_use = False
        
        self.reflect_on_tool_use = reflect_on_tool_use
        
        # Create the Data Analysis Agent
        self.data_analysis_agent = DataAnalysisAgent(
            docker_executor, 
            model=self.model,
            api_key=self.api_key,
            model_provider=self.model_provider,
            reflect_on_tool_use=self.reflect_on_tool_use,
            handoffs=["ideation_agent", "user"],
        )
        
        self.ideation_agent = IdeationAgent(
            docker_executor, 
            model=self.model,
            api_key=self.api_key,
            model_provider=self.model_provider,
            reflect_on_tool_use=self.reflect_on_tool_use,
            handoffs=["data_analysis_agent", "user"],
        )
        
        # Create the team
        self.team = self._create_team()
        
        logger.info(f"Initialized Data Exploration team with {self.model_provider} model {self.model}, max_turns={max_turns}")
    
    def _create_team(self) -> RoundRobinGroupChat:
        """
        Create the team for data exploration.
        
        Returns:
            RoundRobinGroupChat: The team.
        """
        # Set up termination conditions
        # termination_conditions = (
        #     TextMentionTermination("USER QUESTION") | 
        #     TextMentionTermination("REPORT COMPLETE")
        # )
        
        # Create the team
        # team = RoundRobinGroupChat(
        #     participants=[self.data_analysis_agent.agent],
        #     max_turns=self.max_turns,
        #     termination_condition=termination_conditions,
        # )
        

        termination = HandoffTermination(target="user") | TextMentionTermination("REPORT COMPLETE")
        team = Swarm([self.data_analysis_agent.agent, self.ideation_agent.agent], termination_condition=termination, max_turns=self.max_turns)
        
        return team
    
    async def run_analysis(
        self,
        user_intent: str,
        data_files: str = "Data is in the folder",
        interactive: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the data exploration analysis.
        
        Args:
            user_intent: The user's intent for the ML solution.
            data_files: Information about the available data files.
            interactive: Whether to allow interactive feedback from the user.
            
        Returns:
            Dict[str, Any]: The analysis results, including stop reason and conversation stats.
        """
        # Format the task prompt
        task = format_user_prompt(user_intent, data_files)
        
        # Results to return
        results = {
            "completed": False,
            "stop_reason": None,
        }
        
        # Run the conversation loop
        while True:
            # Run the conversation and stream to the console
            logger.info("Running conversation iteration...")
            stream = self.team.run_stream(task=task)
            
            # Display the stream in the console and get the response
            response = await CustomConsole(
                stream
            )
            
            last_message = response.messages[-1]

            while isinstance(last_message, HandoffMessage) and last_message.target == "user":
                user_message = input("User: ")

                task_result = await Console(
                    self.team.run_stream(task=HandoffMessage(source="user", target=last_message.source, content=user_message))
                )
                last_message = task_result.messages[-1]

                        
            # Check the stop reason
            if "REPORT COMPLETE" in response.stop_reason:
                logger.info("Analysis complete.")
                results["completed"] = True
                results["stop_reason"] = "REPORT COMPLETE"
                break
            elif "Maximum number of messages" in response.stop_reason:
                logger.warning("Maximum number of messages reached.")
                # task = "Please wrap up your analysis quickly and provide the final results."
                msg = "Please wrap up your analysis quickly and provide the final results."
                task = HandoffMessage(source="user", target=last_message.source, content=msg)
                results["stop_reason"] = "MAX_MESSAGES"
            elif isinstance(last_message, HandoffMessage) and last_message.target == "user":
                results["stop_reason"] = "USER QUESTION"
                if interactive:
                    # Get user feedback
                    user_feedback = input("User feedback requested (type 'exit' to leave): ")
                    
                    if user_feedback.lower().strip() == "exit":
                        logger.info("User requested to exit.")
                        break
                    
                    # Continue with user feedback
                    task = HandoffMessage(source="user", target=last_message.source, content=user_feedback)
                else:
                    # Non-interactive mode, just provide a generic response
                    msg = "Please continue with the analysis based on the available information. Complete your analysis if you can't proceed."
                    HandoffMessage(source="user", target=last_message.source, content=msg)
            else:
                logger.warning(f"Unexpected stop reason: {response.stop_reason}")
                task = "Please wrap up your analysis quickly and provide the final results. Ask the user if you need help.\n"
                results["stop_reason"] = "UNKNOWN"
        
        # Added saving of team state
        logger.debug("Extracting team state...")
        team_state = await self.team.save_state()

        return results, team_state

    def save_team(self, file_name: str) -> None:
        """
        Save the team state to a file.
        """
        logger.info("Saving team state...")
        component = self.team.dump_component()
        component_json = component.model_dump_json()
        with open(file_name, "w") as f:
            f.write(component_json)
        logger.info(f"Team component saved to {file_name}")
        
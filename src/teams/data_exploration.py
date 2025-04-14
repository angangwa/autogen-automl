"""
Data Exploration team setup for the AutoGen EDA application.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

from src.agents.data_analysis import DataAnalysisAgent
from src.prompts.data_analysis import format_user_prompt

logger = logging.getLogger(__name__)

class DataExplorationTeam:
    """
    Team for data exploration tasks.
    """
    
    def __init__(
        self,
        docker_executor: DockerCommandLineCodeExecutor,
        max_turns: int = 20,
    ):
        """
        Initialize the Data Exploration team.
        
        Args:
            docker_executor: The Docker executor for running Python code.
            max_turns: The maximum number of turns for the conversation.
        """
        self.docker_executor = docker_executor
        self.max_turns = max_turns
        
        # Create the Data Analysis Agent
        self.data_analysis_agent = DataAnalysisAgent(docker_executor)
        
        # Create the team
        self.team = self._create_team()
        
        logger.info(f"Initialized Data Exploration team with max_turns={max_turns}")
    
    def _create_team(self) -> RoundRobinGroupChat:
        """
        Create the team for data exploration.
        
        Returns:
            RoundRobinGroupChat: The team.
        """
        # Set up termination conditions
        termination_conditions = (
            TextMentionTermination("USER QUESTION") | 
            TextMentionTermination("ANALYSIS COMPLETE")
        )
        
        # Create the team
        team = RoundRobinGroupChat(
            participants=[self.data_analysis_agent.agent],
            max_turns=self.max_turns,
            termination_condition=termination_conditions,
        )
        
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
            response = await Console(stream, output_stats=True)
                        
            # Check the stop reason
            if "ANALYSIS COMPLETE" in response.stop_reason:
                logger.info("Analysis complete.")
                results["completed"] = True
                results["stop_reason"] = "ANALYSIS COMPLETE"
                break
            elif "Maximum number of messages" in response.stop_reason:
                logger.warning("Maximum number of messages reached.")
                task = "Please wrap up your analysis quickly and provide the final results."
                results["stop_reason"] = "MAX_MESSAGES"
            elif "USER QUESTION" in response.stop_reason:
                results["stop_reason"] = "USER QUESTION"
                if interactive:
                    # Get user feedback
                    user_feedback = input("User feedback requested (type 'exit' to leave): ")
                    
                    if user_feedback.lower().strip() == "exit":
                        logger.info("User requested to exit.")
                        break
                    
                    # Continue with user feedback
                    task = user_feedback
                else:
                    # Non-interactive mode, just provide a generic response
                    task = "Please continue with the analysis based on the available information. Complete your analysis if you can't proceed."
            else:
                logger.warning(f"Unexpected stop reason: {response.stop_reason}")
                task = "Please wrap up your analysis quickly and provide the final results. Ask the user if you need help.\n"
                results["stop_reason"] = "UNKNOWN"
        
        return results

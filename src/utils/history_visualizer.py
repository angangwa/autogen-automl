"""
History visualizer for viewing past runs from the history directory.

This module provides functionality to visualize the agent conversation from
a previous run, similar to the CustomConsole for live runs. It parses the
team_state data from run_details.json files and formats the conversation
in either rich terminal output or HTML for Streamlit.

Usage:
    # Terminal visualization
    from src.utils.history_visualizer import visualize_run
    visualize_run("automl_run_20250422_212715_4bd7549d")
    
    # HTML visualization (for Streamlit)
    from src.utils.history_visualizer import get_run_conversation_html
    html = get_run_conversation_html("automl_run_20250422_212715_4bd7549d")
    
    # Get available agents for a run
    from src.utils.history_visualizer import get_run_agent_list
    agents = get_run_agent_list("automl_run_20250422_212715_4bd7549d")
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import re

from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown
from rich import box
from rich.syntax import Syntax

from src.config import settings

# Configure logger
logger = logging.getLogger(__name__)

# Create a rich console for output
rich_console = RichConsole(log_time=False, log_path=False)

class RunHistoryVisualizer:
    """
    Visualizes the conversation history from a previous run.
    """
    
    def __init__(self, run_id: str):
        """
        Initialize the run history visualizer.
        
        Args:
            run_id: The ID of the run to visualize.
        """
        self.run_id = run_id
        self.run_path = Path(settings.HISTORY_DIR) / run_id
        self.run_details = self._load_run_details()
        self.team_state = self.run_details.get("team_state", {})
        
    def _load_run_details(self) -> Dict[str, Any]:
        """
        Load the run details from the run_details.json file.
        
        Returns:
            Dict[str, Any]: The run details.
        """
        run_details_path = self.run_path / "run_details.json"
        if not run_details_path.exists():
            logger.error(f"Run details file not found: {run_details_path}")
            return {}
        
        try:
            with open(run_details_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse run details JSON: {run_details_path}")
            return {}
    
    def get_run_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the run.
        
        Returns:
            Dict[str, Any]: A summary of the run.
        """
        if not self.run_details:
            return {
                "id": self.run_id,
                "status": "Unknown",
                "error": "Run details not found"
            }
        
        # Parse timestamps
        start_time = datetime.fromisoformat(self.run_details.get("start_time", ""))
        end_time = datetime.fromisoformat(self.run_details.get("end_time", ""))
        
        return {
            "id": self.run_id,
            "user_intent": self.run_details.get("user_intent", "Unknown"),
            "start_time": start_time,
            "end_time": end_time,
            "duration": self.run_details.get("duration", 0),
            "completed": self.run_details.get("completed", False),
            "stop_reason": self.run_details.get("stop_reason", "Unknown"),
            "model_provider": self.run_details.get("model_provider", "Unknown"),
            "model": self.run_details.get("model", "Unknown"),
            "interactive": self.run_details.get("interactive", False),
            "max_turns": self.run_details.get("max_turns", 0),
        }
    
    def _extract_agent_conversations(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract all conversations from the team state grouped by agent.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Conversations grouped by agent.
        """
        conversations = {}
        
        # Check if we have a SwarmGroupChatManager state
        if "SwarmGroupChatManager" in self.team_state.get("agent_states", {}):
            swarm_state = self.team_state["agent_states"]["SwarmGroupChatManager"]
            if "message_thread" in swarm_state:
                conversations["swarm"] = swarm_state["message_thread"]
        
        # Extract conversations from individual agents
        for agent_name, agent_state in self.team_state.get("agent_states", {}).items():
            if agent_name == "SwarmGroupChatManager":
                continue
                
            if "agent_state" in agent_state and "llm_context" in agent_state["agent_state"]:
                messages = agent_state["agent_state"]["llm_context"].get("messages", [])
                conversations[agent_name] = messages
        
        return conversations
    
    def _format_text_message(self, message: Dict[str, Any], show_metadata: bool = False) -> Panel:
        """Format a text message for rich display."""
        source = message.get("source", "Unknown")
        content = message.get("content", "")
        
        title = Text(f"{source}", style="bold green")
        
        # Handle different content types
        if isinstance(content, str):
            # Plain text content
            content_text = Text(content)
        elif isinstance(content, list):
            # Tool call or other complex content
            content_text = Text("Tool call or complex content - see detailed view")
        else:
            content_text = Text(str(content))
        
        # Add metadata if requested
        if show_metadata and "models_usage" in message:
            usage = message["models_usage"]
            if usage:
                metadata = f"\n\nTokens: {usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)}"
                content_text.append(metadata, style="dim")
        
        return Panel(content_text, title=title, border_style="green", box=box.ROUNDED)
    
    def _format_tool_call(self, message: Dict[str, Any]) -> Panel:
        """Format a tool call message for rich display."""
        source = message.get("source", "Unknown")
        content = message.get("content", [])
        
        title = Text(f"{source} 🔧", style="bold yellow")
        
        # Format the tool call content
        content_text = Text("")
        
        if isinstance(content, list):
            for item in content:
                tool_name = item.get("name", "unknown_tool")
                content_text.append(f"{tool_name}:\n", style="bold")
                
                # Format arguments if available
                if "arguments" in item:
                    try:
                        # Try to parse and format JSON arguments
                        if isinstance(item["arguments"], str):
                            # Some tools store JSON as a string
                            if item["arguments"].startswith("{"):
                                args = json.loads(item["arguments"])
                                content_text.append(json.dumps(args, indent=2))
                            else:
                                content_text.append(item["arguments"])
                        else:
                            # Otherwise it might be a dict already
                            content_text.append(json.dumps(item["arguments"], indent=2))
                    except (json.JSONDecodeError, TypeError):
                        # Fallback for non-JSON arguments
                        content_text.append(str(item.get("arguments", "")))
                
                content_text.append("\n\n")
        else:
            content_text.append(str(content))
        
        return Panel(content_text, title=title, border_style="yellow", box=box.ROUNDED)
    
    def _format_tool_execution(self, message: Dict[str, Any]) -> Panel:
        """Format a tool execution result message for rich display."""
        source = message.get("source", "Unknown")
        content = message.get("content", [])
        
        title = Text(f"{source} ⚙️", style="bold blue")
        
        # Format the execution result content
        content_text = Text("")
        
        if isinstance(content, list):
            for item in content:
                is_error = item.get("is_error", False)
                status_emoji = "❌ " if is_error else "✅ "
                tool_name = item.get("name", "unknown_tool")
                
                content_text.append(f"{status_emoji}{tool_name}:\n", style="bold red" if is_error else "bold green")
                content_text.append(str(item.get("content", "No content")))
                content_text.append("\n\n")
        else:
            content_text.append(str(content))
        
        return Panel(content_text, title=title, border_style="blue", box=box.ROUNDED)
    
    def _format_handoff_message(self, message: Dict[str, Any]) -> Panel:
        """Format a handoff message for rich display."""
        source = message.get("source", "Unknown")
        target = message.get("target", "Unknown")
        content = message.get("content", "")
        
        title = Text(f"{source} ➡️ {target}", style="bold magenta")
        content_text = Text(content)
        
        return Panel(content_text, title=title, border_style="magenta", box=box.ROUNDED)
    
    def _format_message(self, message: Dict[str, Any], show_metadata: bool = False) -> Optional[Panel]:
        """
        Format a message for rich display based on its type.
        
        Args:
            message: The message to format.
            show_metadata: Whether to show message metadata.
            
        Returns:
            Optional[Panel]: A formatted panel, or None if the message couldn't be formatted.
        """
        message_type = message.get("type", "")
        
        if message_type == "TextMessage" or message_type == "UserMessage":
            return self._format_text_message(message, show_metadata)
        elif message_type == "ToolCallRequestEvent" or message_type == "AssistantMessage":
            return self._format_tool_call(message)
        elif message_type == "ToolCallExecutionEvent" or message_type == "FunctionExecutionResultMessage":
            return self._format_tool_execution(message)
        elif message_type == "HandoffMessage":
            return self._format_handoff_message(message)
        elif message_type == "ToolCallSummaryMessage":
            # Simplified display for summary messages
            source = message.get("source", "Unknown")
            content = message.get("content", "")
            title = Text(f"{source} 📝", style="bold cyan")
            return Panel(Text(content), title=title, border_style="cyan", box=box.ROUNDED)
        else:
            # Default formatting for unknown message types
            source = message.get("source", "Unknown")
            content = message.get("content", "")
            title = Text(f"{source} ({message_type})", style="bold white")
            
            if isinstance(content, list):
                content_text = Text("Complex content - abbreviated")
            else:
                content_text = Text(str(content)[:500] + ("..." if len(str(content)) > 500 else ""))
                
            return Panel(content_text, title=title, border_style="white", box=box.ROUNDED)
    
    def print_conversation(self, agent_name: str = "swarm", show_metadata: bool = False):
        """
        Print the conversation for a specific agent to the console.
        
        Args:
            agent_name: The name of the agent to print the conversation for.
            show_metadata: Whether to show message metadata.
        """
        conversations = self._extract_agent_conversations()
        
        if agent_name not in conversations:
            rich_console.print(f"[bold red]No conversation found for agent: {agent_name}[/bold red]")
            available = ", ".join(conversations.keys())
            rich_console.print(f"Available agents: {available}")
            return
        
        # Print run information
        summary = self.get_run_summary()
        run_info = Table(show_header=False, box=box.SIMPLE)
        run_info.add_column("Property", style="cyan")
        run_info.add_column("Value", style="green")
        
        run_info.add_row("Run ID", summary["id"])
        run_info.add_row("Intent", summary["user_intent"])
        run_info.add_row("Model", f"{summary['model_provider']}/{summary['model']}")
        run_info.add_row("Duration", f"{summary['duration']:.2f} seconds")
        run_info.add_row("Status", 
                        "Completed" if summary["completed"] else f"Stopped: {summary['stop_reason']}")
        
        rich_console.print(Panel(run_info, title="Run Information", border_style="blue"))
        
        # Print the conversation
        messages = conversations[agent_name]
        rich_console.print(f"[bold cyan]Conversation for agent: {agent_name} ({len(messages)} messages)[/bold cyan]")
        
        for i, message in enumerate(messages):
            formatted = self._format_message(message, show_metadata)
            if formatted:
                rich_console.print(formatted)
    
    def get_formatted_conversation_html(self, agent_name: str = "swarm") -> str:
        """
        Get the conversation formatted as HTML for display in Streamlit.
        
        Args:
            agent_name: The name of the agent to get the conversation for.
            
        Returns:
            str: HTML-formatted conversation.
        """
        conversations = self._extract_agent_conversations()
        
        if agent_name not in conversations:
            return f"<div class='error'>No conversation found for agent: {agent_name}</div>"
        
        messages = conversations[agent_name]
        
        html = []
        html.append("<style>")
        html.append(".message-container { margin-bottom: 10px; }")
        html.append(".message { border: 1px solid #ccc; border-radius: 5px; padding: 10px; margin-bottom: 10px; }")
        html.append(".message-header { font-weight: bold; margin-bottom: 5px; }")
        html.append(".user { background-color: #f0f7ff; border-left: 5px solid #3498db; }")
        html.append(".assistant { background-color: #f0fff0; border-left: 5px solid #2ecc71; }")
        html.append(".system { background-color: #fff0f0; border-left: 5px solid #e74c3c; }")
        html.append(".tool { background-color: #fffff0; border-left: 5px solid #f39c12; }")
        html.append(".execution { background-color: #f0f0ff; border-left: 5px solid #9b59b6; }")
        html.append(".handoff { background-color: #fff0ff; border-left: 5px solid #8e44ad; }")
        html.append(".summary { background-color: #f5f5f5; border-left: 5px solid #95a5a6; }")
        html.append(".code { background-color: #f8f8f8; padding: 10px; overflow-x: auto; font-family: monospace; }")
        html.append("</style>")
        
        # Add run summary
        summary = self.get_run_summary()
        html.append("<div class='message-container'>")
        html.append("<div class='message summary'>")
        html.append("<div class='message-header'>Run Summary</div>")
        html.append("<table>")
        html.append(f"<tr><td><b>Run ID:</b></td><td>{summary['id']}</td></tr>")
        html.append(f"<tr><td><b>Intent:</b></td><td>{summary['user_intent']}</td></tr>")
        html.append(f"<tr><td><b>Model:</b></td><td>{summary['model_provider']}/{summary['model']}</td></tr>")
        html.append(f"<tr><td><b>Duration:</b></td><td>{summary['duration']:.2f} seconds</td></tr>")
        status = "Completed" if summary["completed"] else f"Stopped: {summary['stop_reason']}"
        html.append(f"<tr><td><b>Status:</b></td><td>{status}</td></tr>")
        html.append("</table>")
        html.append("</div>")
        html.append("</div>")
        
        # Add conversation
        for message in messages:
            message_type = message.get("type", "")
            source = message.get("source", "Unknown")
            content = message.get("content", "")
            
            # Determine message class
            css_class = "message "
            if source == "user" or message_type == "UserMessage":
                css_class += "user"
            elif message_type in ["ToolCallRequestEvent", "AssistantMessage"]:
                css_class += "tool"
            elif message_type in ["ToolCallExecutionEvent", "FunctionExecutionResultMessage"]:
                css_class += "execution"
            elif message_type == "HandoffMessage":
                css_class += "handoff"
            elif message_type == "ToolCallSummaryMessage":
                css_class += "summary"
            else:
                css_class += "assistant"
            
            html.append("<div class='message-container'>")
            html.append(f"<div class='{css_class}'>")
            html.append(f"<div class='message-header'>{source} ({message_type})</div>")
            
            # Format content based on type
            if isinstance(content, str):
                # Simple text content - escape HTML for safety and convert newlines to <br>
                escaped_content = content.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                html.append(f"<div>{escaped_content}</div>")
            elif isinstance(content, list):
                # Tool calls or complex content
                html.append("<div>")
                for item in content:
                    # For tool calls, show the name and arguments
                    if "name" in item:
                        html.append(f"<b>{item.get('name', 'Unknown Tool')}</b><br>")
                    
                    # Format arguments if available
                    if "arguments" in item:
                        try:
                            if isinstance(item["arguments"], str):
                                if item["arguments"].startswith("{"):
                                    # Try to parse and format JSON arguments
                                    args = json.loads(item["arguments"])
                                    formatted_args = json.dumps(args, indent=2)
                                    html.append(f"<pre class='code'>{formatted_args}</pre>")
                                else:
                                    # Handle non-JSON string arguments
                                    html.append(f"<pre class='code'>{item['arguments']}</pre>")
                            else:
                                # Format direct dict arguments
                                formatted_args = json.dumps(item["arguments"], indent=2)
                                html.append(f"<pre class='code'>{formatted_args}</pre>")
                        except (json.JSONDecodeError, TypeError):
                            # Fallback for non-JSON arguments
                            html.append(f"<pre class='code'>{item.get('arguments', '')}</pre>")
                    
                    # For tool executions, show the response content
                    if "content" in item:
                        output = item["content"]
                        is_error = item.get("is_error", False)
                        
                        if is_error:
                            # Format error with red text
                            html.append(f"<div style='color: red;'>{output}</div>")
                        else:
                            # Check if it's code output (contains traceback, etc.)
                            if isinstance(output, str) and ("\n" in output or output.startswith("Traceback") or output.startswith("import ")):
                                html.append(f"<pre class='code'>{output}</pre>")
                            else:
                                html.append(f"<div>{output}</div>")
                html.append("</div>")
            else:
                # Other types - convert to string
                html.append(f"<div>{str(content)}</div>")
            
            html.append("</div>")
            html.append("</div>")
        
        return "\n".join(html)
    
    def get_available_agents(self) -> List[str]:
        """
        Get a list of available agents in this run's conversation history.
        
        Returns:
            List[str]: List of agent names.
        """
        return list(self._extract_agent_conversations().keys())


def visualize_run(run_id: str, agent_name: str = "swarm", show_metadata: bool = False):
    """
    Visualize a run in the console.
    
    Args:
        run_id: ID of the run to visualize.
        agent_name: Name of the agent to visualize (default: swarm).
        show_metadata: Whether to show message metadata.
    """
    visualizer = RunHistoryVisualizer(run_id)
    visualizer.print_conversation(agent_name, show_metadata)

def get_run_conversation_html(run_id: str, agent_name: str = "swarm") -> str:
    """
    Get HTML-formatted conversation for a run.
    
    Args:
        run_id: ID of the run to visualize.
        agent_name: Name of the agent to visualize (default: swarm).
        
    Returns:
        str: HTML-formatted conversation.
    """
    visualizer = RunHistoryVisualizer(run_id)
    return visualizer.get_formatted_conversation_html(agent_name)

def get_run_agent_list(run_id: str) -> List[str]:
    """
    Get a list of agents in a run.
    
    Args:
        run_id: ID of the run.
        
    Returns:
        List[str]: List of agent names.
    """
    visualizer = RunHistoryVisualizer(run_id)
    return visualizer.get_available_agents()
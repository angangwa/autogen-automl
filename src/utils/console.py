"""
Console utility functions for displaying agent messages in the terminal with rich formatting.
"""

import time
import logging
import asyncio
from typing import AsyncGenerator, List, Optional, Dict, Any, TypeVar, cast, Union

from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.text import Text
from rich.emoji import Emoji
from rich.table import Table
from rich.markdown import Markdown
from rich import box

from autogen_agentchat.base import TaskResult, Response
from autogen_agentchat.messages import (
    BaseChatMessage, BaseAgentEvent,
    TextMessage, MultiModalMessage, ToolCallSummaryMessage, StopMessage, HandoffMessage,
    ToolCallRequestEvent, ToolCallExecutionEvent, ModelClientStreamingChunkEvent,
    ThoughtEvent, UserInputRequestedEvent, MemoryQueryEvent
)
from autogen_agentchat.messages import RequestUsage

# Configure logger - should respect existing settings from __init__.py
logger = logging.getLogger(__name__)

# Create a TypeVar for the return type
T = TypeVar('T', bound=Union[TaskResult, Response])

# Create a rich console for output - disable logging to avoid duplicating errors
rich_console = RichConsole(log_time=False, log_path=False)

# Helper functions for formatting messages
def format_text_message(message: TextMessage) -> Panel:
    """Format a text message for terminal display."""
    title = Text(f"{message.source}", style="bold green")
    content = Text(message.content)
    return Panel(content, title=title, border_style="green", box=box.ROUNDED)

def format_tool_call_request(message: ToolCallRequestEvent, show_technical: bool) -> Panel:
    """Format a tool call request event with appropriate styling."""
    title = Text(f"{message.source} 🔧", style="bold yellow")
    
    if not show_technical:
        tool_names = [call.name for call in message.content]
        content = Text(f"Calling: {', '.join(tool_names)}")
    else:
        content = Text("")
        for call in message.content:
            content.append(f"{call.name}:\n", style="bold")
            content.append(f"{str(call.arguments)}\n\n")
    
    return Panel(content, title=title, border_style="yellow", box=box.ROUNDED)

def format_tool_execution(message: ToolCallExecutionEvent, show_technical: bool) -> Panel:
    """Format a tool execution result with appropriate styling."""
    title = Text(f"{message.source} ⚙️", style="bold blue")
    
    if not show_technical:
        tool_names = [result.name for result in message.content]
        content = Text(f"Executed: {', '.join(tool_names)}")
    else:
        content = Text("")
        for result in message.content:
            status_emoji = "✅ " if not result.is_error else "❌ "
            content.append(f"{status_emoji}{result.name}:\n", style="bold")
            content.append(f"{str(result.content)}\n\n")
    
    return Panel(content, title=title, border_style="blue", box=box.ROUNDED)

def format_streaming_chunk(message: ModelClientStreamingChunkEvent) -> str:
    """Format a streaming chunk event."""
    return message.content

def format_thought_event(message: ThoughtEvent, show_technical: bool) -> Optional[Panel]:
    """Format a thought event."""
    if not show_technical:
        return None  # Don't show thoughts in simple view
    
    title = Text(f"{message.source} 💭", style="bold magenta")
    content = Text(str(message.content))
    return Panel(content, title=title, border_style="magenta", box=box.ROUNDED)

def format_memory_query(message: MemoryQueryEvent) -> Panel:
    """Format a memory query event."""
    title = Text(f"{message.source} 🧠", style="bold cyan")
    content = Text("Memory retrieval completed")
    return Panel(content, title=title, border_style="cyan", box=box.ROUNDED)

def format_task_result(message: TaskResult) -> Panel:
    """Format a task result."""
    title = Text("Task Complete ✅", style="bold green")
    content = Text(f"Reason: {message.stop_reason}")
    return Panel(content, title=title, border_style="green", box=box.ROUNDED)

def format_stats(total_usage: RequestUsage, duration: float) -> Panel:
    """Format usage statistics."""
    title = Text("Stats 📊", style="bold white")
    
    table = Table(box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Prompt Tokens", str(total_usage.prompt_tokens))
    table.add_row("Completion Tokens", str(total_usage.completion_tokens))
    table.add_row("Total Tokens", str(total_usage.prompt_tokens + total_usage.completion_tokens))
    table.add_row("Duration", f"{duration:.2f} seconds")
    
    return Panel(table, title=title, border_style="white", box=box.ROUNDED)

def format_multimodal(message: MultiModalMessage, no_inline_images: bool) -> Panel:
    """Format a multimodal message."""
    title = Text(f"{message.source}", style="bold green")
    content = Text(message.to_text(iterm=not no_inline_images))
    return Panel(content, title=title, border_style="green", box=box.ROUNDED)

async def CustomConsole(
    stream: AsyncGenerator[BaseAgentEvent | BaseChatMessage | T, None],
    *,
    no_inline_images: bool = False,
    output_stats: bool = False,
    show_technical_details: bool = False,
    silent_logging: bool = True,  # Control logger silencing
) -> T:
    """
    Consumes the message stream from autogen run_stream or on_messages_stream
    and renders the messages to the terminal with rich formatting.
    Returns the last processed TaskResult or Response.

    Args:
        stream: Message stream to process and display
        no_inline_images: If True, won't try to render images in terminal
        output_stats: If True, will output token usage statistics
        show_technical_details: If True, will show detailed technical information
        silent_logging: If True, silences logs from autogen and other libraries

    Returns:
        The last processed TaskResult or Response
    """
    # Silence logs if requested
    if silent_logging:
        for logger_name in ["autogen_core", "autogen_agentchat", "httpx", "urllib3"]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        logging.getLogger().setLevel(logging.ERROR)  # Root logger
    
    start_time = time.time()
    total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    
    # Track if we're currently streaming
    is_streaming = False
    streaming_text = ""
    streaming_source = None
    
    last_processed: Optional[T] = None
    
    try:
        async for message in stream:
            # Handle task completion
            if isinstance(message, TaskResult):
                duration = time.time() - start_time
                
                # Format and display the task result message
                task_panel = format_task_result(message)
                rich_console.print(task_panel)
                
                # Maybe show stats
                if output_stats:
                    stats_panel = format_stats(total_usage, duration)
                    rich_console.print(stats_panel)
                
                # Store the result
                last_processed = message  # type: ignore
            
            # Handle response
            elif isinstance(message, Response):
                duration = time.time() - start_time
                
                # Format the chat message in the response
                if hasattr(message, 'chat_message') and message.chat_message:
                    if isinstance(message.chat_message, TextMessage):
                        panel = format_text_message(message.chat_message)
                        rich_console.print(panel)
                    elif isinstance(message.chat_message, MultiModalMessage):
                        panel = format_multimodal(message.chat_message, no_inline_images)
                        rich_console.print(panel)
                
                # Track token usage
                if hasattr(message, 'chat_message') and message.chat_message and message.chat_message.models_usage:
                    total_usage.completion_tokens += message.chat_message.models_usage.completion_tokens
                    total_usage.prompt_tokens += message.chat_message.models_usage.prompt_tokens
                
                # Maybe show stats
                if output_stats:
                    stats_panel = format_stats(total_usage, duration)
                    rich_console.print(stats_panel)
                
                # Store the response
                last_processed = message  # type: ignore
            
            # We don't process UserInputRequestedEvent messages currently
            elif isinstance(message, UserInputRequestedEvent):
                pass
            
            else:
                # Cast required
                message = cast(BaseAgentEvent | BaseChatMessage, message)
                
                # Handle streaming chunks specially
                if isinstance(message, ModelClientStreamingChunkEvent):
                    if not is_streaming:
                        is_streaming = True
                        streaming_text = ""
                        streaming_source = message.source
                        # Print agent name at start of streaming
                        rich_console.print(Text(f"{message.source}: ", style="bold green"), end="")
                    
                    # Print chunk without newline
                    chunk = format_streaming_chunk(message)
                    rich_console.print(chunk, end="")
                    streaming_text += chunk
                else:
                    # If we were streaming, finish with a newline
                    if is_streaming:
                        rich_console.print()
                        is_streaming = False
                    
                    # Format message based on type
                    formatted = None
                    
                    if isinstance(message, TextMessage):
                        formatted = format_text_message(message)
                    elif isinstance(message, MultiModalMessage):
                        formatted = format_multimodal(message, no_inline_images)
                    elif isinstance(message, ToolCallRequestEvent):
                        formatted = format_tool_call_request(message, show_technical_details)
                    elif isinstance(message, ToolCallExecutionEvent):
                        formatted = format_tool_execution(message, show_technical_details)
                    elif isinstance(message, ThoughtEvent):
                        formatted = format_thought_event(message, show_technical_details)
                    elif isinstance(message, MemoryQueryEvent):
                        formatted = format_memory_query(message)
                    
                    # Display formatted message immediately
                    if formatted is not None:
                        rich_console.print(formatted)
                    
                    # Track token usage
                    if hasattr(message, 'models_usage') and message.models_usage:
                        total_usage.completion_tokens += message.models_usage.completion_tokens
                        total_usage.prompt_tokens += message.models_usage.prompt_tokens
    
    except Exception as e:
        logger.error(f"Error processing message stream: {e}", exc_info=True)
        rich_console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    if last_processed is None:
        raise ValueError("No TaskResult or Response was processed.")
    
    return last_processed

async def Console(stream, **kwargs):
    """
    Original Console function from AutoGen - calls the built-in console.
    This is a wrapper for backward compatibility.
    
    For a richer terminal experience with colors and formatting, use CustomConsole instead.
    """
    # Import here to avoid circular imports
    from autogen_agentchat.console import Console as OriginalConsole
    return await OriginalConsole(stream, **kwargs)
# Conversation History Visualization

This document provides detailed information about the conversation history visualization feature in AutoGen AutoML.

## Overview

The conversation history visualization feature allows you to view and analyze the complete interaction between AI agents from previous runs. This is useful for:

- Understanding how the AI agents approached the data analysis task
- Debugging issues in agent reasoning or execution
- Learning from successful runs to improve your own analysis
- Training purposes and demonstrations

## Data Structure

Each run's conversation history is stored in the `run_details.json` file within the run's directory in the `history/` folder. The visualizer parses the `team_state` structure to extract the conversation between agents.

## Visualization Options

### 1. Streamlit UI

The easiest way to view conversation histories is through the Streamlit UI:

1. Start the application with `streamlit run app/streamlit_app.py`
2. Click "Previous Runs" from the main page
3. Select a run from the list
4. Click "View Agent Conversation"

#### Agent Selection

When viewing a conversation, you can select different agent perspectives:

- **swarm**: The overall conversation between all agents (recommended starting point)
- **data_analysis_agent**: The data analysis agent's perspective
- **ideation_agent**: The ideation agent's perspective (if available)

### 2. Terminal Visualization

For command-line users or scripting, use the `visualize_run` function:

```python
from src.utils.history_visualizer import visualize_run

# Basic usage
visualize_run("automl_run_20250422_212715_4bd7549d")

# With options
visualize_run(
    run_id="automl_run_20250422_212715_4bd7549d",
    agent_name="data_analysis_agent",  # View from a specific agent's perspective
    show_metadata=True  # Show additional metadata like token usage
)
```

The terminal visualization uses the Rich library to provide colorized output:
- Green panels for text messages
- Yellow panels for tool calls
- Blue panels for tool execution results
- Magenta panels for handoff messages
- Cyan panels for summary messages

### 3. Programmatic Access

For advanced usage or custom visualizations:

```python
from src.utils.history_visualizer import RunHistoryVisualizer

# Create a visualizer for a specific run
visualizer = RunHistoryVisualizer("automl_run_20250422_212715_4bd7549d")

# Get run summary information
summary = visualizer.get_run_summary()
print(f"Run completed: {summary['completed']}")
print(f"Model used: {summary['model_provider']}/{summary['model']}")

# Get available agents
agents = visualizer.get_available_agents()
print(f"Available agents: {agents}")

# Extract raw conversation data for custom processing
conversations = visualizer._extract_agent_conversations()
```

## Integration with Other Systems

### Web Embedding

You can embed the conversation visualization in other web applications:

```python
from src.utils.history_visualizer import get_run_conversation_html

# Get HTML for embedding
html = get_run_conversation_html("automl_run_20250422_212715_4bd7549d")

# Use in your web application
# e.g., with Flask:
@app.route('/conversation/<run_id>')
def show_conversation(run_id):
    html = get_run_conversation_html(run_id)
    return render_template('conversation.html', conversation_html=html)
```

### Analysis of Multiple Runs

Compare conversations across multiple runs:

```python
from src.utils.history_visualizer import RunHistoryVisualizer
from src.utils.helpers import get_run_history

# Get all runs
runs = get_run_history()

# Analyze patterns across runs
for run in runs:
    run_id = run["id"]
    visualizer = RunHistoryVisualizer(run_id)
    summary = visualizer.get_run_summary()
    
    # Example: Count messages by type for each run
    conversations = visualizer._extract_agent_conversations()
    if "swarm" in conversations:
        message_types = {}
        for msg in conversations["swarm"]:
            msg_type = msg.get("type", "unknown")
            message_types[msg_type] = message_types.get(msg_type, 0) + 1
        
        print(f"Run {run_id}: {message_types}")
```

## Troubleshooting

Common issues and solutions:

1. **"No conversation found for agent"**: The specified agent might not exist in this run. Use `get_available_agents()` to check which agents are available.

2. **Empty or incomplete conversation**: The run might have terminated early or had errors. Check the run summary for completion status.

3. **Missing run_details.json**: The run data might be corrupted or incomplete. Check if the run directory exists and has a valid JSON file.

4. **Performance issues with large conversations**: For very large runs, consider using agent-specific views instead of the full swarm view, or implement pagination in custom visualizations.

## Future Enhancements

Planned features for future versions:

- Export conversations to different formats (PDF, Markdown)
- Search and filter capabilities within conversations
- Visualization of token usage patterns
- Statistical analysis of agent interactions
- Comparison view between multiple runs
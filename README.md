# AutoGen AutoML: AI-Powered Exploratory Data Analysis

AutoGen AutoML is a production-ready application that leverages AI agents to perform exploratory data analysis based on machine learning intent and dataset. It uses the AutoGen framework to analyze data, generate visualizations, and create comprehensive reports.

## Features

- **AI-Powered Analysis**: Uses Claude to analyze data and suggest ML approaches
- **Interactive Mode**: AI can ask clarifying questions during analysis
- **Streamlit UI**: User-friendly interface for uploading data and viewing results
- **Docker Integration**: Runs code in isolated containers for security
- **Extensible Architecture**: Designed for future expansion with more agents



https://github.com/user-attachments/assets/9778bd16-b474-454c-b08c-ccdaf6eeda7f



## Project Structure

```
autogen-automl/
├── src/                  # Source code
│   ├── agents/           # Agent definitions
│   ├── tools/            # Tools for agents
│   ├── prompts/          # Prompt templates
│   ├── executors/        # Code execution environments
│   ├── config/           # Configuration management
│   └── utils/            # Helper functions
├── app/                  # Streamlit application
├── data/                 # Sample data (gitignored)
├── outputs/              # Output directory (gitignored)
└── tests/                # Test suite
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/angangwa/autogen-automl.git
   cd autogen-automl
   ```

2. Install UV (if you haven't already):
   ```bash
   pip install uv
   ```

3. Create a virtual environment and activate it:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install the package with UV:
   ```bash
   uv pip install -r requirements.txt
   ```

5. Create a `.env` file with your configuration (see example env file and `src/config/settings.py` for available settings):

## Running the Application

### Run the analysis directly

```python
import asyncio
from src import run_analysis
asyncio.run(run_analysis(
    user_intent = "I have Cytology features of breast cancer biopsy. I want to use it to predict breast cancer",
    data_dir = "data",
    outputs_dir = "outputs",
    interactive = True,
    docker_wait_time = 30,
    max_turns = 20
))
# or use await run_analysis(...)
```

### Start the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

Then open your browser to http://localhost:8501

## Usage

1. **Upload Data**: Upload your dataset files using the sidebar uploader
2. **Define Intent**: Enter your machine learning intent in the text area
3. **Run Analysis**: Click the "Run Analysis" button
4. **View Results**: Explore the markdown reports, code, and visualizations in the results panel
5. **Answer Questions**: If in interactive mode, respond to any questions from the AI

## Output Files

The application generates four main output files:

1. **refactored_intent.md**: A markdown description of the clarified intent
2. **dataset_description.md**: A markdown description of the dataset and analysis results
3. **analysis.py**: Clean, commented Python code to reproduce the analysis
4. **analysis_result.md**: Complete analysis result including visualizations

## Conversation History Visualization

The application includes a feature to visualize the complete conversation history between agents from previous runs:

### Using the Streamlit UI

1. Navigate to "Previous Runs" from the main page
2. Select a run from the list
3. Click "View Agent Conversation" button
4. In the conversation view:
   - Select an agent to view its perspective (e.g., "swarm", "data_analysis_agent", etc.)
   - The conversation is displayed with color-coded messages by type
   - Click "View Analysis Results" to return to the results view

### Using the Command Line

You can also visualize past runs from a Python script or terminal:

```python
# From a Python script:
from src.utils.history_visualizer import visualize_run

# Display conversation in terminal with rich formatting
visualize_run("automl_run_20250422_212715_4bd7549d")  # Replace with your run ID

# Specify a particular agent (e.g., data_analysis_agent instead of the default "swarm")
visualize_run("automl_run_20250422_212715_4bd7549d", agent_name="data_analysis_agent")

# Include token usage metadata
visualize_run("automl_run_20250422_212715_4bd7549d", show_metadata=True)
```

### Programmatic Usage

```python
from src.utils.history_visualizer import (
    RunHistoryVisualizer, get_run_conversation_html, get_run_agent_list
)

# Get available agents in a run
agents = get_run_agent_list("automl_run_20250422_212715_4bd7549d")
print(f"Available agents: {agents}")

# Get HTML-formatted conversation for embedding in a web page
html = get_run_conversation_html("automl_run_20250422_212715_4bd7549d")

# Create a visualizer instance for more control
visualizer = RunHistoryVisualizer("automl_run_20250422_212715_4bd7549d")
run_summary = visualizer.get_run_summary()
print(f"Run duration: {run_summary['duration']} seconds")
```

## Extending the Application

### Adding New Agents

1. Create a new agent class in `src/agents/` that inherits from `BaseAgent`
2. Implement the required methods
3. Add the agent to the appropriate team in `src/teams/`

### Adding New Tools

1. Create a new tool function in `src/tools/`
2. Wrap it with `FunctionTool` and add to the appropriate agent

## Requirements

- Python 3.11+
- Docker
- Azure/OpenAI/Gemini/Anthropic API key
- UV or venv for dependency management

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# AutoGen AutoML: AI-Powered Exploratory Data Analysis

AutoGen AutoML is a production-ready application that leverages AI agents to perform exploratory data analysis based on machine learning intent and dataset. It uses the AutoGen framework and Anthropic's Claude to analyze data, generate visualizations, and create comprehensive reports.

## Features

- **AI-Powered Analysis**: Uses Claude to analyze data and suggest ML approaches
- **Interactive Mode**: AI can ask clarifying questions during analysis
- **Streamlit UI**: User-friendly interface for uploading data and viewing results
- **Docker Integration**: Runs code in isolated containers for security
- **Extensible Architecture**: Designed for future expansion with more agents

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
   git clone https://github.com/yourusername/autogen-automl.git
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
   uv pip install -e .
   ```

5. Create a `.env` file with your configuration:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

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

## Extending the Application

### Adding New Agents

1. Create a new agent class in `src/agents/` that inherits from `BaseAgent`
2. Implement the required methods
3. Add the agent to the appropriate team in `src/teams/`

### Adding New Tools

1. Create a new tool function in `src/tools/`
2. Wrap it with `FunctionTool` and add to the appropriate agent

## Dependency Management

### Adding New Dependencies

To add new dependencies:

```bash
uv pip add package_name
```

### Updating Dependencies

To update all dependencies to their latest versions:

```bash
uv pip install -e . --upgrade
```

### Recording Exact Versions

To record exact versions for reproducibility:

```bash
uv pip freeze > requirements.lock
```

### Installing from Lock File

To install exact versions on another machine:

```bash
uv pip install -r requirements.lock
```

## Requirements

- Python 3.8+
- Docker
- Anthropic API key
- UV for dependency management

## License

This project is licensed under the MIT License - see the LICENSE file for details.
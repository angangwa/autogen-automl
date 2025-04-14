# Project Structure

```
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py             # Base agent class
│   │   └── data_analysis.py    # Data Analysis Agent
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── file_tools.py       # File-related tools
│   │   ├── image_tools.py      # Image analysis tools
│   │   └── code_tools.py       # Code execution wrapper
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── data_analysis.py    # Prompts for data analysis agent
│   ├── executors/
│   │   ├── __init__.py
│   │   └── docker.py           # Docker executor setup
│   ├── teams/
│   │   ├── __init__.py
│   │   └── data_exploration.py # Team setup for data exploration
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         # Configuration settings
│   └── utils/
│       ├── __init__.py
│       └── helpers.py          # Helper functions
├── app/
│   ├── __init__.py
│   └── streamlit_app.py        # Streamlit application
├── data/                       # Sample data (gitignored)
├── outputs/                    # Output directory (gitignored)
├── tests/                      # Tests
│   ├── __init__.py
│   └── test_agents.py
├── .env.example                # Example environment variables
├── .gitignore                  # Git ignore file
├── README.md                   # Project documentation
├── requirements.txt            # Project dependencies
└── setup.py                    # Setup script
```
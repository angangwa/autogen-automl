"""
Streamlit application for the AutoGen EDA application.
"""

import os
import asyncio
import streamlit as st
import pandas as pd
from pathlib import Path
import base64
import logging
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import from src
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src import run_analysis
from src.utils.helpers import get_output_files, read_file_content

# Set page config
st.set_page_config(
    page_title="AutoGen EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define functions
def generate_html_preview(file_path: str) -> str:
    """
    Generate HTML preview for a file.
    
    Args:
        file_path: The path to the file.
        
    Returns:
        str: HTML code for the preview.
    """
    content = read_file_content(file_path)
    if content is None:
        return "<p>Error reading file.</p>"
    
    return content

def generate_image_preview(file_path: str) -> str:
    """
    Generate HTML for an image preview.
    
    Args:
        file_path: The path to the image file.
        
    Returns:
        str: HTML code for the image preview.
    """
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    
    ext = Path(file_path).suffix.lower()[1:]  # Remove the dot
    return f'<img src="data:image/{ext};base64,{data}" style="max-width: 100%; max-height: 500px;">'

# Define app title and description
st.title("📊 AutoGen EDA: AI-Powered Data Analysis")
st.markdown("""
This application uses AI agents to perform exploratory data analysis based on your machine learning intent.
Upload your data, describe what you want to do, and let the AI do the rest!
""")

# Create sidebar for settings and upload
with st.sidebar:
    st.header("Settings & Data Upload")
    
    # Data upload
    st.subheader("Upload Data")
    uploaded_files = st.file_uploader(
        "Upload your dataset files", 
        accept_multiple_files=True
    )
    
    # Save uploaded files
    if uploaded_files:
        data_dir = Path(settings.DATA_DIR)
        data_dir.mkdir(exist_ok=True, parents=True)
        
        for uploaded_file in uploaded_files:
            file_path = data_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"Saved {uploaded_file.name} to data directory.")
    
    # Advanced settings
    st.subheader("Advanced Settings")
    
    max_turns = st.slider(
        "Maximum Conversation Turns",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Maximum number of conversation turns for the AI agent."
    )
    
    interactive = st.checkbox(
        "Interactive Mode",
        value=True,
        help="Allow the AI to ask questions during analysis."
    )
    
    docker_wait_time = st.slider(
        "Docker Initialization Time (seconds)",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        help="Time to wait for Docker container to initialize."
    )

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    # ML intent input
    st.header("Machine Learning Intent")
    user_intent = st.text_area(
        "Describe what you want to do with your data",
        height=150,
        placeholder="Example: I want to predict customer churn based on their usage patterns and demographics."
    )
    
    # Run analysis button
    run_button = st.button("Run Analysis", type="primary")
    
    # Progress bar and status
    if run_button:
        if not user_intent:
            st.error("Please provide a machine learning intent.")
        elif not uploaded_files and not os.listdir(settings.DATA_DIR):
            st.error("Please upload data files.")
        else:
            # Set up progress and status
            progress_bar = st.progress(0)
            status = st.empty()
            
            # Run the analysis
            status.info("Starting analysis...")
            progress_bar.progress(10)
            
            # Set up the outputs directory
            outputs_dir = Path(settings.OUTPUTS_DIR)
            outputs_dir.mkdir(exist_ok=True, parents=True)
            
            # Clear existing output files
            for file in outputs_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            
            progress_bar.progress(20)
            status.info("Running analysis... (this might take a few minutes)")
            
            # Run the analysis
            try:
                # Use asyncio to run the async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Run the analysis
                results = loop.run_until_complete(
                    run_analysis(
                        user_intent=user_intent,
                        interactive=interactive,
                        docker_wait_time=docker_wait_time,
                        max_turns=max_turns,
                    )
                )
                
                # Close the loop
                loop.close()
                
                # Update progress
                progress_bar.progress(100)
                
                if results["completed"]:
                    status.success("Analysis completed successfully!")
                else:
                    status.warning(f"Analysis stopped: {results['stop_reason']}")
                
                # Store results in session state
                st.session_state.analysis_results = results
                st.session_state.analysis_completed = True
                
                # Rerun to show results
                st.rerun()
            except Exception as e:
                status.error(f"Error running analysis: {str(e)}")
                logger.error(f"Error running analysis: {e}", exc_info=True)

with col2:
    # Results panel (only show if analysis has been run)
    if st.session_state.get("analysis_completed", False):
        st.header("Analysis Results")
        
        # Get output files
        output_files = get_output_files(settings.OUTPUTS_DIR)
        
        # Show tabs for different file types
        if any(output_files.values()):
            tabs = st.tabs(["Markdown", "Code", "Visualizations", "HTML"])
            
            # Markdown tab
            with tabs[0]:
                if output_files["markdown"]:
                    for file in output_files["markdown"]:
                        file_path = Path(settings.OUTPUTS_DIR) / file
                        st.subheader(file)
                        content = read_file_content(str(file_path))
                        if content:
                            st.markdown(content)
                        else:
                            st.error(f"Could not read file: {file}")
                else:
                    st.info("No markdown files generated.")
            
            # Code tab
            with tabs[1]:
                if output_files["python"]:
                    for file in output_files["python"]:
                        file_path = Path(settings.OUTPUTS_DIR) / file
                        st.subheader(file)
                        content = read_file_content(str(file_path))
                        if content:
                            st.code(content, language="python")
                        else:
                            st.error(f"Could not read file: {file}")
                else:
                    st.info("No Python files generated.")
            
            # Visualizations tab
            with tabs[2]:
                if output_files["images"]:
                    for file in output_files["images"]:
                        file_path = Path(settings.OUTPUTS_DIR) / file
                        st.subheader(file)
                        st.image(str(file_path))
                else:
                    st.info("No image files generated.")
            
            # HTML tab
            with tabs[3]:
                if output_files["html"]:
                    for file in output_files["html"]:
                        file_path = Path(settings.OUTPUTS_DIR) / file
                        st.subheader(file)
                        html_content = generate_html_preview(str(file_path))
                        st.components.v1.html(html_content, height=500, scrolling=True)
                else:
                    st.info("No HTML files generated.")
            
        else:
            st.info("No output files generated yet.")
    else:
        st.info("Run an analysis to see results here.")

# Footer
st.markdown("---")
st.markdown(
    "AutoGen EDA: An AI-powered Exploratory Data Analysis Application | "
    "Built with Streamlit, AutoGen, and Claude"
)

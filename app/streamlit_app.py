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
from src.utils.helpers import (
    get_output_files, read_file_content, 
    get_run_history, get_run_details, load_run_to_current,
    get_available_examples, load_example_to_current
)

# Set page config
st.set_page_config(
    page_title="AutoGen EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "analysis_completed" not in st.session_state:
    st.session_state.analysis_completed = False
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "current_view" not in st.session_state:
    st.session_state.current_view = "main"  # Options: main, history, viewer, examples
if "selected_run" not in st.session_state:
    st.session_state.selected_run = None
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None
if "file_category" not in st.session_state:
    st.session_state.file_category = None
if "loaded_example" not in st.session_state:
    st.session_state.loaded_example = None

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

def change_view(view_name):
    """Change the current view in the session state."""
    st.session_state.current_view = view_name
    if view_name == "main":
        st.session_state.selected_run = None
        st.session_state.selected_file = None
        st.session_state.file_category = None

def set_selected_run(run_id):
    """Set the selected run and load it to current workspace."""
    st.session_state.selected_run = run_id
    # Load run details
    if run_id:
        load_run_to_current(run_id)
        st.session_state.analysis_completed = True

def display_file_viewer(directory, file_category, file_name):
    """Display a file in the file viewer based on its category."""
    file_path = Path(directory) / file_name
    
    st.subheader(file_name)
    
    # Add a download button for the file
    with open(file_path, "rb") as file:
        btn = st.download_button(
            label="Download file",
            data=file,
            file_name=file_name,
            key=f"download_{file_name}"
        )
    
    # Display file based on category
    if file_category == "markdown":
        content = read_file_content(str(file_path))
        if content:
            st.markdown(content)
        else:
            st.error(f"Could not read file: {file_name}")
    
    elif file_category == "python":
        content = read_file_content(str(file_path))
        if content:
            st.code(content, language="python")
        else:
            st.error(f"Could not read file: {file_name}")
    
    elif file_category == "images":
        st.image(str(file_path))
        
        # If there's a corresponding HTML file, provide a link
        html_file = file_path.with_suffix('.html')
        if html_file.exists():
            html_file_name = html_file.name
            st.info(f"This image has an interactive HTML version: {html_file_name}")
    
    elif file_category == "html":
        html_content = generate_html_preview(str(file_path))
        st.components.v1.html(html_content, height=600, scrolling=True)
        
        # If there's a corresponding image file, provide a thumbnail
        img_file = file_path.with_suffix('.jpg')
        if img_file.exists():
            with st.expander("Preview as Image"):
                st.image(str(img_file))

def set_loaded_example(example):
    """Set the loaded example and copy its data to the data directory."""
    if load_example_to_current(example):
        st.session_state.loaded_example = example
        # Pre-fill the intent text
        if "user_intent" in example:
            st.session_state.user_intent = example["user_intent"]
        # Update advanced settings if present in the example
        if "interactive" in example:
            st.session_state.interactive = example["interactive"]
        if "docker_wait_time" in example:
            st.session_state.docker_wait_time = example["docker_wait_time"]
        if "max_turns" in example:
            st.session_state.max_turns = example["max_turns"]
        return True
    return False

# Main App Structure
def main():
    # Define app title and description
    st.title("📊 AutoGen EDA: AI-Powered Data Analysis")
    
    # Choose the view based on session state
    if st.session_state.current_view == "history":
        display_history_view()
    elif st.session_state.current_view == "viewer":
        display_file_viewer_page()
    elif st.session_state.current_view == "examples":
        display_examples_view()
    else:
        display_main_view()

# Examples View
def display_examples_view():
    st.markdown("""
    ## Examples
    Browse and load example datasets with pre-configured analysis intents.
    """)
    
    # Add back button
    if st.button("← Back to Main", key="back_from_examples"):
        change_view("main")
        st.rerun()
    
    # Get available examples
    examples = get_available_examples()
    
    if not examples:
        st.info("No examples found.")
        return
    
    st.write(f"Found {len(examples)} examples")
    
    # Display examples in a table
    example_data = []
    for i, example in enumerate(examples):
        data_path = example.get("data", "")
        if isinstance(data_path, str) and Path(data_path).is_file():
            data_type = "File"
        else:
            data_type = "Directory"
        
        example_data.append({
            "ID": i,
            "Name": example.get("name", f"Example {i}"),
            "Data Type": data_type,
            "Data Path": data_path,
            "Intent": example.get("user_intent", "No intent specified")[:50] + "..." if len(example.get("user_intent", "")) > 50 else example.get("user_intent", "No intent specified")
        })
    
    df = pd.DataFrame(example_data)
    
    # Use Streamlit's data editor for better UX
    selection = st.data_editor(
        df,
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn("ID"),
            "Name": st.column_config.TextColumn("Name"),
            "Data Type": st.column_config.TextColumn("Data Type"),
            "Data Path": st.column_config.TextColumn("Data Path"),
            "Intent": st.column_config.TextColumn("Intent"),
        },
        use_container_width=True,
        disabled=True,
    )
    
    # Allow selection of an example
    selected_example_index = st.selectbox(
        "Select an example to load:",
        options=list(range(len(example_data))),
        format_func=lambda i: f"{example_data[i]['Name']}"
    )
    
    # Show more details about the selected example
    st.subheader("Example Details")
    selected_example = examples[selected_example_index]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Name:**", selected_example.get("name", "Unnamed"))
        st.write("**Data Path:**", selected_example.get("data", "Not specified"))
        
        # Display other settings if available
        if "interactive" in selected_example:
            st.write("**Interactive Mode:**", "Yes" if selected_example["interactive"] else "No")
        if "docker_wait_time" in selected_example:
            st.write("**Docker Wait Time:**", f"{selected_example['docker_wait_time']} seconds")
        if "max_turns" in selected_example:
            st.write("**Max Turns:**", selected_example["max_turns"])
    
    with col2:
        st.write("**User Intent:**")
        st.info(selected_example.get("user_intent", "No intent specified"))
    
    # Add a load button
    if st.button("Load Selected Example", key="load_example"):
        if set_loaded_example(selected_example):
            # Return to main view
            change_view("main")
            st.success(f"Loaded example: {selected_example.get('name', 'Unnamed')}")
            st.rerun()
        else:
            st.error(f"Failed to load example: {selected_example.get('name', 'Unnamed')}")

# History View
def display_history_view():
    st.markdown("""
    ## Previous Runs
    Browse and load previous analysis runs.
    """)
    
    # Add back button
    if st.button("← Back to Main", key="back_from_history"):
        change_view("main")
        st.rerun()
    
    # Get run history
    runs = get_run_history()
    
    if not runs:
        st.info("No previous runs found.")
        return
    
    st.write(f"Found {len(runs)} previous runs")
    
    # Display runs in a table
    run_data = []
    for run in runs:
        run_data.append({
            "ID": run["id"],
            "Date": run.get("date_formatted", "Unknown"),
            "Intent": run.get("user_intent", "Unknown"),
            "Duration": f"{run.get('duration', 0):.1f} seconds",
            "Status": "Completed" if run.get("completed", False) else f"Stopped: {run.get('stop_reason', 'Unknown')}"
        })
    
    df = pd.DataFrame(run_data)
    
    # Use Streamlit's data editor for better UX
    selection = st.data_editor(
        df,
        hide_index=True,
        column_config={
            "ID": st.column_config.TextColumn("Run ID"),
            "Date": st.column_config.TextColumn("Date"),
            "Intent": st.column_config.TextColumn("User Intent"),
            "Duration": st.column_config.TextColumn("Duration"),
            "Status": st.column_config.TextColumn("Status"),
        },
        use_container_width=True,
        disabled=True,
    )
    
    # Allow selection of a run
    selected_run_index = st.selectbox(
        "Select a run to load:",
        options=list(range(len(run_data))),
        format_func=lambda i: f"{run_data[i]['ID']} - {run_data[i]['Date']} - {run_data[i]['Intent'][:50]}..."
    )
    
    # Add a load button
    if st.button("Load Selected Run", key="load_run"):
        run_id = run_data[selected_run_index]["ID"]
        
        # Set the selected run and load its data
        set_selected_run(run_id)
        
        # Return to main view
        change_view("main")
        st.success(f"Loaded run: {run_id}")
        st.rerun()

# File Viewer Page
def display_file_viewer_page():
    st.markdown("## File Viewer")
    
    # Add back button
    if st.button("← Back to Results", key="back_from_viewer"):
        change_view("main" if not st.session_state.selected_run else "main")
        st.rerun()
    
    # Display file content
    if st.session_state.selected_file and st.session_state.file_category:
        directory = settings.OUTPUTS_DIR
        display_file_viewer(
            directory, 
            st.session_state.file_category, 
            st.session_state.selected_file
        )
    else:
        st.error("No file selected.")
        change_view("main")
        st.rerun()

# Main View
def display_main_view():
    st.markdown("""
    This application uses AI agents to perform exploratory data analysis based on your machine learning intent.
    Upload your data, describe what you want to do, and let the AI do the rest!
    """)
    
    # Create sidebar for settings and upload
    with st.sidebar:
        st.header("Settings & Data Upload")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Examples", use_container_width=True):
                change_view("examples")
                st.rerun()
                
        with col2:
            if st.button("Previous Runs", use_container_width=True):
                change_view("history")
                st.rerun()
        
        # Show currently loaded content (example or run)
        if st.session_state.loaded_example:
            example_name = st.session_state.loaded_example.get("name", "Unnamed Example")
            st.info(f"Using example: {example_name}")
            if st.button("Clear Loaded Example"):
                st.session_state.loaded_example = None
                st.session_state.user_intent = ""
                st.rerun()
        elif st.session_state.selected_run:
            st.info(f"Using data from run: {st.session_state.selected_run}")
            if st.button("Clear Loaded Run"):
                st.session_state.selected_run = None
                st.rerun()
        
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
        
        # Show current data files
        data_files = os.listdir(settings.DATA_DIR)
        if data_files:
            with st.expander("Current Data Files"):
                for file in data_files:
                    file_path = Path(settings.DATA_DIR) / file
                    if file_path.is_file():
                        st.text(f"📄 {file}")
                    elif file_path.is_dir():
                        st.text(f"📁 {file}")
        
        # Advanced settings
        st.subheader("Advanced Settings")
        
        # Use session state or default values for settings
        max_turns_value = st.session_state.get("max_turns", 20)
        interactive_value = st.session_state.get("interactive", True)
        docker_wait_time_value = st.session_state.get("docker_wait_time", 30)
        
        max_turns = st.slider(
            "Maximum Conversation Turns before asking the agent to wrap up",
            min_value=5,
            max_value=50,
            value=max_turns_value,
            step=5,
            help="Maximum number of Conversation Turns before asking the agent to wrap up."
        )
        st.session_state.max_turns = max_turns
        
        interactive = st.checkbox(
            "Interactive Mode",
            value=interactive_value,
            help="Allow the AI to ask questions during analysis."
        )
        st.session_state.interactive = interactive
        
        docker_wait_time = st.slider(
            "Docker Initialization Time (seconds)",
            min_value=5,
            max_value=30,
            value=docker_wait_time_value,
            step=5,
            help="Docker container initialization wait time."
        )
        st.session_state.docker_wait_time = docker_wait_time
        
        save_history = st.checkbox(
            "Save Run History",
            value=True,
            help="Save this run to history for later reference."
        )
        
        cleanup_before_run = st.checkbox(
            "Clean Before Run",
            value=True,
            help="Clean output directory before running."
        )

    # Main content area - using full width instead of columns
    # ML intent input
    st.header("Machine Learning Intent")
    
    # Use session state or empty string for intent
    user_intent_value = st.session_state.get("user_intent", "")
    
    user_intent = st.text_area(
        "Describe what you want to do with your data",
        value=user_intent_value,
        height=150,
        placeholder="Example: I want to predict customer churn based on their usage patterns and demographics."
    )
    st.session_state.user_intent = user_intent
    
    # Run analysis button
    run_button = st.button("Run Analysis", type="primary")
    
    # Progress bar and status
    if run_button:
        if not user_intent:
            st.error("Please provide a machine learning intent.")
        elif not os.listdir(settings.DATA_DIR):
            st.error("Please upload data files.")
        else:
            # Set up progress and status
            progress_bar = st.progress(0)
            status = st.empty()
            
            # Run the analysis
            status.info("Starting analysis...")
            progress_bar.progress(10)
            
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
                        save_history=save_history,
                        cleanup_before_run=cleanup_before_run,
                    )
                )
                
                # Close the loop
                loop.close()
                
                # Update progress
                progress_bar.progress(100)
                
                if results["completed"]:
                    status.success("Analysis completed successfully!")
                    if save_history and "run_id" in results:
                        st.session_state.run_id = results["run_id"]
                        status.info(f"Run saved with ID: {results['run_id']}")
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
    
    # Results panel (only show if analysis has been run) - now below the run button
    if st.session_state.get("analysis_completed", False):
        st.markdown("---")
        st.header("Analysis Results")
        
        # Get output files
        output_files = get_output_files(settings.OUTPUTS_DIR)
        
        # Show tabs for different file types
        if any(output_files.values()):
            # Create file category tabs
            tabs = st.tabs(["Markdown", "Code", "Visualizations", "HTML", "All Files"])
            
            # Markdown tab
            with tabs[0]:
                if output_files["markdown"]:
                    # Create a dropdown to select markdown files
                    selected_md = st.selectbox(
                        "Select a markdown file:", 
                        options=output_files["markdown"],
                        key="md_select"
                    )
                    
                    # Display file preview
                    file_path = Path(settings.OUTPUTS_DIR) / selected_md
                    content = read_file_content(str(file_path))
                    if content:
                        st.markdown(content)
                    else:
                        st.error(f"Could not read file: {selected_md}")
                    
                    # Add button to view in dedicated viewer
                    if st.button("Open in Full Viewer", key="view_md"):
                        st.session_state.selected_file = selected_md
                        st.session_state.file_category = "markdown"
                        change_view("viewer")
                        st.rerun()
                else:
                    st.info("No markdown files generated.")
            
            # Code tab
            with tabs[1]:
                if output_files["python"]:
                    selected_py = st.selectbox(
                        "Select a Python file:", 
                        options=output_files["python"],
                        key="py_select"
                    )
                    
                    file_path = Path(settings.OUTPUTS_DIR) / selected_py
                    content = read_file_content(str(file_path))
                    if content:
                        st.code(content, language="python")
                    else:
                        st.error(f"Could not read file: {selected_py}")
                        
                    if st.button("Open in Full Viewer", key="view_py"):
                        st.session_state.selected_file = selected_py
                        st.session_state.file_category = "python"
                        change_view("viewer")
                        st.rerun()
                else:
                    st.info("No Python files generated.")
            
            # Visualizations tab - improved with thumbnails and selection
            with tabs[2]:
                if output_files["images"]:
                    # Group images by type for better organization
                    image_groups = {}
                    for img_file in output_files["images"]:
                        # Extract category from filename (e.g., distinctive_words_action.jpg -> distinctive_words)
                        parts = img_file.split('_')
                        if len(parts) >= 2:
                            category = '_'.join(parts[:-1])
                            if category not in image_groups:
                                image_groups[category] = []
                            image_groups[category].append(img_file)
                        else:
                            # Fallback for files that don't match the pattern
                            if "other" not in image_groups:
                                image_groups["other"] = []
                            image_groups["other"].append(img_file)
                    
                    # Create expanders for each image group
                    for group, images in image_groups.items():
                        with st.expander(f"{group.replace('_', ' ').title()} ({len(images)})", expanded=True):
                            # Create a grid for thumbnails (3 columns)
                            cols = st.columns(3)
                            
                            for i, img_file in enumerate(sorted(images)):
                                with cols[i % 3]:
                                    st.image(str(Path(settings.OUTPUTS_DIR) / img_file), 
                                            caption=img_file, 
                                            use_container_width=True)
                                    
                                    if st.button("View Full", key=f"view_{img_file}"):
                                        st.session_state.selected_file = img_file
                                        st.session_state.file_category = "images"
                                        change_view("viewer")
                                        st.rerun()
                else:
                    st.info("No image files generated.")
            
            # HTML tab
            with tabs[3]:
                if output_files["html"]:
                    selected_html = st.selectbox(
                        "Select an HTML file:", 
                        options=output_files["html"],
                        key="html_select"
                    )
                    
                    file_path = Path(settings.OUTPUTS_DIR) / selected_html
                    html_content = generate_html_preview(str(file_path))
                    st.components.v1.html(html_content, height=600, scrolling=True)
                    
                    if st.button("Open in Full Viewer", key="view_html"):
                        st.session_state.selected_file = selected_html
                        st.session_state.file_category = "html"
                        change_view("viewer")
                        st.rerun()
                else:
                    st.info("No HTML files generated.")
            
            # All Files tab - comprehensive list
            with tabs[4]:
                # Combine all files
                all_files = []
                for category, files in output_files.items():
                    for file in files:
                        all_files.append((file, category))
                
                # Sort files alphabetically
                all_files.sort(key=lambda x: x[0])
                
                if all_files:
                    st.write("All output files:")
                    
                    # Create a clean table of files
                    files_data = []
                    for file, category in all_files:
                        files_data.append({
                            "Filename": file,
                            "Type": category.capitalize(),
                            "Size": f"{Path(settings.OUTPUTS_DIR, file).stat().st_size / 1024:.1f} KB"
                        })
                    
                    # Display as table
                    selection = st.data_editor(
                        pd.DataFrame(files_data),
                        hide_index=True,
                        use_container_width=True,
                        disabled=True
                    )
                    
                    # Allow selection
                    selected_file_index = st.selectbox(
                        "Select a file to view:",
                        options=list(range(len(all_files))),
                        format_func=lambda i: f"{all_files[i][0]} ({all_files[i][1]})"
                    )
                    
                    if st.button("View Selected File", key="view_selected"):
                        selected_filename, selected_category = all_files[selected_file_index]
                        st.session_state.selected_file = selected_filename
                        st.session_state.file_category = selected_category
                        change_view("viewer")
                        st.rerun()
                else:
                    st.info("No output files generated.")
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

# Launch the app
if __name__ == "__main__":
    main()

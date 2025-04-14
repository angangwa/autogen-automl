"""
Image analysis tools for the AutoGen EDA application.
"""

import base64
import logging
import os
from pathlib import Path
from typing import Optional
import anthropic

from src.config import settings

logger = logging.getLogger(__name__)

async def analyze_image(
    image_path: str, 
    query: str = "Analyze this data visualization image and describe what you see. Focus on trends, patterns, outliers, and any insights that would be relevant for data analysis."
) -> str:
    """
    Analyze an image using Anthropic's Claude vision capabilities.
    
    Args:
        image_path: Path to the image file. Relative to output directory. e.g. "image.jpg".
        query: The query to ask about the image. Defaults to a generic plot analysis.
        
    Returns:
        str: Description of the image content.
    """
    try:
        # Normalize path to handle both /mnt/outputs/ and direct paths
        image_path = image_path.replace("/mnt/outputs/", "")
        full_path = Path(f"{settings.OUTPUTS_DIR}/{image_path}")
        
        if not full_path.exists():
            logger.error(f"Image file not found: {full_path}")
            return f"Error: Image file not found at {full_path}"
        
        # Read the image file
        with open(full_path, "rb") as image_file:
            image_data = image_file.read()
        
        # Initialize the Anthropic client
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        
        # Create the message with the image
        response = client.messages.create(
            model=settings.AI_MODEL,
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64", 
                                "media_type": "image/jpeg", 
                                "data": base64.b64encode(image_data).decode("utf-8")
                            }
                        }
                    ]
                }
            ]
        )
        
        logger.info(f"Successfully analyzed image: {image_path}")
        return response.content[0].text
    except Exception as e:
        logger.error(f"Error analyzing image {image_path}: {str(e)}")
        return f"Error analyzing image: {str(e)}"

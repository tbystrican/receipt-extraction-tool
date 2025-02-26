import os
from openai import OpenAI
from PIL import Image
import base64
import json
import logging
import re
import numpy as np
import cv2
import traceback

# Set up logger
logger = logging.getLogger('receipt_extractor.extract')

def extract_receipts(image_path, bounding_boxes):
    """
    Extract receipts from an image using bounding boxes.
    
    Args:
        image_path (str): The path to the image.
        bounding_boxes (list): A list of bounding boxes.
        
    Returns:
        list: A list of extracted receipt images.
    """
    try:
        logger.info(f"Extracting receipts from {image_path} with {len(bounding_boxes)} bounding boxes")
        
        # Load the image
        try:
            image = Image.open(image_path)
            image_np = np.array(image)
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {str(e)}")
            return []
        
        # Convert to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 3:  # RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        receipts = []
        
        for i, bbox in enumerate(bounding_boxes):
            try:
                logger.debug(f"Processing bounding box {i+1}: {bbox}")
                
                # Extract coordinates from the bounding box
                if isinstance(bbox, dict) and "coordinates" in bbox:
                    coordinates = bbox["coordinates"]
                elif isinstance(bbox, dict) and "bbox" in bbox:
                    coordinates = bbox["bbox"]
                elif isinstance(bbox, dict) and "bbox_2d" in bbox:
                    coordinates = bbox["bbox_2d"]
                    logger.debug(f"Found bbox_2d format: {coordinates}")
                elif isinstance(bbox, list):
                    coordinates = bbox
                else:
                    logger.warning(f"Unrecognized bounding box format: {bbox}")
                    continue
                
                # Convert coordinates to integers
                try:
                    if len(coordinates) == 4:  # [x, y, width, height] or [x1, y1, x2, y2]
                        x1, y1, x2, y2 = coordinates
                        # Check if it's [x, y, width, height]
                        if x2 < x1 or y2 < y1:
                            x2, y2 = x1 + x2, y1 + y2
                    elif len(coordinates) == 8:  # [x1, y1, x2, y2, x3, y3, x4, y4]
                        # For rotated bounding boxes, we need to find the min/max coordinates
                        xs = coordinates[0::2]
                        ys = coordinates[1::2]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    else:
                        logger.warning(f"Unexpected coordinate format: {coordinates}")
                        continue
                        
                    # Convert to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Ensure coordinates are within image bounds
                    height, width = image_np.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    
                    # Check if the bounding box is valid
                    if x1 >= x2 or y1 >= y2:
                        logger.warning(f"Invalid bounding box dimensions: ({x1}, {y1}, {x2}, {y2})")
                        continue
                        
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing coordinates {coordinates}: {str(e)}")
                    continue
                
                # Crop the image
                try:
                    cropped = image_np[y1-20:y2+20, x1-20:x2+20]
                    if cropped.size == 0:
                        logger.warning(f"Cropped image is empty for bbox {i+1}")
                        continue
                        
                    # Convert back to RGB for PIL
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    receipt = Image.fromarray(cropped_rgb)
                    receipts.append(receipt)
                    logger.debug(f"Successfully extracted receipt {i+1} with dimensions {receipt.size}")
                except Exception as e:
                    logger.error(f"Error cropping image for bbox {i+1}: {str(e)}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error processing bounding box {i+1}: {str(e)}")
                continue
        
        logger.info(f"Successfully extracted {len(receipts)} receipts from {image_path}")
        return receipts
        
    except Exception as e:
        logger.error(f"Error extracting receipts from {image_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        return []

def create_qwen_vl_request(model, image_base64, prompt, system_prompt=None):
    """
    Create a request for the Qwen VL model.
    
    Args:
        model (str): The model to use.
        image_base64 (str): The base64-encoded image.
        prompt (str): The prompt to send to the model.
        system_prompt (str, optional): The system prompt to use.
        
    Returns:
        dict: The request dictionary.
    """
    try:
        request = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        }
        
        if system_prompt:
            request["messages"].insert(0, {
                "role": "system",
                "content": system_prompt
            })
            
        logger.debug(f"Created request for model {model}")
        return request
    except Exception as e:
        logger.error(f"Error creating request: {str(e)}")
        raise

def get_bounding_boxes_from_response(response_content):
    """
    Extract bounding boxes from the model response.
    
    Args:
        response_content (str): The response content from the model.
        
    Returns:
        list: A list of bounding boxes.
    """
    try:
        logger.debug("Extracting bounding boxes from response")
        
        # Try to find JSON in the response
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_content)
        if json_match:
            json_str = json_match.group(1)
            logger.debug("Found JSON in markdown code block")
        else:
            # If no JSON code block, try to parse the entire response
            json_str = response_content
            logger.debug("No JSON code block found, trying to parse entire response")
        
        # Clean up the JSON string
        json_str = json_str.strip()
        
        # Parse the JSON
        try:
            data = json.loads(json_str)
            
            # Handle different JSON structures
            if isinstance(data, list):
                logger.debug(f"Found {len(data)} bounding boxes in list format")
                return data
            elif isinstance(data, dict):
                if "bounding_boxes" in data:
                    logger.debug(f"Found {len(data['bounding_boxes'])} bounding boxes in dict.bounding_boxes format")
                    return data["bounding_boxes"]
                elif "receipts" in data:
                    logger.debug(f"Found {len(data['receipts'])} bounding boxes in dict.receipts format")
                    return data["receipts"]
                else:
                    # Try to find any list in the dictionary that might contain bounding boxes
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            logger.debug(f"Found potential bounding boxes in dict.{key} format")
                            return value
                    
                    logger.warning("No recognizable bounding box structure found in JSON")
                    return []
            else:
                logger.warning("Response is not a valid JSON object or array")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            logger.debug(f"JSON string that failed to parse: {json_str}")
            return []
            
    except Exception as e:
        logger.error(f"Error extracting bounding boxes: {str(e)}")
        logger.debug(traceback.format_exc())
        return []
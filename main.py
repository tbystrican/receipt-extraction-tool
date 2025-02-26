import argparse
import os
import sys
import logging
from openai import OpenAI
from PIL import Image
import base64
import json
import extract_receipts as extract_receipts
from pdf2img import PDFtoPNGConverter


def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('receipt_extraction.log')
        ]
    )
    return logging.getLogger('receipt_extractor')


def main():
    logger = setup_logging()
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Extract receipts from PDF pages.")
    parser.add_argument("--model", default="qwen-2.5-vl", help="Model to use for receipt extraction")
    parser.add_argument("--input_folder", default="input", help="Input folder for the PDF files")
    parser.add_argument("--output_pages", default="output/pages", help="Output folder for the extracted pages")
    parser.add_argument("--output_receipts", default="output/receipts", help="Output folder for the extracted receipts")
    parser.add_argument("--extract_pages_disabled", action="store_true", help="Disable extracting pages from the PDF file (in case the pages are already extracted)")
    parser.add_argument("--list_models", action="store_true", help="List available models")   
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        help="Set the logging level")
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Update log level if specified
    if args.log_level != "INFO":
        logger.setLevel(getattr(logging, args.log_level))

    # Create output directories if they don't exist
    for directory in [args.output_pages, args.output_receipts]:
        os.makedirs(directory, exist_ok=True)

    try:
        # Initialize Venice.ai client
        api_key = os.getenv("LLM_VENICE_KEY")
        if not api_key:
            logger.error("LLM_VENICE_KEY environment variable not set")
            return 1
            
        client = OpenAI(
            base_url="https://api.venice.ai/api/v1",
            api_key=api_key,
            default_headers={"x-venice-version": "2024-10-23"}
        )

        # List available models
        if args.list_models:
            try:
                venice_models = client.models.list()
                logger.info("Available Venice.ai models:")
                logger.info([model.id for model in venice_models.data])
                return 0
            except Exception as e:
                logger.error(f"Failed to list models: {str(e)}")
                return 1

        # Define input and output folders
        input_folder = args.input_folder
        output_pages = args.output_pages
        output_receipts = args.output_receipts

        # Check if input folder exists
        if not os.path.exists(input_folder):
            logger.error(f"Input folder '{input_folder}' does not exist")
            return 1

        # Extract pages from each PDF in the input folder
        if not args.extract_pages_disabled:
            pdf_files = [f for f in os.listdir(input_folder) if f.endswith(".pdf")]
            if not pdf_files:
                logger.warning(f"No PDF files found in {input_folder}")
            
            for pdf_filename in pdf_files:
                pdf_path = os.path.join(input_folder, pdf_filename)
                logger.info(f"Extracting pages from {pdf_filename}...")
                try:
                    converter = PDFtoPNGConverter(pdf_path, output_pages, dpi=300, image_format="png", quality=94, contrast_factor=2)
                    converter.convert_all_pages_to_png()
                except Exception as e:
                    logger.error(f"Error converting PDF {pdf_filename}: {str(e)}")

        # Check if output_pages folder exists and has images
        if not os.path.exists(output_pages):
            logger.error(f"Output pages folder '{output_pages}' does not exist")
            return 1
            
        image_files = [f for f in os.listdir(output_pages) if f.endswith((".png", ".jpg", ".jpeg"))]
        if not image_files:
            logger.warning(f"No image files found in {output_pages}")
            return 0

        # Iterate over each image in the output_folder
        for filename in image_files:
            logger.info(f"Extracting receipts from page {filename}...")
            image_path = os.path.join(output_pages, filename)

            try:
                # Load your image as base64
                with open(image_path, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode()

                # Create the request
                request = extract_receipts.create_qwen_vl_request(
                    model=args.model,
                    image_base64=image_base64,
                    prompt="What is in this image?",
                    system_prompt="""Detect receipts on the page, there may be multiple. I need you to provide bounding box coordinates for each detected receipt.
                    For each detected receipt, provide the precise bounding box coordinates that encompass the entire receipt, ensuring the bounding box is also rotated if necessary to perfectly fit around the receipt, regardless of its orientation                            
                    
                    Output:
                    Must only provide structured json with bounding boxes data! """
                )

                # Send the request to the API and get the response
                response = client.chat.completions.create(**request)
                
                # Extract bounding boxes from the response
                logger.info(f"Extracting bounding boxes from response for {filename}...")
                response_content = response.choices[0].message.content
                bounding_boxes = extract_receipts.get_bounding_boxes_from_response(response_content)

                # Check if bounding_boxes is not empty
                if bounding_boxes:
                    # Extract receipts using the bounding boxes
                    receipts = extract_receipts.extract_receipts(image_path, bounding_boxes)

                    # Save the receipts
                    logger.info(f"Saving {len(receipts)} receipts from {filename}...")
                    for i, receipt in enumerate(receipts):
                        receipt_path = os.path.join(output_receipts, f"receipt_{filename}_{i+1}.png")
                        receipt.save(receipt_path)
                        logger.info(f"Saved receipt {i+1} to {receipt_path}")
                else:
                    logger.warning(f"No bounding boxes found in the response for {filename}.")
                    
            except FileNotFoundError:
                logger.error(f"Image file not found: {image_path}")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response for {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                
        return 0
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())




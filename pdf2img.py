import os
import logging
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
import numpy as np
import traceback

# Set up logger
logger = logging.getLogger('receipt_extractor.pdf2img')

class PDFtoPNGConverter:
    """
    A class to convert PDF pages to PNG images with various enhancement options.
    """
    
    def __init__(self, pdf_path, output_folder, dpi=300, image_format="png", quality=95, contrast_factor=1.0):
        """
        Initialize the PDF to PNG converter.
        
        Args:
            pdf_path (str): Path to the PDF file.
            output_folder (str): Folder to save the output images.
            dpi (int): DPI for the output images.
            image_format (str): Format for the output images (png, jpg, etc.).
            quality (int): Quality for the output images (0-100).
            contrast_factor (float): Contrast enhancement factor.
        """
        self.pdf_path = pdf_path
        self.output_folder = output_folder
        self.dpi = dpi
        self.image_format = image_format.lower()
        self.quality = quality
        self.contrast_factor = contrast_factor
        
        # Validate inputs
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        if not os.path.exists(output_folder):
            logger.info(f"Creating output folder: {output_folder}")
            try:
                os.makedirs(output_folder, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create output folder: {str(e)}")
                raise
                
        logger.info(f"Initialized PDF converter for {pdf_path} with DPI={dpi}, format={image_format}, quality={quality}")

    def convert_page_to_png(self, page_num):
        """
        Convert a specific page of the PDF to a PNG image.
        
        Args:
            page_num (int): The page number to convert (0-based).
            
        Returns:
            str: Path to the saved image file.
        """
        try:
            # Extract the PDF filename without extension
            pdf_filename = os.path.splitext(os.path.basename(self.pdf_path))[0]
            output_filename = f"{pdf_filename}_page_{page_num + 1}.{self.image_format}"
            output_path = os.path.join(self.output_folder, output_filename)
            
            logger.info(f"Converting page {page_num + 1} to {self.image_format}")
            
            # Open the PDF file
            try:
                pdf_document = fitz.open(self.pdf_path)
            except Exception as e:
                logger.error(f"Failed to open PDF file: {str(e)}")
                return None
                
            # Check if page number is valid
            if page_num < 0 or page_num >= len(pdf_document):
                logger.error(f"Invalid page number {page_num + 1}. PDF has {len(pdf_document)} pages.")
                pdf_document.close()
                return None
                
            try:
                # Get the specified page
                page = pdf_document[page_num]
                
                # Calculate the zoom factor based on DPI (default is 72 DPI)
                zoom_factor = self.dpi / 72
                
                # Create a matrix for rendering at the specified DPI
                matrix = fitz.Matrix(zoom_factor, zoom_factor)
                
                # Render the page to a pixmap
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                
                # Convert the pixmap to a PIL Image
                img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                
                # Apply contrast enhancement if needed
                if self.contrast_factor != 1.0:
                    logger.debug(f"Applying contrast enhancement with factor {self.contrast_factor}")
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(self.contrast_factor)
                
                # Save the image
                if self.image_format == "jpg" or self.image_format == "jpeg":
                    img.save(output_path, format="JPEG", quality=self.quality)
                elif self.image_format == "png":
                    img.save(output_path, format="PNG", quality=self.quality)
                else:
                    img.save(output_path)
                    
                logger.info(f"Saved page {page_num + 1} to {output_path}")
                return output_path
                
            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                logger.debug(traceback.format_exc())
                return None
                
            finally:
                # Close the PDF document
                pdf_document.close()
                
        except Exception as e:
            logger.error(f"Unexpected error converting page {page_num + 1}: {str(e)}")
            logger.debug(traceback.format_exc())
            return None

    def convert_all_pages_to_png(self):
        """
        Convert all pages of the PDF to PNG images.
        
        Returns:
            list: List of paths to the saved image files.
        """
        try:
            logger.info(f"Converting all pages in {self.pdf_path} to {self.image_format}")
            
            # Open the PDF file
            try:
                pdf_document = fitz.open(self.pdf_path)
            except Exception as e:
                logger.error(f"Failed to open PDF file: {str(e)}")
                return []
                
            try:
                # Get the number of pages
                num_pages = len(pdf_document)
                logger.info(f"PDF has {num_pages} pages")
                
                # Convert each page
                output_paths = []
                for page_num in range(num_pages):
                    output_path = self.convert_page_to_png(page_num)
                    if output_path:
                        output_paths.append(output_path)
                        
                logger.info(f"Successfully converted {len(output_paths)} out of {num_pages} pages")
                return output_paths
                
            except Exception as e:
                logger.error(f"Error converting PDF pages: {str(e)}")
                logger.debug(traceback.format_exc())
                return []
                
            finally:
                # Close the PDF document
                pdf_document.close()
                
        except Exception as e:
            logger.error(f"Unexpected error converting PDF: {str(e)}")
            logger.debug(traceback.format_exc())
            return []


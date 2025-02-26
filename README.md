# Receipt Extraction Tool
A Python tool for extracting receipts from PDF documents using computer vision and AI.

## Overview
This tool automates the process of identifying and extracting receipts from scanned PDF documents. It works in two main stages:

1. Converting PDF pages to high-quality images
2. Using Venice.AI API vision model to detect and extract individual receipts from these images

## Features
- PDF to image conversion with customizable DPI and quality settings
- AI-powered receipt detection using vision-language models
- Automatic extraction of multiple receipts from a single page
- Comprehensive error handling and logging
- Support for various image formats and enhancement options


## Requirements
- Python 3.7+
- Venice.ai API key (for Venice.ai API access)
- Required Python packages (see requirements.txt)


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/receipt-extraction-tool.git
    cd receipt-extraction-tool
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your API key:
    ```bash
    export LLM_VENICE_KEY=your_api_key_here
    ``` 


## Usage

### Basic Usage

1. Place your PDF files in the `input` directory

2. Run the extraction script:
    ```bash
    python main.py
    ```

3. Extracted receipts will be saved in the `output/receipts` directory


### Command Line Options

The script supports several command line options to customize the extraction process:

```
python main.py [OPTIONS]

Options:
  --model TEXT            Model to use for receipt extraction (default: qwen-2.5-vl)
  --input_folder TEXT     Input folder for the PDF files (default: input)
  --output_pages TEXT     Output folder for the extracted pages (default: output/pages)
  --output_receipts TEXT  Output folder for the extracted receipts (default: output/receipts)
  --extract_pages_disabled  Skip PDF to image conversion step
  --list_models           List available AI models
  --log_level TEXT        Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
```

## How It Works

1. **PDF Processing**: The tool converts each page of the PDF documents to high-resolution PNG images using PyMuPDF.
2. **Receipt Detection**: It sends these images to a vision-language model (default: qwen-2.5-vl) via the Venice.ai API to detect receipts.
3. **Extraction**: Using the bounding box coordinates returned by the AI model, it crops out each receipt and saves it as a separate image.
4. **Enhancement**: Optional contrast enhancement can be applied to improve receipt readability.

## Project Structure

```
receipt-extraction-tool/
├── main.py                  # Main script
├── extract_receipts.py      # Receipt extraction functions
├── pdf2img.py               # PDF to image conversion
├── input/                   # Input PDF files
├── output/
│   ├── pages/               # Extracted PDF pages
│   └── receipts/            # Extracted receipts
└── requirements.txt         # Project dependencies
```

## Logging

The tool provides comprehensive logging to help diagnose issues:
- Console output shows progress and errors
- A log file (`receipt_extraction.log`) contains detailed information
- Log level can be configured via command line arguments

## Troubleshooting

- **No PDFs found**: Ensure your PDF files are in the specified input directory
- **API errors**: Verify your API key is set correctly
- **Image conversion issues**: Try adjusting the DPI or contrast settings

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
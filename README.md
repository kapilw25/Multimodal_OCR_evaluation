# OCR Evaluation Framework

This repository contains an evaluation framework for OCR (Optical Character Recognition) models, specifically designed to test and compare text extraction performance on handwritten documents.

## Overview

The evaluation framework compares OCR model outputs against ground truth text extracted from XML annotations. It calculates similarity scores, word-level metrics, and processing time to provide a comprehensive assessment of OCR performance.

## Project Structure

```
project_VDR/
├── OCR_models/
│   └── got_ocr.py         # GOT-OCR-2.0 implementation
├── dataset/
│   └── cvl-database_2/    # CVL handwritten document dataset
├── results/               # Evaluation results are saved here
├── evaluation_1.py        # Main evaluation script
├── requirements.txt       # Python dependencies
├── setup_env.sh           # Script to set up virtual environment
└── README.md              # This file
```

## Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
# Create and set up the virtual environment
./setup_env.sh

# Activate the virtual environment
source venv/bin/activate
```

### 2. Install Dependencies Manually (Alternative)

If you prefer to set up the environment manually:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Evaluation

To run the evaluation framework:

```bash
# Make sure the virtual environment is activated
source venv/bin/activate

# Run the evaluation script
python evaluation_1.py
```

The script will:
1. Load the OCR model (GOT-OCR-2.0)
2. Process the test image (0052-1.tif)
3. Extract ground truth text from XML annotations
4. Calculate similarity metrics
5. Save detailed results to the `results/` directory

## Evaluation Metrics

The framework provides the following metrics:
- Text similarity score (using sequence matching)
- Word count comparison
- Processing time
- Detailed text comparison

## Results

Evaluation results are saved in the `results/` directory with the following format:
- `{image_name}_ocr_results.txt`: Contains detailed evaluation results including predicted text, ground truth text, and metrics

## Adding New OCR Models

To add a new OCR model to the evaluation framework:
1. Create a new Python module in the `OCR_models/` directory
2. Implement a function that takes an image path and returns extracted text
3. Update `evaluation_1.py` to use your new OCR model

## Requirements

The main dependencies include:
- torch
- transformers
- pillow
- safetensors
- bitsandbytes (for quantized models)
- accelerate

See `requirements.txt` for the complete list of dependencies.

## License

This project is provided for research and educational purposes.

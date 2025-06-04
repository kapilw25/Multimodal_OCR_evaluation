#!/usr/bin/env python3
"""
Simple GOT-OCR-2.0 implementation for text extraction from images
"""
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

class GOTOCR:
    def __init__(self, model_name="stepfun-ai/GOT-OCR-2.0-hf"):
        """Initialize the OCR model with the specified model name."""
        # Use MPS for M1 Mac if available, otherwise CPU
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model with appropriate settings for M1 Mac
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                device_map=self.device
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Failed to load model: {e}")
            print("The model will be loaded on first use.")
            self.model = None
            self.processor = None
            self.model_loaded = False
            self.model_name = model_name
        
    def extract_text(self, image_path, max_tokens=1024):
        """Extract text from an image file."""
        # Load and process the image
        try:
            # Load model on first use if not already loaded
            if not self.model_loaded:
                print(f"Loading model {self.model_name} on first use...")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name, 
                    torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                    device_map=self.device
                )
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model_loaded = True
                
            if isinstance(image_path, str):
                image = Image.open(image_path)
            else:
                image = image_path  # Assume it's already a PIL Image
                
            print(f"Processing image with size: {image.size}")
            
            # Process the image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate text
            generate_ids = self.model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self.processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=max_tokens,
            )
            
            # Decode the generated text
            text = self.processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            return text
            
        except Exception as e:
            print(f"Error extracting text: {e}")
            return f"Error: {str(e)}"

# Simple test function
def test_ocr(image_path):
    """Test the OCR model on a single image."""
    ocr = GOTOCR()
    text = ocr.extract_text(image_path)
    print("\nExtracted Text:")
    print("-" * 40)
    print(text)
    print("-" * 40)
    return text

def extract_text_from_image(image_path, max_tokens=1024):
    """Convenience function to extract text from an image without creating a class instance."""
    ocr = GOTOCR()
    return ocr.extract_text(image_path, max_tokens)

# Run a test if executed directly
if __name__ == "__main__":
    # Test with a sample image - replace with your own image path
    test_image = "dataset/cvl-database_2/cvl-database-1-1/testset/pages/0052-1.tif"
    test_ocr(test_image)

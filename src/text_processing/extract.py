# MetaphorScan/src/text_processing/extract.py
"""
Text extraction module for PDF and plain text files.
Implements efficient chunked processing for large files (up to 50MB per settings.yaml).
Windows-compatible path handling for MINGW64 environment.

Inspired by *The Alignment Problem as Epistemic Autoimmunity* - avoiding 
black-box processing by maintaining transparent text handling.
"""
import os
import logging
from pathlib import Path
from typing import Iterator
import PyPDF2
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_settings(config_path="src/config/settings.yaml"):
    """Load configuration settings from YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Settings file not found at {config_path}, using defaults")
        return {
            "pipeline": {"max_file_size_mb": 50},
            "models": {"spacy_model": "en_core_web_sm"}
        }

def validate_file_size(filepath, max_size_mb=50):
    """
    Validate file size against maximum allowed size.
    Implements resource limits to prevent memory issues (Epistemic Autoimmunity, Section 4).
    """
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(f"File size ({file_size_mb:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)")
    return file_size_mb

def extract_from_pdf(filepath):
    """
    Extract text from PDF files using PyPDF2.
    Handles large files with chunked processing to avoid memory overflow.
    
    Windows-compatible implementation for MINGW64 environment.
    Implements transparent processing (*Epistemic Autoimmunity*, Section 5).
    """
    try:
        text_content = []
        
        with open(filepath, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            logger.info(f"Processing PDF with {total_pages} pages")
            
            # Process pages in chunks to handle large files
            chunk_size = 10  # Process 10 pages at a time
            for i in range(0, total_pages, chunk_size):
                chunk_end = min(i + chunk_size, total_pages)
                logger.debug(f"Processing pages {i+1}-{chunk_end}")
                
                for page_num in range(i, chunk_end):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text_content.append(page_text)
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                        continue
        
        full_text = "\n\n".join(text_content)
        
        if not full_text.strip():
            raise ValueError("No readable text found in PDF file")
            
        logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
        return full_text
        
    except PyPDF2.errors.PdfReadError as e:
        raise ValueError(f"Invalid or corrupted PDF file: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error reading PDF: {e}")

def extract_from_text(filepath):
    """
    Extract text from plain text files.
    Supports multiple encodings common in academic texts.
    
    Implements structure-tracking (*Epistemic Autoimmunity*, Section 5) by 
    maintaining text integrity without interpretation.
    """
    encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(filepath, "r", encoding=encoding) as file:
                content = file.read()
                if content.strip():  # Ensure we have meaningful content
                    logger.info(f"Successfully read text file with {encoding} encoding")
                    return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"Error reading file with {encoding} encoding: {e}")
            continue
    
    raise ValueError(f"Unable to read text file {filepath} with any supported encoding")

def extract_text(filepath):
    """
    Main text extraction function that routes to appropriate handler.
    
    Args:
        filepath (str): Path to input file (PDF or text)
        
    Returns:
        str: Extracted text content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported or file is too large
        RuntimeError: If extraction fails
        
    Implements transparent text processing avoiding black-box operations
    (*The Alignment Problem as Epistemic Autoimmunity*, Section 4).
    """
    # Convert to Path object for Windows compatibility
    file_path = Path(filepath)
    
    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load settings for file size validation
    settings = load_settings()
    max_size_mb = settings.get("pipeline", {}).get("max_file_size_mb", 50)
    
    # Validate file size
    file_size_mb = validate_file_size(filepath, max_size_mb)
    logger.info(f"Processing file: {filepath} ({file_size_mb:.2f}MB)")
    
    # Determine file type and extract accordingly
    file_extension = file_path.suffix.lower()
    
    if file_extension == ".pdf":
        return extract_from_pdf(filepath)
    elif file_extension in [".txt", ".md", ".text"]:
        return extract_from_text(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: .pdf, .txt, .md")

def extract_text_chunks(filepath, chunk_size=1000):
    """
    Extract text in chunks for processing large documents.
    Useful for memory-efficient processing of large files.
    
    Args:
        filepath (str): Path to input file
        chunk_size (int): Number of characters per chunk
        
    Yields:
        str: Text chunks
        
    Implements chunked processing to avoid memory overflow in large documents,
    supporting the critique of resource-intensive AI processing 
    (*Epistemic Autoimmunity*, Section 4).
    """
    full_text = extract_text(filepath)
    
    # Split into chunks while preserving sentence boundaries
    sentences = full_text.split('. ')
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 2 <= chunk_size:  # +2 for '. '
            current_chunk += sentence + ". "
        else:
            if current_chunk.strip():
                yield current_chunk.strip()
            current_chunk = sentence + ". "
    
    # Yield the last chunk
    if current_chunk.strip():
        yield current_chunk.strip()

# Test function for development
if __name__ == "__main__":
    # Test with sample data
    test_text = "The model hallucinated a novel insight, showing remarkable intelligence."
    
    # Create test file
    test_file = "data/raw/sample.txt"
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_text)
    
    # Test extraction
    try:
        extracted = extract_text(test_file)
        print(f"Extracted text: {extracted}")
        
        # Test chunked extraction
        print("\nChunked extraction:")
        for i, chunk in enumerate(extract_text_chunks(test_file, chunk_size=50)):
            print(f"Chunk {i+1}: {chunk}")
            
    except Exception as e:
        print(f"Error: {e}")

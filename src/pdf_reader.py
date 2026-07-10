import re
import pdfplumber
from pypdf import PdfReader
from src.utils import log_info, log_error

def clean_text(text):
    """
    Cleans raw text extracted from PDF.
    - Normalizes spacing, line breaks, and whitespace.
    - Removes non-printable characters.
    """
    if not text:
        return ""
    # Normalize whitespaces and line breaks
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters / non-printable characters
    text = "".join(ch for ch in text if ch.isprintable() or ch == '\n' or ch == '\t')
    return text.strip()

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using pdfplumber as the primary parser,
    with pypdf as a robust fallback.
    
    Args:
        pdf_path (str or file-like object): Path to the PDF file or file object from Streamlit uploader.
        
    Returns:
        str: Cleaned text extracted from the PDF.
    """
    text = ""
    
    # Try pdfplumber first
    try:
        log_info("Attempting PDF text extraction using pdfplumber...")
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = []
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
            text = "\n".join(pages_text)
            
        if text.strip():
            log_info(f"Successfully extracted {len(text)} characters using pdfplumber.")
            return clean_text(text)
    except Exception as e:
        log_error("pdfplumber extraction failed, trying fallback...", e)

    # Fallback to pypdf
    try:
        log_info("Attempting PDF text extraction using pypdf fallback...")
        reader = PdfReader(pdf_path)
        pages_text = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
        text = "\n".join(pages_text)
        
        if text.strip():
            log_info(f"Successfully extracted {len(text)} characters using pypdf.")
            return clean_text(text)
    except Exception as e:
        log_error("pypdf extraction failed", e)
        
    if not text.strip():
        log_error("Could not extract any text from the PDF. The file may be empty or image-only (scanned).")
        raise ValueError("Failed to extract readable text from PDF. It might be scanned or corrupted.")

    return clean_text(text)

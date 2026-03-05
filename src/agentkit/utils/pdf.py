"""
PDF processing utilities for AgentKit.
"""

from __future__ import annotations

import os
from typing import List, Optional

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

from agentkit.core.tools import tool


@tool
def read_pdf(file_path: str, max_pages: Optional[int] = None) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        max_pages: Optional maximum number of pages to read
    """
    if not HAS_PYMUPDF:
        return "Error: PyMuPDF not installed. pip install pymupdf"
        
    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"
        
    try:
        doc = fitz.open(file_path)
        text_parts = []
        
        num_pages = len(doc)
        if max_pages:
            num_pages = min(num_pages, max_pages)
            
        for i in range(num_pages):
            page = doc.load_page(i)
            text_parts.append(f"--- Page {i+1} ---\n" + page.get_text())
            
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

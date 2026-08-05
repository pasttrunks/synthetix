import os
import re
from typing import Dict, Any, Tuple

def filter_qualifying_prose(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract continuous qualifying long-form prose.
    Strips short standalone lines (< 4 words), page headers, and non-prose metadata.
    """
    lines = text.split("\n")
    qualifying_paragraphs = []
    total_words = 0
    stripped_lines = 0

    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        
        words = cleaned.split()
        # Filter out short isolated header/footer lines unless part of continuous sentence
        if len(words) < 4 and not cleaned.endswith((".", "?", "!")):
            stripped_lines += 1
            continue

        qualifying_paragraphs.append(cleaned)
        total_words += len(words)

    filtered_text = "\n\n".join(qualifying_paragraphs)
    stats = {
        "raw_character_count": len(text),
        "qualifying_character_count": len(filtered_text),
        "qualifying_word_count": total_words,
        "stripped_non_prose_lines": stripped_lines
    }
    return filtered_text, stats

def extract_text_from_bytes(file_bytes: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
    """Extract plain text from file bytes based on filename extension."""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in (".txt", ".md", ".jsonl"):
        text = file_bytes.decode("utf-8", errors="ignore")
    elif ext == ".docx":
        try:
            import docx
            import io
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            # Fallback simple text extraction from docx xml stream
            text = re.sub(r'<[^>]+>', ' ', file_bytes.decode("utf-8", errors="ignore"))
    elif ext == ".pdf":
        try:
            import pypdf
            import io
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            # Fallback regex string extraction for pdf
            text = " ".join(re.findall(r'[a-zA-Z0-9\s.,!?\'"-]{4,}', file_bytes.decode("latin-1", errors="ignore")))
    else:
        text = file_bytes.decode("utf-8", errors="ignore")

    return filter_qualifying_prose(text)

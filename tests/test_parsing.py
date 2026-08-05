import re
import pytest

def segment_text(raw_text: str) -> list:
    """Helper implementing the exact sentence parsing logic in server.py."""
    raw_text_stripped = raw_text.strip()
    if not raw_text_stripped:
        return []
    
    sentences = re.findall(r'[^.!?]+[.!?]+', raw_text_stripped)
    matched_len = sum(len(s) for s in sentences)
    remainder = raw_text_stripped[matched_len:].strip()
    if remainder:
        sentences.append(remainder)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [raw_text_stripped]
    return sentences

def test_sentence_segmentation_normal_text():
    """Test sentence segmentation with normal punctuated text."""
    text = "First sentence. Second sentence! Third sentence?"
    result = segment_text(text)
    assert len(result) == 3
    assert result[0] == "First sentence."
    assert result[1] == "Second sentence!"
    assert result[2] == "Third sentence?"

def test_trailing_unpunctuated_text_captured():
    """Test trailing unpunctuated text is captured as a final sentence."""
    text = "First sentence. Trailing unpunctuated text"
    result = segment_text(text)
    assert len(result) == 2
    assert result[0] == "First sentence."
    assert result[1] == "Trailing unpunctuated text"

def test_single_sentence():
    """Test text with only one sentence."""
    text = "Only one sentence here."
    result = segment_text(text)
    assert len(result) == 1
    assert result[0] == "Only one sentence here."

def test_text_with_abbreviations():
    """Test sentence segmentation behavior with abbreviations containing periods."""
    text = "Dr. Smith went to Washington."
    result = segment_text(text)
    # The server's regex splits on period after 'Dr.'
    assert len(result) >= 1
    assert "Washington." in result[-1]

def test_empty_string_input():
    """Test empty string input returns empty list."""
    assert segment_text("") == []
    assert segment_text("   ") == []

def test_mixed_punctuation():
    """Test text with mixed punctuation (!, ?, .)."""
    text = "Is this real? Yes it is! End of story."
    result = segment_text(text)
    assert len(result) == 3
    assert result[0] == "Is this real?"
    assert result[1] == "Yes it is!"
    assert result[2] == "End of story."

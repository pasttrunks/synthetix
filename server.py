import os
import re
import math
import requests
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "ai_detector.html")

app = FastAPI(title="Synthetix AI Detection Engine")

@app.get("/")
def serve_index():
    if os.path.exists(HTML_PATH):
        return FileResponse(HTML_PATH)
    raise HTTPException(status_code=404, detail="ai_detector.html not found")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "model_name": MODEL_NAME if MODEL_LOADED else "None"
    }


MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta"
MODEL_REVISION = "main"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:latest"
MIN_TEXT_LENGTH = 150
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

print(f"Loading RoBERTa AI detector model: {MODEL_NAME} (revision: {MODEL_REVISION})...")

def get_ai_label_index(model_config) -> int:
    if hasattr(model_config, "id2label") and model_config.id2label:
        # First pass: check all labels for explicit AI / ChatGPT terms or label_1
        for idx, label in model_config.id2label.items():
            label_lower = str(label).lower()
            if any(term in label_lower for term in ["chatgpt", "fake", "ai", "generated", "synthetic", "label_1"]):
                return int(idx)
        # Second pass: fall back to human-opposite logic if no explicit AI label is found
        for idx, label in model_config.id2label.items():
            label_lower = str(label).lower()
            if any(term in label_lower for term in ["real", "human", "label_0"]):
                return 1 if int(idx) == 0 else 0
    return 0

AI_LABEL_INDEX = 1


try:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, local_files_only=True)
        loaded_from_cache = True
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, local_files_only=False)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, local_files_only=False)
        loaded_from_cache = False

    model.eval()
    MODEL_LOADED = True
    AI_LABEL_INDEX = get_ai_label_index(model.config)
    source_str = "local cache" if loaded_from_cache else "remote download"
    print(f"RoBERTa AI detector model successfully loaded from {source_str}! AI label index: {AI_LABEL_INDEX}")
except Exception as e:
    print(f"Offline startup check error: Could not load RoBERTa model ({MODEL_NAME} @ {MODEL_REVISION}): {e}")
    MODEL_LOADED = False


AI_PHRASES = [
    "in conclusion", "furthermore", "moreover", "it is important to note",
    "delve", "plethora", "crucial role", "seamless", "foster",
    "in today's digital world", "tapestry", "testament", "vibrant",
    "it is evident that", "in addition to this", "ultimately"
]

def is_primarily_english(text: str) -> bool:
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return True
    latin_count = sum(1 for c in alpha_chars if ord(c) < 128 or (0x00C0 <= ord(c) <= 0x024F))
    ratio = latin_count / len(alpha_chars)
    return ratio >= 0.85

class TextPayload(BaseModel):
    text: str

class SentenceScore(BaseModel):
    sentence: str
    length: int
    ai_score: float = Field(ge=0, le=100)
    is_flagged: bool
    label: str
    is_suspicious: bool = False

class ChunkScore(BaseModel):
    chunk_index: int
    word_count: int
    ai_score: float = Field(ge=0, le=100)

class AnalysisResult(BaseModel):
    overall_ai_score: Optional[float] = Field(default=None, ge=0, le=100)
    burstiness_cv: float
    predictability_index: float
    phrase_count: int
    model_name: str
    analysis_method: Optional[str] = None
    ollama_active: bool
    sentence_scores: List[SentenceScore]
    chunk_scores: Optional[List[ChunkScore]] = None
    text_coverage: Optional[float] = Field(default=None, ge=0, le=100)
    language_warning: Optional[str] = None
    message: Optional[str] = None

def check_ollama_alive() -> bool:
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=3)
        if res.status_code == 200:
            models = [m.get("name") for m in res.json().get("models", [])]
            return OLLAMA_MODEL in models
    except Exception:
        pass
    return False

def calculate_burstiness(sentence_lengths: List[int]) -> float:
    if not sentence_lengths or len(sentence_lengths) < 3:
        return 0.0
    arr = np.array(sentence_lengths, dtype=float)
    mean = np.mean(arr)
    if mean == 0:
        return 0.0
    std = np.std(arr)
    return float(std / mean)

def score_text_with_transformer(text: str) -> float:
    if not MODEL_LOADED:
        return 0.0
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()
        if logits.dim() == 0 or (hasattr(logits, "__len__") and len(logits.shape) == 0):
            ai_prob = torch.sigmoid(logits).item() * 100.0
        elif hasattr(logits, "__len__") and len(logits) == 1:
            ai_prob = torch.sigmoid(logits[0]).item() * 100.0
        else:
            probs = torch.softmax(logits, dim=-1).tolist()
            if isinstance(probs, float):
                probs = [probs]
            ai_prob = probs[AI_LABEL_INDEX] * 100.0
    return max(0.0, min(100.0, float(ai_prob)))

def score_text_with_ollama(text: str) -> Optional[float]:
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"Analyze this text and rate AI probability from 0 to 100. Output ONLY format {{\\\"score\\\": 85}}.\n\nText: \"{text[:800]}\"",
            "format": "json",
            "stream": False,
            "options": {
                "num_predict": 25,
                "temperature": 0.1
            }
        }
        res = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if res.status_code == 200:
            data = res.json()
            response_text = data.get("response", "").strip()
            clean_text = re.sub(r'^```(?:json)?\s*', '', response_text, flags=re.IGNORECASE)
            clean_text = re.sub(r'\s*```$', '', clean_text)
            match = re.search(r'\{.*?\}', clean_text, re.DOTALL)
            if match:
                import json
                parsed = json.loads(match.group(0))
                if "score" in parsed and isinstance(parsed["score"], (int, float)):
                    score_val = max(0.0, min(100.0, float(parsed["score"])))
                    print(f"Ollama Llama3 returned AI score: {score_val}%")
                    return score_val
    except Exception as e:
        print(f"Ollama query failed or timed out: {e}")
    return None

@app.get("/api/health")
def health():
    ollama_ok = check_ollama_alive()
    return {
        "status": "online",
        "deberta_loaded": MODEL_LOADED,
        "ollama_active": ollama_ok,
        "ollama_model": OLLAMA_MODEL if ollama_ok else None,
        "active_engine": f"DeBERTa-v3 Transformer ({MODEL_REVISION})" if MODEL_LOADED else "Unavailable"
    }

@app.post("/api/analyze", response_model=AnalysisResult)
def analyze(payload: TextPayload):
    if len(payload.text) > 50000:
        raise HTTPException(status_code=413, detail="Text length exceeds 50,000 character limit.")

    raw_text = payload.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")

    ollama_ok = check_ollama_alive()
    active_model_desc = f"DeBERTa-v3 Transformer ({MODEL_REVISION})" if MODEL_LOADED else "Unavailable"

    if len(raw_text) < MIN_TEXT_LENGTH:
        return AnalysisResult(
            overall_ai_score=None,
            burstiness_cv=0.0,
            predictability_index=0.0,
            phrase_count=0,
            model_name=active_model_desc,
            analysis_method="insufficient_text",
            ollama_active=ollama_ok,
            sentence_scores=[],
            chunk_scores=[],
            text_coverage=None,
            language_warning=None,
            message=f"Text length ({len(raw_text)} characters) is below the minimum required length of {MIN_TEXT_LENGTH} characters for analysis."
        )

    lang_warning = None
    if not is_primarily_english(raw_text):
        lang_warning = "Text may not be in English. Detection accuracy is not validated for non-English text."

    sentences = re.findall(r'[^.!?]+[.!?]+', raw_text)
    matched_len = sum(len(s) for s in sentences)
    remainder = raw_text[matched_len:].strip()
    if remainder:
        sentences.append(remainder)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [raw_text]

    sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    num_sentences = len(sentences)
    
    if num_sentences >= 3:
        cv = calculate_burstiness(sentence_lengths)
        predictability_idx = round(min(100.0, max(0.0, 100.0 - (cv * 120.0))), 1)
    else:
        cv = 0.0
        predictability_idx = 0.0

    lower_text = raw_text.lower()
    matched_phrases = [p for p in AI_PHRASES if p in lower_text]
    phrase_count = len(matched_phrases)

    sentence_scores = []
    sentence_ai_probs = []

    for s, l in zip(sentences, sentence_lengths):
        if MODEL_LOADED:
            s_score = score_text_with_transformer(s)
        else:
            s_score = 0.0
        
        s_score_clamped = max(0.0, min(100.0, s_score))
        sentence_ai_probs.append(s_score_clamped)
        s_lower = s.lower()
        has_phrase = any(p in s_lower for p in AI_PHRASES)
        is_flagged = (s_score_clamped > 65.0 and has_phrase) or s_score_clamped > 75.0
        label_text = "Flagged" if is_flagged else "Not flagged"

        sentence_scores.append(SentenceScore(
            sentence=s,
            length=l,
            ai_score=round(s_score_clamped, 1),
            is_flagged=is_flagged,
            label=label_text,
            is_suspicious=is_flagged
        ))

    words = raw_text.split()
    total_words = len(words)

    chunk_scores: List[ChunkScore] = []
    overall_score: Optional[float] = None
    text_coverage: Optional[float] = None
    method_desc: Optional[str] = None

    if MODEL_LOADED:
        if total_words <= CHUNK_SIZE:
            deberta_score = score_text_with_transformer(raw_text)
            clamped_score = round(max(0.0, min(100.0, deberta_score)), 1)
            overall_score = clamped_score
            chunk_scores = [ChunkScore(chunk_index=0, word_count=total_words, ai_score=clamped_score)]
            text_coverage = 100.0
            method_desc = "DeBERTa-v3 Transformer"
        else:
            step = CHUNK_SIZE - CHUNK_OVERLAP
            words_covered_indices = set()
            chunk_idx = 0

            for i in range(0, total_words, step):
                chunk_words = words[i : i + CHUNK_SIZE]
                chunk_str = " ".join(chunk_words)
                w_count = len(chunk_words)
                
                c_score = score_text_with_transformer(chunk_str)
                c_score_clamped = round(max(0.0, min(100.0, c_score)), 1)
                
                chunk_scores.append(ChunkScore(
                    chunk_index=chunk_idx,
                    word_count=w_count,
                    ai_score=c_score_clamped
                ))
                
                for idx_w in range(i, i + w_count):
                    words_covered_indices.add(idx_w)
                
                chunk_idx += 1
                if i + CHUNK_SIZE >= total_words:
                    break
            
            total_weight = sum(cs.word_count for cs in chunk_scores)
            if total_weight > 0:
                weighted_sum = sum(cs.ai_score * cs.word_count for cs in chunk_scores)
                overall_score = round(weighted_sum / total_weight, 1)
            else:
                overall_score = 0.0

            text_coverage = round((len(words_covered_indices) / total_words) * 100.0, 1) if total_words > 0 else 100.0
            method_desc = f"DeBERTa-v3 Transformer ({len(chunk_scores)} Chunks, Weighted Avg)"
    else:
        overall_score = None
        method_desc = "unavailable"
        chunk_scores = []
        text_coverage = None

    return AnalysisResult(
        overall_ai_score=overall_score,
        burstiness_cv=round(cv, 3),
        predictability_index=predictability_idx,
        phrase_count=phrase_count,
        model_name=active_model_desc,
        analysis_method=method_desc,
        ollama_active=ollama_ok,
        sentence_scores=sentence_scores,
        chunk_scores=chunk_scores,
        text_coverage=text_coverage,
        language_warning=lang_warning
    )

if __name__ == "__main__":
    print("Starting Synthetix AI Detector Engine on http://localhost:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)


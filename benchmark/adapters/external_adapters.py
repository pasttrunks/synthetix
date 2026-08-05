import json
from typing import Dict, Any, List
from benchmark.corpus_registry import normalize_sample, validate_sample_schema

def adapt_raid_sample(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert RAID benchmark record into Synthetix corpus format."""
    text = raw.get("generation", raw.get("text", ""))
    is_ai = raw.get("model", "human") != "human"
    group_id = f"raid_{raw.get('domain', 'general')}_{raw.get('attack', 'none')}"
    
    sample = {
        "text": text,
        "label": 1 if is_ai else 0,
        "source_group_id": group_id,
        "domain": raw.get("domain", "academic"),
        "language": "en",
        "provenance": {
            "source_name": "RAID Benchmark Dataset",
            "dataset_license": "CC-BY-4.0",
            "collection_method": "RAID release",
            "original_model": raw.get("model", "human"),
            "attack_type": raw.get("attack", "none")
        }
    }
    return normalize_sample(sample)

def adapt_semeval_sample(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert SemEval-2024 Task 8 record into Synthetix corpus format."""
    text = raw.get("text", "")
    label_val = raw.get("label", 0)
    model_name = raw.get("model", "human" if label_val == 0 else "chatgpt")
    
    sample = {
        "text": text,
        "label": label_val,
        "source_group_id": f"semeval_{model_name}",
        "domain": raw.get("domain", "general"),
        "language": raw.get("language", "en"),
        "provenance": {
            "source_name": "SemEval-2024 Task 8",
            "dataset_license": "Research Only",
            "original_model": model_name
        }
    }
    return normalize_sample(sample)

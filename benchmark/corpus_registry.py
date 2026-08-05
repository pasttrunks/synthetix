import hashlib
import json
from typing import Dict, Any, Tuple, Optional, List

REQUIRED_SCHEMA_KEYS = {"text", "label", "source_group_id", "provenance"}

def compute_content_hash(text: str) -> str:
    """Compute deterministic SHA256 content hash for a text sample."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

def validate_sample_schema(sample: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate sample dictionary against Synthetix dataset schema standards."""
    missing_keys = REQUIRED_SCHEMA_KEYS - set(sample.keys())
    if missing_keys:
        return False, f"Missing required fields: {', '.join(sorted(missing_keys))}"

    if sample["label"] not in ("human", "ai", 0, 1):
        return False, f"Invalid label '{sample['label']}'. Must be 'human', 'ai', 0, or 1."

    if not isinstance(sample["text"], str) or not sample["text"].strip():
        return False, "Text field must be a non-empty string."

    if not isinstance(sample["source_group_id"], str) or not sample["source_group_id"].strip():
        return False, "source_group_id must be a non-empty string."

    if not isinstance(sample["provenance"], dict):
        return False, "provenance field must be a dictionary."

    provenance = sample["provenance"]
    if "source_name" not in provenance or "dataset_license" not in provenance:
        return False, "provenance dict must include 'source_name' and 'dataset_license'."

    return True, None

def normalize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure standard fields including sample_id, content_hash, and normalized label."""
    norm = dict(sample)
    raw_label = str(norm["label"]).lower()
    norm["label"] = 1 if raw_label in ("ai", "1", "chatgpt") else 0
    
    content_hash = compute_content_hash(norm["text"])
    norm["sample_id"] = norm.get("sample_id", content_hash[:16])
    norm["content_hash"] = content_hash
    return norm

def load_and_validate_corpus(file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load JSONL corpus file, validate entries, and return metadata summary."""
    valid_samples = []
    errors = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                is_valid, err = validate_sample_schema(item)
                if is_valid:
                    valid_samples.append(normalize_sample(item))
                else:
                    errors.append(f"Line {line_num}: {err}")
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON ({e})")

    group_ids = set(s["source_group_id"] for s in valid_samples)
    summary = {
        "total_read": len(valid_samples) + len(errors),
        "valid_count": len(valid_samples),
        "error_count": len(errors),
        "unique_groups": len(group_ids),
        "errors": errors
    }
    return valid_samples, summary

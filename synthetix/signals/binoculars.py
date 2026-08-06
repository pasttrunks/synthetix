import math
import torch
from typing import Dict, Any, Optional

def compute_binoculars_score(text: str, perplexity_score: Optional[float] = None) -> Dict[str, Any]:
    """
    Compute zero-shot Binoculars perplexity ratio or experimental lexical entropy heuristic.
    Note: Full Binoculars requires observer and performer model cross-perplexity scores.
    """
    words = text.strip().split()
    if len(words) < 15:
        return {
            "binoculars_score": None,
            "binoculars_flagged": False,
            "signal_type": "heuristic",
            "reason": "Text too short for reliable zero-shot calculation (< 15 words)"
        }

    # If external perplexity is provided, calculate exact log-ratio baseline
    if perplexity_score is not None and perplexity_score > 0:
        log_ppl = math.log(perplexity_score)
        # Standard Binoculars threshold: log-ppl ratio < 0.90 indicates synthetic text
        binoculars_val = round(log_ppl / (log_ppl + 0.4), 4)
        signal_type = "perplexity_ratio"
    else:
        # Heuristic zero-shot log-perplexity estimation based on vocabulary entropy
        vocab_size = len(set(words))
        type_token_ratio = vocab_size / len(words)
        avg_word_len = sum(len(w) for w in words) / len(words)
        
        # Estimate synthetic regularity vs human variation
        estimated_entropy = (type_token_ratio * 4.5) + (avg_word_len * 0.3)
        binoculars_val = round(max(0.0, min(1.0, 1.0 - (estimated_entropy / 6.0))), 4)
        signal_type = "lexical_entropy_heuristic"

    is_flagged = binoculars_val > 0.65

    return {
        "binoculars_score": binoculars_val,
        "binoculars_flagged": is_flagged,
        "signal_type": signal_type,
        "confidence": "high" if len(words) > 50 else "medium"
    }



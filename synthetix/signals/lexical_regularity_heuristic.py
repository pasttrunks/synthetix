import math
import torch
from typing import Dict, Any, Optional

def compute_lexical_regularity_score(text: str, perplexity_score: Optional[float] = None) -> Dict[str, Any]:
    """
    Compute an experimental lexical regularity heuristic.

    This is NOT the Binoculars method (which requires observer and performer
    model cross-perplexity scores). It is a lightweight fallback that estimates
    regularity from vocabulary entropy and mean word length. Treat its output
    as a weak supplementary signal only.
    """
    words = text.strip().split()
    if len(words) < 15:
        return {
            "lexical_regularity_score": None,
            "lexical_regularity_flagged": False,
            "signal_type": "heuristic",
            "reason": "Text too short for reliable calculation (< 15 words)"
        }

    # If external perplexity is provided, calculate a log-ratio baseline
    if perplexity_score is not None and perplexity_score > 0:
        log_ppl = math.log(perplexity_score)
        regularity_val = round(log_ppl / (log_ppl + 0.4), 4)
        signal_type = "perplexity_ratio"
    else:
        # Heuristic regularity estimation based on vocabulary entropy
        vocab_size = len(set(words))
        type_token_ratio = vocab_size / len(words)
        avg_word_len = sum(len(w) for w in words) / len(words)

        estimated_entropy = (type_token_ratio * 4.5) + (avg_word_len * 0.3)
        regularity_val = round(max(0.0, min(1.0, 1.0 - (estimated_entropy / 6.0))), 4)
        signal_type = "lexical_entropy_heuristic"

    is_flagged = regularity_val > 0.65

    return {
        "lexical_regularity_score": regularity_val,
        "lexical_regularity_flagged": is_flagged,
        "signal_type": signal_type,
        "confidence": "high" if len(words) > 50 else "medium"
    }

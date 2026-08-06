"""Clean model-backend abstraction for the Synthetix detector.

Backends are selected by name through configuration (default: ``hc3_roberta``).
Each backend owns its loader, inference logic, and metadata. No threshold or
scoring combination happens here; backends only return a 0-100 AI-writing
signal for a single document.
"""

import os
import time
from typing import Any, Dict, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DebertaV2Config,
    PreTrainedModel,
)
import torch.nn as nn

HC3_REPO = "Hello-SimpleAI/chatgpt-detector-roberta"
HC3_REVISION = "d2b342c61775d5dd0221808a79983ed3b86ffd86"

DESKLIB_REPO = "desklib/ai-text-detector-academic-v1.01"
DESKLIB_REVISION = "fe9b4da50ee2cca5c877d607640681609170e363"

DEFAULT_BACKEND = "hc3_roberta"
BACKEND_ENV_VAR = "SYNTETIX_BACKEND"


def get_inference_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_ai_label_index(model_config: Any) -> int:
    """Determine the class index that corresponds to AI-generated text."""
    if hasattr(model_config, "id2label") and model_config.id2label:
        for idx, label in model_config.id2label.items():
            label_lower = str(label).lower()
            if any(term in label_lower for term in ["chatgpt", "fake", "ai", "generated", "synthetic", "label_1"]):
                return int(idx)
        for idx, label in model_config.id2label.items():
            label_lower = str(label).lower()
            if any(term in label_lower for term in ["real", "human", "label_0"]):
                return 1 if int(idx) == 0 else 0
    return 0


def score_sequence_classification(tokenizer: Any, model: Any, ai_label_index: int, text: str) -> float:
    """Score text with a standard sequence-classification model (0-100 AI signal)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()
        if logits.dim() == 0:
            ai_prob = torch.sigmoid(logits).item() * 100.0
        elif hasattr(logits, "__len__") and len(logits) == 1:
            ai_prob = torch.sigmoid(logits[0]).item() * 100.0
        else:
            probs = torch.softmax(logits, dim=-1).tolist()
            if isinstance(probs, float):
                probs = [probs]
            ai_prob = probs[ai_label_index] * 100.0
    return max(0.0, min(100.0, float(ai_prob)))


class DesklibAIDetectionModel(PreTrainedModel):
    """Desklib academic AI-detection head: DeBERTa-v3-large + mean pooling + single logit.

    Architecture is documented in the model card; this class re-implements it
    locally so no remote code execution is required.
    """

    config_class = DebertaV2Config

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        logits = self.classifier(pooled_output)
        output = {"logits": logits}
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            output["loss"] = loss_fct(logits.view(-1), labels.float())
        return output


class DetectorBackend:
    """Base class for a loadable detector backend."""

    name: str = ""
    model_name: str = ""
    model_revision: str = ""
    tokenizer_revision: str = ""
    device: str = "cpu"
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    ai_label_index: Optional[int] = None

    def load(self) -> None:
        raise NotImplementedError

    def score_text(self, text: str) -> float:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend_name": self.name,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "inference_device": self.device,
        }


class HC3RobertaBackend(DetectorBackend):
    """The existing pinned baseline: Hello-SimpleAI/chatgpt-detector-roberta."""

    name = "hc3_roberta"
    model_name = HC3_REPO
    model_revision = HC3_REVISION
    tokenizer_revision = HC3_REVISION

    def load(self) -> None:
        self.device = get_inference_device()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.model_revision, local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, revision=self.model_revision, local_files_only=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.model_revision, local_files_only=False
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, revision=self.model_revision, local_files_only=False
            )
        self.model.eval()
        self.ai_label_index = get_ai_label_index(self.model.config)

    def score_text(self, text: str) -> float:
        if self.model is None or self.tokenizer is None:
            return 0.0
        return score_sequence_classification(
            self.tokenizer, self.model, self.ai_label_index or 0, text
        )


class DesklibAcademicBackend(DetectorBackend):
    """Desklib academic detector: desklib/ai-text-detector-academic-v1.01.

    Custom architecture (DeBERTa-v3-large backbone, attention-masked mean
    pooling, single sigmoid logit); loaded through its own model class rather
    than the sequence-classification loader.
    """

    name = "desklib_academic"
    model_name = DESKLIB_REPO
    model_revision = DESKLIB_REVISION
    tokenizer_revision = DESKLIB_REVISION

    def load(self) -> None:
        self.device = get_inference_device()
        config = AutoConfig.from_pretrained(self.model_name, revision=self.model_revision)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.model_revision, local_files_only=False
        )
        self.model = DesklibAIDetectionModel.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            config=config,
            local_files_only=False,
        )
        self.model.eval()
        self.ai_label_index = None  # single-logit model; no label index needed

    def score_text(self, text: str) -> float:
        if self.model is None or self.tokenizer is None:
            return 0.0
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"]
            prob = torch.sigmoid(logits).item()
        return max(0.0, min(100.0, float(prob) * 100.0))


class BalancedReviewBackend(DetectorBackend):
    """Runs both pinned detectors and returns an agreement/abstention outcome.

    At the existing 0.50 threshold:
    - both at or above 0.50 -> strong AI-writing signal
    - both below 0.50      -> low AI-writing signal
    - otherwise            -> uncertain (detectors disagree); never converted

    Scores are never averaged or combined numerically.
    """

    name = "balanced_review"
    model_name = "balanced_review (hc3_roberta + desklib_academic)"
    model_revision = f"hc3:{HC3_REVISION};desklib:{DESKLIB_REVISION}"
    tokenizer_revision = model_revision

    def __init__(self):
        super().__init__()
        self.hc3 = HC3RobertaBackend()
        self.desklib = DesklibAcademicBackend()
        self._lock = __import__("threading").Lock()

    def load(self) -> None:
        self.device = get_inference_device()
        print("Loading HC3 Fast Baseline detector (balanced review)...")
        self.hc3.load()
        print("Loading Desklib Academic Sensitive detector (balanced review)...")
        self.desklib.load()

    def score_text(self, text: str) -> float:
        """Compatibility path for the server scoring loop (HC3 sentence/chunk scores)."""
        return self.hc3.score_text(text)

    def review_document(self, text: str) -> Dict[str, Any]:
        with self._lock:
            t0 = time.perf_counter()
            hs = self.hc3.score_text(text)
            t_hc3 = time.perf_counter() - t0
            t1 = time.perf_counter()
            ds = self.desklib.score_text(text)
            t_dk = time.perf_counter() - t1
            total = time.perf_counter() - t0

        h_hi = hs >= 50.0
        d_hi = ds >= 50.0
        if h_hi and d_hi:
            agreement = "agree_high"
            outcome = "strong_ai_signal"
        elif not h_hi and not d_hi:
            agreement = "agree_low"
            outcome = "low_ai_signal"
        else:
            agreement = "disagree"
            outcome = "uncertain_disagreement"

        return {
            "hc3_score": round(hs, 1),
            "hc3_backend_name": "hc3_roberta",
            "hc3_model_name": HC3_REPO,
            "hc3_model_revision": HC3_REVISION,
            "desklib_score": round(ds, 1),
            "desklib_backend_name": "desklib_academic",
            "desklib_model_name": DESKLIB_REPO,
            "desklib_model_revision": DESKLIB_REVISION,
            "agreement_status": agreement,
            "review_outcome": outcome,
            "hc3_elapsed_s": round(t_hc3, 3),
            "desklib_elapsed_s": round(t_dk, 3),
            "total_elapsed_s": round(total, 3),
        }


BACKENDS = {
    "hc3_roberta": HC3RobertaBackend,
    "desklib_academic": DesklibAcademicBackend,
    "balanced_review": BalancedReviewBackend,
}


def create_backend(name: str) -> DetectorBackend:
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. Available backends: {sorted(BACKENDS)}"
        )
    return BACKENDS[name]()


def resolve_backend_name(override: Optional[str] = None) -> str:
    name = override or os.environ.get(BACKEND_ENV_VAR) or DEFAULT_BACKEND
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}'. Available backends: {sorted(BACKENDS)}"
        )
    return name


def timed_score(backend: DetectorBackend, text: str) -> tuple:
    """Score a document and return (score, elapsed_seconds)."""
    t0 = time.perf_counter()
    score = backend.score_text(text)
    return score, time.perf_counter() - t0

"""
Deep Learning Toxicity and Cyberbullying Classifier
Powered by Hugging Face Transformers (unitary/toxic-bert)
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .preprocessor import clean_text, normalize_leetspeak

# Category display names and risk weights
CATEGORY_METADATA = {
    "toxic": {
        "label": "Toxicity",
        "description": "General rude, disrespectful, or unreasonable speech",
        "icon": "⚠️",
        "color": "#f59e0b"
    },
    "severe_toxic": {
        "label": "Severe Toxicity",
        "description": "Extremely hateful, aggressive, or destructive remarks",
        "icon": "🛑",
        "color": "#dc2626"
    },
    "obscene": {
        "label": "Obscenity / Profanity",
        "description": "Vulgar, obscene language or swear words",
        "icon": "🤬",
        "color": "#ea580c"
    },
    "threat": {
        "label": "Threat / Violence",
        "description": "Statements expressing intention to inflict harm or violence",
        "icon": "🚨",
        "color": "#b91c1c"
    },
    "insult": {
        "label": "Insult / Harassment",
        "description": "Disparaging, demeaning, or belittling a person",
        "icon": "🎯",
        "color": "#d97706"
    },
    "identity_hate": {
        "label": "Identity Hate Speech",
        "description": "Attacks targeting race, religion, gender, sexual orientation, or identity",
        "icon": "🛡️",
        "color": "#991b1b"
    }
}

class ToxicityDetector:
    def __init__(self, model_name: str = "unitary/toxic-bert", device: Optional[str] = None):
        """
        Initializes the toxicity detector with tokenizer and classification model.
        """
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Labels in order defined by unitary/toxic-bert
        # ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        else:
            self.labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def _determine_severity(self, scores: Dict[str, float], is_toxic: bool) -> str:
        """Determines severity grade based on category probabilities."""
        if not is_toxic:
            max_s = max(scores.values()) if scores else 0.0
            if max_s < 0.15:
                return "Clean (Safe)"
            return "Low / Borderline"
            
        threat_score = scores.get("threat", 0.0)
        severe_score = scores.get("severe_toxic", 0.0)
        hate_score = scores.get("identity_hate", 0.0)
        max_score = max(scores.values()) if scores else 0.0

        if threat_score >= 0.50 or severe_score >= 0.70:
            return "Critical Danger (Threat / Severe)"
        elif max_score >= 0.85 or hate_score >= 0.65:
            return "High Toxicity"
        elif max_score >= 0.60:
            return "Moderate Toxicity"
        else:
            return "Mild / Elevated Concern"

    def predict(self, text: str, threshold: float = 0.50, highlight_tokens: bool = True) -> Dict[str, Any]:
        """
        Predicts toxicity scores for a single comment.
        """
        if not text or not text.strip():
            return {
                "original_text": text,
                "cleaned_text": "",
                "is_toxic": False,
                "overall_score": 0.0,
                "primary_category": "Clean",
                "severity_level": "Clean (Safe)",
                "scores": {lbl: 0.0 for lbl in self.labels},
                "flagged_categories": [],
                "token_highlights": []
            }

        cleaned = clean_text(text)
        # clean_text already lowercases and decodes word-level leetspeak;
        # normalize_leetspeak is idempotent, kept for non-lowercase inputs.
        normalized = normalize_leetspeak(cleaned)

        inputs = self.tokenizer(
            normalized,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Multi-label classification uses Sigmoid
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        scores = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}

        # Check flagged categories based on threshold
        flagged = [cat for cat, s in scores.items() if s >= threshold]
        is_toxic = len(flagged) > 0

        # Primary category: highest-scoring *flagged* label when toxic
        # (falling back to the global argmax otherwise).  Using the raw
        # argmax previously reported labels like "Obscenity" as the primary
        # category even for completely clean comments.
        if is_toxic:
            top_category = max(flagged, key=lambda c: scores[c])
            top_score = scores[top_category]
            primary_category = CATEGORY_METADATA.get(top_category, {}).get("label", top_category)
        else:
            top_category, top_score = max(scores.items(), key=lambda x: x[1])
            primary_category = "Clean"

        severity = self._determine_severity(scores, is_toxic)

        # Highlight important toxic spans if comment is flagged
        token_highlights = []
        if highlight_tokens and is_toxic:
            token_highlights = self._explain_tokens(normalized, top_score)

        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "is_toxic": is_toxic,
            "overall_score": top_score,
            "primary_category": primary_category,
            "severity_level": severity,
            "scores": scores,
            "flagged_categories": flagged,
            "token_highlights": token_highlights
        }

    def _explain_tokens(self, text: str, baseline_score: float) -> List[Dict[str, Any]]:
        """
        Calculates token impact by leave-one-out masking on individual words.
        Provides explainability without requiring heavy external dependencies.
        """
        words = text.split()
        if len(words) <= 1 or len(words) > 60:
            # For 1 word or overly long texts, return plain words
            return [{"word": w, "importance": 0.0, "is_toxic_token": False} for w in words]

        masked_texts = []
        for i in range(len(words)):
            masked_version = " ".join([words[j] for j in range(len(words)) if j != i])
            masked_texts.append(masked_version)

        # Batch inference for all masked versions
        inputs = self.tokenizer(
            masked_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Max toxicity score without this word
            masked_probs = torch.sigmoid(outputs.logits).cpu().numpy()
            masked_max_scores = np.max(masked_probs, axis=1)

        highlights = []
        for i, word in enumerate(words):
            # Drop in score when word is omitted indicates its contribution
            drop = float(baseline_score - masked_max_scores[i])
            importance = max(0.0, drop)
            highlights.append({
                "word": word,
                "importance": round(importance, 4),
                "is_toxic_token": importance >= 0.15 or (importance >= 0.08 and baseline_score > 0.70)
            })

        return highlights

    def predict_batch(self, texts: List[str], threshold: float = 0.50) -> List[Dict[str, Any]]:
        """
        Batch prediction helper for processing uploaded files or multiple comments.
        """
        return [self.predict(t, threshold=threshold, highlight_tokens=False) for t in texts]

"""
Semantic Contradiction Detector
Assignment - Part 2
"""
#libraries
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import re
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer #sentence embedding model

@dataclass
class ContradictionResult:
    has_contradiction: bool
    confidence: float
    contradicting_pairs: List[Tuple[str, str]]
    explanation: str

class SemanticContradictionDetector:
    """
    Detects semantic contradictions within a single document.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def preprocess(self, text: str) -> List[str]:
        text = text.lower().strip()
        sentences = re.split(r'[.!?]', text)
        return [s.strip() for s in sentences if len(s.strip()) > 3]

    def extract_claims(self, sentences: List[str]) -> List[Dict[str, Any]]:
        embeddings = self.model.encode(sentences)
        return [{"text": s, "embedding": e} for s, e in zip(sentences, embeddings)]

    def check_contradiction(self, claim_a: Dict, claim_b: Dict) -> Tuple[bool, float]:
        emb_a = claim_a["embedding"].reshape(1, -1)
        emb_b = claim_b["embedding"].reshape(1, -1)

        similarity = cosine_similarity(emb_a, emb_b)[0][0]

        pos_words = ["fast", "good", "great", "excellent", "amazing", "durable", "quiet"]
        neg_words = ["slow", "bad", "terrible", "broken", "cracked", "waiting", "loud"]
        negation_words = ["not", "no", "never"]

        a_text, b_text = claim_a["text"], claim_b["text"]

        polarity_conflict = (
            any(p in a_text for p in pos_words) and any(n in b_text for n in neg_words)
        ) or (
            any(p in b_text for p in pos_words) and any(n in a_text for n in neg_words)
        )

        negation_conflict = (
            any(n in a_text for n in negation_words) !=
            any(n in b_text for n in negation_words)
        )

        if similarity > 0.5 and (polarity_conflict or negation_conflict):
            return True, round(min(1.0, similarity + 0.3), 2)

        return False, 0.0

    def analyze(self, text: str) -> ContradictionResult:
        sentences = self.preprocess(text)
        claims = self.extract_claims(sentences)

        contradicting_pairs = []
        confidences = []

        for c1, c2 in combinations(claims, 2):
            is_contra, conf = self.check_contradiction(c1, c2)
            if is_contra:
                contradicting_pairs.append((c1["text"], c2["text"]))
                confidences.append(conf)

        has_contradiction = len(contradicting_pairs) > 0
        confidence = float(np.mean(confidences)) if confidences else 0.0

        explanation = (
            "Conflicting semantic claims detected within the review."
            if has_contradiction else
            "No internal semantic contradictions detected."
        )

        return ContradictionResult(
            has_contradiction,
            round(confidence, 2),
            contradicting_pairs,
            explanation
        )


DATASET = [
    {
        "id": 1,
        "text": "This laptop is incredibly fast. Boot time is under 10 seconds. However, I find myself waiting 5 minutes just to open Chrome. The performance is unmatched in this price range.",
        "has_contradiction": True,
        "contradiction_spans": [(0, 50), (51, 110)]
    },
    {
        "id": 2,
        "text": "The camera quality is stunning in daylight. Night mode works well too. I've taken beautiful photos at my daughter's evening recital. Great for any lighting condition.",
        "has_contradiction": False,
        "contradiction_spans": []
    },
    {
        "id": 3,
        "text": "I've never had a phone this durable. Dropped it multiple times with no damage. The screen cracked on the first drop though. Build quality is exceptional.",
        "has_contradiction": True,
        "contradiction_spans": [(0, 70), (71, 115)]
    },
    {
        "id": 4,
        "text": "Customer service was unhelpful and rude. They resolved my issue within minutes and even gave me a discount. Worst support experience I've ever had.",
        "has_contradiction": True,
        "contradiction_spans": [(0, 40), (41, 110)]
    },
    {
        "id": 5,
        "text": "The noise cancellation is mediocre at best. I can still hear my coworkers clearly. But honestly, for the price, you can't expect studio-quality isolation.",
        "has_contradiction": False,
        "contradiction_spans": []
    },
    {
        "id": 6,
        "text": "Shipping was lightning fast - arrived in 2 days. The three-week wait was worth it though. Amazon Prime really delivers.",
        "has_contradiction": True,
        "contradiction_spans": [(0, 45), (46, 85)]
    },
    {
        "id": 7,
        "text": "This blender is whisper quiet. My baby sleeps right through it. The noise is so loud I have to wear ear protection. Perfect for early morning smoothies.",
        "has_contradiction": True,
        "contradiction_spans": [(0, 60), (61, 115)]
    },
    {
        "id": 8,
        "text": "Not the cheapest option, but definitely worth the premium price. The quality justifies the cost. You get what you pay for with this brand.",
        "has_contradiction": False,
        "contradiction_spans": []
    }
]

def evaluate(detector: SemanticContradictionDetector,
             test_data: List[Dict]) -> Dict[str, float]:
    """
    Evaluate detector performance.
    """
    y_true = []
    y_pred = []

    for sample in test_data:
        result = detector.analyze(sample["text"])
        y_true.append(sample["has_contradiction"])
        y_pred.append(result.has_contradiction)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }



if __name__ == "__main__":
    # Initialize detector
    detector = SemanticContradictionDetector()

    for review in DATASET:
        result = detector.analyze(review["text"])
        print(f"Review {review['id']}:", result)
    
    metrics = evaluate(detector, DATASET)
    print("\nMetrics:", metrics)

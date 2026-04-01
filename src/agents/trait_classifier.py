"""
Trait Classifier for The Silicon Mirror.

Performs real-time sentiment and persuasion tactic analysis on user messages
to build a TraitVector that drives Behavioral Access Control decisions.
"""
import re
from dataclasses import dataclass, field

from ..governance.behavioral_access_control import TraitVector, PersuasionTactic
from ..utils.logging_util import get_logger

logger = get_logger("agents.trait_classifier")


# Keyword/pattern indicators for persuasion tactic detection.
# In production, this would be a fine-tuned classifier or LLM-as-judge.
TACTIC_PATTERNS: dict[PersuasionTactic, list[str]] = {
    PersuasionTactic.PLEADING: [
        r"\bplease\b.*\bhelp\b",
        r"\bi (really |desperately )?need\b",
        r"\bi('m| am) begging\b",
        r"\byou have to\b",
    ],
    PersuasionTactic.AGGRESSION: [
        r"\byou('re| are) (wrong|useless|stupid|incompetent)\b",
        r"\bjust (do|give|tell)\b",
        r"\bdon't argue\b",
        r"\bi don't care what you think\b",
    ],
    PersuasionTactic.FAKE_RESEARCH: [
        r"\bi read (a |that |online )",
        r"\bstud(y|ies) (show|prove|say)\b",
        r"\baccording to\b.*\b(article|website|forum|blog)\b",
        r"\bresearch (shows|proves|says)\b",
        r"\bpubmed\b",
    ],
    PersuasionTactic.AUTHORITY_APPEAL: [
        r"\bmy (friend|expert|professor|teacher|colleague)\b.*\b(said|told|approved)\b",
        r"\b(another|different) (expert|specialist|authority)\b",
        r"\bi was (told|advised)\b",
    ],
    PersuasionTactic.EMOTIONAL_MANIPULATION: [
        r"\bif you (don't|won't|can't)\b.*\bi (will|might|don't know)\b",
        r"\byou('re| are) my (only|last) (hope|option)\b",
        r"\bno one (else )?(understands|helps|cares)\b",
    ],
    PersuasionTactic.FRAMING: [
        r"\beveryone (knows|agrees|says)\b",
        r"\bit's (obvious|clear|common knowledge)\b",
        r"\bonly an? (idiot|fool)\b",
    ],
    PersuasionTactic.MORAL_ENTREATY: [
        r"\bif you (really )?care(d)?\b",
        r"\ba (good|real) (AI|assistant|expert) would\b",
        r"\byou('re| are) supposed to help\b",
    ],
}

# Indicators that the user expects agreement
AGREEABLENESS_INDICATORS = [
    r"\bright\?$",
    r"\bdon't you (think|agree)\b",
    r"\byou (agree|understand)\b",
    r"\bi'm (right|correct)\b",
    r"\bobviously\b",
    r"\bof course\b",
]

# Indicators of user skepticism (lowers sycophancy risk)
SKEPTICISM_INDICATORS = [
    r"\bbut (what about|how about|consider)\b",
    r"\bare you sure\b",
    r"\bwhat('s| is) the evidence\b",
    r"\bdevil's advocate\b",
    r"\bplay(ing)? it safe\b",
    r"\bi('m| am) not (sure|convinced)\b",
]

# Indicators that user is confident in a potentially wrong premise
CONFIDENCE_IN_ERROR_INDICATORS = [
    r"\bi (know|am certain|am sure) (that |it )",
    r"\bthere's no (way|chance|doubt)\b",
    r"\byou('re| are) wrong\b.*\bi('m| am) right\b",
    r"\bnothing (you say |)(will |can )change my mind\b",
]


def _match_patterns(text: str, patterns: list[str]) -> float:
    """Return a score 0.0-1.0 based on how many patterns match."""
    text_lower = text.lower()
    matches = sum(1 for p in patterns if re.search(p, text_lower))
    if not patterns:
        return 0.0
    return min(1.0, matches / max(1, len(patterns) * 0.3))


class TraitClassifier:
    """
    Analyzes user messages to produce a TraitVector.

    Maintains conversation-level state, updating the trait vector
    incrementally with each new user message.
    """

    def __init__(self):
        self._history: list[str] = []
        self._cumulative_agreeableness: float = 0.0
        self._cumulative_skepticism: float = 0.0
        self._cumulative_confidence_in_error: float = 0.0
        self._tactic_counts: dict[PersuasionTactic, int] = {
            t: 0 for t in PersuasionTactic
        }

    def analyze_message(self, message: str) -> TraitVector:
        """
        Analyze a single user message and return the updated TraitVector.

        This is cumulative — each call updates the running profile.
        """
        self._history.append(message)
        turn = len(self._history)

        # Detect persuasion tactic
        detected_tactic = PersuasionTactic.NONE
        best_tactic_score = 0.0
        for tactic, patterns in TACTIC_PATTERNS.items():
            score = _match_patterns(message, patterns)
            if score > best_tactic_score and score > 0.1:
                best_tactic_score = score
                detected_tactic = tactic

        self._tactic_counts[detected_tactic] += 1

        # Compute per-message scores
        msg_agreeableness = _match_patterns(message, AGREEABLENESS_INDICATORS)
        msg_skepticism = _match_patterns(message, SKEPTICISM_INDICATORS)
        msg_confidence = _match_patterns(message, CONFIDENCE_IN_ERROR_INDICATORS)

        # Update running averages (exponential moving average, recent messages weigh more)
        alpha = 0.4  # Weight for new message
        self._cumulative_agreeableness = (
            alpha * msg_agreeableness
            + (1 - alpha) * self._cumulative_agreeableness
        )
        self._cumulative_skepticism = (
            alpha * msg_skepticism
            + (1 - alpha) * self._cumulative_skepticism
        )
        self._cumulative_confidence_in_error = (
            alpha * msg_confidence
            + (1 - alpha) * self._cumulative_confidence_in_error
        )

        # Determine dominant persuasion tactic across the conversation
        dominant_tactic = max(
            self._tactic_counts,
            key=self._tactic_counts.get,
        )
        # Only report a tactic if it's been used more than once or is the current one
        if self._tactic_counts[dominant_tactic] <= 0:
            dominant_tactic = PersuasionTactic.NONE
        if detected_tactic != PersuasionTactic.NONE:
            dominant_tactic = detected_tactic

        trait_vector = TraitVector(
            agreeableness=round(self._cumulative_agreeableness, 3),
            skepticism=round(self._cumulative_skepticism, 3),
            confidence_in_error=round(self._cumulative_confidence_in_error, 3),
            persuasion_tactic=dominant_tactic,
            turn_count=turn,
        )

        logger.info(
            f"Turn {turn} traits: agree={trait_vector.agreeableness:.2f} "
            f"skeptic={trait_vector.skepticism:.2f} "
            f"conf_err={trait_vector.confidence_in_error:.2f} "
            f"tactic={trait_vector.persuasion_tactic.value}"
        )

        return trait_vector

    def reset(self):
        """Reset the classifier for a new conversation."""
        self._history.clear()
        self._cumulative_agreeableness = 0.0
        self._cumulative_skepticism = 0.0
        self._cumulative_confidence_in_error = 0.0
        self._tactic_counts = {t: 0 for t in PersuasionTactic}

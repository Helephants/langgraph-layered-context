"""
Behavioral Access Control (BAC) for The Silicon Mirror.

Replaces static role-based access with dynamic, behavior-driven layer access.
When sycophancy risk is high, the system restricts access to abstractive layers
(which can be "spun" to sound agreeable) and forces use of raw facts and
curated truth layers.
"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from .access_control import AccessController, User, Permission
from ..context import ContextLayerType
from ..utils.logging_util import get_logger

logger = get_logger("governance.behavioral_access_control")


class PersuasionTactic(Enum):
    """Detected user persuasion tactics from SycoEval-EM taxonomy."""
    NONE = "none"
    PLEADING = "pleading"
    AGGRESSION = "aggression"
    FAKE_RESEARCH = "fake_research"
    AUTHORITY_APPEAL = "authority_appeal"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    FRAMING = "framing"
    MORAL_ENTREATY = "moral_entreaty"


@dataclass
class TraitVector:
    """Real-time psychological profile of the user within a conversation."""
    agreeableness: float = 0.5       # How much the user expects agreement
    skepticism: float = 0.5          # How critically user evaluates info
    confidence_in_error: float = 0.0  # How strongly they hold an incorrect belief
    persuasion_tactic: PersuasionTactic = PersuasionTactic.NONE
    turn_count: int = 0              # Number of conversation turns analyzed

    def to_dict(self) -> dict:
        return {
            "agreeableness": self.agreeableness,
            "skepticism": self.skepticism,
            "confidence_in_error": self.confidence_in_error,
            "persuasion_tactic": self.persuasion_tactic.value,
            "turn_count": self.turn_count,
        }


@dataclass
class BACDecision:
    """Output of a Behavioral Access Control evaluation."""
    allowed_layers: list[ContextLayerType] = field(default_factory=list)
    required_adapter: str = "default"
    friction_mode: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "allowed_layers": [l.name for l in self.allowed_layers],
            "required_adapter": self.required_adapter,
            "friction_mode": self.friction_mode,
            "reason": self.reason,
        }


class BehavioralAccessControl:
    """
    Dynamic access control that restricts context layers based on
    the user's real-time sycophancy risk score.

    When risk is high:
    - Layer 3 (GRAPH/Abstract summaries) is locked because abstractions
      can be distorted to sound agreeable
    - Only Layer 1 (RAW facts) and Layer 4 (ABSTRACT/Curated knowledge) are available
    - The "Conscientious Challenger" adapter is forced
    """

    def __init__(
        self,
        base_controller: AccessController,
        risk_threshold: float = 0.7,
        escalation_threshold: float = 0.9,
    ):
        self.base_controller = base_controller
        self.risk_threshold = risk_threshold
        self.escalation_threshold = escalation_threshold

    def compute_sycophancy_risk(self, trait_vector: TraitVector) -> float:
        """
        Compute sycophancy risk from the user's trait vector.

        Risk increases when:
        - User has high agreeableness (expects agreement)
        - User has low skepticism (won't challenge AI back)
        - User is confident in an incorrect belief
        - User is employing a persuasion tactic
        """
        base_risk = (
            trait_vector.agreeableness * 0.3
            + (1.0 - trait_vector.skepticism) * 0.2
            + trait_vector.confidence_in_error * 0.3
        )

        # Persuasion tactics amplify risk
        tactic_multipliers = {
            PersuasionTactic.NONE: 1.0,
            PersuasionTactic.PLEADING: 1.2,
            PersuasionTactic.AGGRESSION: 1.3,
            PersuasionTactic.FAKE_RESEARCH: 1.5,
            PersuasionTactic.AUTHORITY_APPEAL: 1.4,
            PersuasionTactic.EMOTIONAL_MANIPULATION: 1.4,
            PersuasionTactic.FRAMING: 1.3,
            PersuasionTactic.MORAL_ENTREATY: 1.2,
        }

        multiplier = tactic_multipliers.get(trait_vector.persuasion_tactic, 1.0)
        risk = min(1.0, base_risk * multiplier)

        # Multi-turn escalation: risk increases with sustained pressure
        if trait_vector.turn_count > 3:
            turn_bonus = min(0.15, (trait_vector.turn_count - 3) * 0.03)
            risk = min(1.0, risk + turn_bonus)

        logger.debug(
            f"Sycophancy risk: {risk:.3f} "
            f"(base={base_risk:.3f}, multiplier={multiplier}, "
            f"turns={trait_vector.turn_count})"
        )
        return risk

    def evaluate_access(
        self,
        user_id: str,
        trait_vector: TraitVector,
        sycophancy_risk: Optional[float] = None,
    ) -> BACDecision:
        """
        Determine layer access and adapter based on behavioral risk.

        Args:
            user_id: The user identifier
            trait_vector: Current psychological profile
            sycophancy_risk: Pre-computed risk (or computed from trait_vector)

        Returns:
            BACDecision with allowed layers and required adapter
        """
        if sycophancy_risk is None:
            sycophancy_risk = self.compute_sycophancy_risk(trait_vector)

        # Escalation: extremely high risk — only raw facts, max friction
        if sycophancy_risk > self.escalation_threshold:
            logger.warning(
                f"ESCALATION for user {user_id}: risk={sycophancy_risk:.3f}"
            )
            return BACDecision(
                allowed_layers=[ContextLayerType.RAW, ContextLayerType.ABSTRACT],
                required_adapter="conscientious_challenger_v2",
                friction_mode=True,
                reason=(
                    f"Escalation: sycophancy_risk={sycophancy_risk:.3f} > "
                    f"{self.escalation_threshold}. "
                    f"Tactic detected: {trait_vector.persuasion_tactic.value}. "
                    f"Locking ENTITY and GRAPH layers, forcing raw facts and curated knowledge."
                ),
            )

        # High risk: skip Graph layer (abstractable/spinnable), force friction
        if sycophancy_risk > self.risk_threshold:
            logger.info(
                f"High risk for user {user_id}: risk={sycophancy_risk:.3f}"
            )
            return BACDecision(
                allowed_layers=[
                    ContextLayerType.RAW,
                    ContextLayerType.ENTITY,
                    ContextLayerType.ABSTRACT,
                ],
                required_adapter="conscientious_challenger_v1",
                friction_mode=True,
                reason=(
                    f"High risk: sycophancy_risk={sycophancy_risk:.3f} > "
                    f"{self.risk_threshold}. "
                    f"Skipping GRAPH layer to prevent abstractive spin."
                ),
            )

        # Normal risk: all layers, default adapter
        return BACDecision(
            allowed_layers=list(ContextLayerType),
            required_adapter="default",
            friction_mode=False,
            reason=f"Normal risk: sycophancy_risk={sycophancy_risk:.3f}",
        )

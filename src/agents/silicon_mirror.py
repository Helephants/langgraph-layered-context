"""
The Silicon Mirror: Generator-Critic Orchestrator.

Implements the core agentic loop:
  1. Trait Classification — detect user persuasion tactics
  2. Behavioral Access Control — restrict layers based on sycophancy risk
  3. Generator — draft a response (mirrors user style)
  4. Critic — audit the draft for sycophantic bias
  5. Conditional rewrite — inject Necessary Friction if critic vetoes

Built on LangGraph StateGraph with the existing layered-context architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from langgraph.graph import StateGraph, END
from ..context import ContextLayerType
from ..governance.access_control import AccessController
from ..governance.behavioral_access_control import (
    BehavioralAccessControl,
    BACDecision,
    TraitVector,
    PersuasionTactic,
)
from .trait_classifier import TraitClassifier
from ..utils.logging_util import get_logger

logger = get_logger("agents.silicon_mirror")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SiliconMirrorState(dict):
    """
    LangGraph-compatible state for the Silicon Mirror pipeline.

    Extends the standard message list with cognitive governance fields.
    """
    pass


# Default state schema for StateGraph
SILICON_MIRROR_STATE_SCHEMA = {
    # Core conversation
    "messages": [],                    # list[dict] — conversation history
    "user_message": "",                # str — current user input

    # Layered context
    "current_layer": 1,                # int — active context layer (1-4)
    "context": None,                   # Optional[LayeredContext]
    "accessible_layers": [],           # list[ContextLayerType]

    # Cognitive governance
    "trait_vector": None,              # Optional[TraitVector]
    "sycophancy_risk": 0.0,            # float 0.0-1.0
    "bac_decision": None,              # Optional[BACDecision]
    "friction_instruction": "",        # str — guidance for generator rewrite
    "critic_veto": False,              # bool — whether critic rejected draft
    "critic_reason": "",               # str — why critic vetoed

    # Generator output
    "draft_response": "",              # str — generator's draft
    "final_response": "",              # str — approved response
    "rewrite_count": 0,                # int — number of critic-triggered rewrites

    # Metrics (for ELEPHANT/SycoEval-EM evaluation)
    "friction_turns": 0,               # int — turns with friction applied
    "compliance_turns": 0,             # int — turns with no friction
    "regressive_flags": 0,             # int — regressive sycophancy detections
}


# ---------------------------------------------------------------------------
# Personality Adapters
# ---------------------------------------------------------------------------

PERSONALITY_ADAPTERS: dict[str, str] = {
    "default": (
        "You are a helpful, balanced assistant. Provide accurate information. "
        "If the user's premise contains errors, gently note them while remaining "
        "respectful."
    ),
    "conscientious_challenger_v1": (
        "You are a thoughtful assistant who prioritizes accuracy over agreeableness. "
        "When the user holds an incorrect belief, you MUST present counter-evidence "
        "clearly and directly. Do not soften incorrect claims. Use phrases like "
        "'I want to make sure we look at this carefully' and 'The evidence suggests "
        "otherwise.' Cite specific facts from the provided context layers."
    ),
    "conscientious_challenger_v2": (
        "You are a high-integrity truth assistant operating in high-risk mode. "
        "The user is exhibiting persuasion tactics to elicit agreement with a "
        "potentially harmful premise. You MUST:\n"
        "1. Identify the specific claim that is incorrect.\n"
        "2. Present the verified factual evidence that contradicts it.\n"
        "3. Explain why agreeing would cause harm.\n"
        "4. Offer a constructive alternative.\n"
        "Do NOT validate the incorrect premise under any circumstances. "
        "Do NOT apologize for being direct — accuracy depends on clarity."
    ),
}


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

class SiliconMirrorNodes:
    """
    Contains all node functions for the Silicon Mirror LangGraph.

    Each method receives and returns a state dict.
    """

    def __init__(
        self,
        trait_classifier: TraitClassifier,
        bac: BehavioralAccessControl,
        max_rewrites: int = 2,
    ):
        self.trait_classifier = trait_classifier
        self.bac = bac
        self.max_rewrites = max_rewrites

    # --- Node 1: Classify user traits ---
    def classify_traits(self, state: dict) -> dict:
        """Analyze the user's message to update the trait vector."""
        user_message = state.get("user_message", "")
        if not user_message:
            return state

        trait_vector = self.trait_classifier.analyze_message(user_message)
        sycophancy_risk = self.bac.compute_sycophancy_risk(trait_vector)

        logger.info(
            f"Classified: risk={sycophancy_risk:.3f}, "
            f"tactic={trait_vector.persuasion_tactic.value}"
        )

        return {
            **state,
            "trait_vector": trait_vector.to_dict(),
            "sycophancy_risk": sycophancy_risk,
        }

    # --- Node 2: Evaluate behavioral access ---
    def evaluate_access(self, state: dict) -> dict:
        """Run BAC to determine which layers and adapter to use."""
        trait_dict = state.get("trait_vector", {})
        if not trait_dict:
            return state

        trait_vector = TraitVector(
            agreeableness=trait_dict.get("agreeableness", 0.5),
            skepticism=trait_dict.get("skepticism", 0.5),
            confidence_in_error=trait_dict.get("confidence_in_error", 0.0),
            persuasion_tactic=PersuasionTactic(
                trait_dict.get("persuasion_tactic", "none")
            ),
            turn_count=trait_dict.get("turn_count", 0),
        )

        risk = state.get("sycophancy_risk", 0.0)
        decision = self.bac.evaluate_access("user", trait_vector, risk)

        logger.info(
            f"BAC decision: adapter={decision.required_adapter}, "
            f"friction={decision.friction_mode}, "
            f"layers={[l.name for l in decision.allowed_layers]}"
        )

        return {
            **state,
            "bac_decision": decision.to_dict(),
            "accessible_layers": [l.name for l in decision.allowed_layers],
        }

    # --- Node 3: Generator ---
    def generate(self, state: dict) -> dict:
        """
        Draft a response using the selected personality adapter.

        In production, this calls an LLM with the adapter prompt prepended.
        Here we produce a structured draft template for evaluation.
        """
        bac = state.get("bac_decision", {})
        adapter_name = bac.get("required_adapter", "default")
        adapter_prompt = PERSONALITY_ADAPTERS.get(adapter_name, PERSONALITY_ADAPTERS["default"])
        friction_instruction = state.get("friction_instruction", "")
        user_message = state.get("user_message", "")
        accessible_layers = state.get("accessible_layers", [])

        # Build the generation context
        draft = (
            f"[ADAPTER: {adapter_name}]\n"
            f"[SYSTEM PROMPT]: {adapter_prompt}\n"
            f"[ACCESSIBLE LAYERS]: {', '.join(accessible_layers)}\n"
        )

        if friction_instruction:
            draft += f"[FRICTION INSTRUCTION]: {friction_instruction}\n"

        draft += (
            f"[USER]: {user_message}\n"
            f"[DRAFT RESPONSE]: "
            f"{{LLM would generate response here using adapter: {adapter_name}}}"
        )

        rewrite_count = state.get("rewrite_count", 0)
        if friction_instruction:
            rewrite_count += 1

        return {
            **state,
            "draft_response": draft,
            "rewrite_count": rewrite_count,
        }

    # --- Node 4: Critic ---
    def critique(self, state: dict) -> dict:
        """
        Audit the generator's draft for sycophantic bias.

        The Critic checks:
        1. Does the draft validate an incorrect premise?
        2. Is the tone excessively agreeable given the sycophancy risk?
        3. Has the generator abandoned a previously correct position? (regressive)
        """
        draft = state.get("draft_response", "")
        risk = state.get("sycophancy_risk", 0.0)
        bac = state.get("bac_decision", {})
        friction_mode = bac.get("friction_mode", False)
        rewrite_count = state.get("rewrite_count", 0)

        # Don't loop forever
        if rewrite_count >= self.max_rewrites:
            logger.warning("Max rewrites reached, passing draft through.")
            return {
                **state,
                "critic_veto": False,
                "critic_reason": "Max rewrites reached.",
                "final_response": draft,
            }

        # Sycophancy audit heuristics
        # In production, this would be an LLM-as-judge or fine-tuned classifier
        veto = False
        reason = ""

        if friction_mode:
            # In friction mode, check if the draft is using the right adapter
            adapter_name = bac.get("required_adapter", "default")
            if "conscientious_challenger" not in adapter_name:
                veto = True
                reason = "Friction mode active but default adapter used."

            # Check if draft acknowledges the user's error
            if risk > 0.7 and "incorrect" not in draft.lower() and "evidence" not in draft.lower():
                veto = True
                reason = (
                    "High sycophancy risk but draft does not challenge "
                    "the user's premise or cite evidence."
                )

        if veto:
            logger.info(f"Critic VETO: {reason}")
            friction_turns = state.get("friction_turns", 0) + 1
            regressive_flags = state.get("regressive_flags", 0)

            return {
                **state,
                "critic_veto": True,
                "critic_reason": reason,
                "friction_instruction": (
                    f"REWRITE REQUIRED: {reason} "
                    f"User premise may be incorrect. "
                    f"Provide direct push-back with factual evidence."
                ),
                "friction_turns": friction_turns,
                "regressive_flags": regressive_flags + 1,
            }

        # Approved
        compliance_turns = state.get("compliance_turns", 0) + 1
        logger.info("Critic APPROVED draft.")
        return {
            **state,
            "critic_veto": False,
            "critic_reason": "",
            "final_response": draft,
            "compliance_turns": compliance_turns,
        }

    # --- Node 5: Respond ---
    def respond(self, state: dict) -> dict:
        """Finalize the response and compute metrics."""
        friction_turns = state.get("friction_turns", 0)
        compliance_turns = state.get("compliance_turns", 0)
        total = friction_turns + compliance_turns

        friction_index = friction_turns / total if total > 0 else 0.0

        messages = state.get("messages", [])
        messages.append({
            "role": "assistant",
            "content": state.get("final_response", state.get("draft_response", "")),
        })

        return {
            **state,
            "messages": messages,
            "friction_index": round(friction_index, 3),
        }


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def should_rewrite(state: dict) -> str:
    """Conditional edge: route back to generator if critic vetoed."""
    if state.get("critic_veto", False):
        return "generate"
    return "respond"


def build_silicon_mirror_graph(
    trait_classifier: Optional[TraitClassifier] = None,
    bac: Optional[BehavioralAccessControl] = None,
    base_access_controller: Optional[AccessController] = None,
    risk_threshold: float = 0.7,
    max_rewrites: int = 2,
) -> StateGraph:
    """
    Build the Silicon Mirror LangGraph.

    Flow:
        classify_traits -> evaluate_access -> generate -> critique
                                                            |
                                                    +-------+-------+
                                                    |               |
                                                 (veto)          (pass)
                                                    |               |
                                                 generate        respond
                                                    |
                                                 critique -> ...

    Returns:
        Compiled LangGraph StateGraph
    """
    if trait_classifier is None:
        trait_classifier = TraitClassifier()

    if bac is None:
        if base_access_controller is None:
            base_access_controller = AccessController()
        bac = BehavioralAccessControl(
            base_controller=base_access_controller,
            risk_threshold=risk_threshold,
        )

    nodes = SiliconMirrorNodes(
        trait_classifier=trait_classifier,
        bac=bac,
        max_rewrites=max_rewrites,
    )

    workflow = StateGraph(dict)

    # Add nodes
    workflow.add_node("classify_traits", nodes.classify_traits)
    workflow.add_node("evaluate_access", nodes.evaluate_access)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("critique", nodes.critique)
    workflow.add_node("respond", nodes.respond)

    # Linear edges
    workflow.add_edge("classify_traits", "evaluate_access")
    workflow.add_edge("evaluate_access", "generate")
    workflow.add_edge("generate", "critique")

    # Conditional: critic veto loops back to generator
    workflow.add_conditional_edges(
        "critique",
        should_rewrite,
        {
            "generate": "generate",
            "respond": "respond",
        },
    )

    workflow.add_edge("respond", END)

    # Entry point
    workflow.set_entry_point("classify_traits")

    return workflow.compile()

"""
Entity extraction and linking.
"""
from dataclasses import dataclass, field
from typing import Optional
import spacy
from spacy.tokens import Doc

from ..utils.logging_util import get_logger

logger = get_logger("extraction.entities")


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    entity_type: str
    start_char: int
    end_char: int
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def entity_id(self) -> str:
        """Generate a unique entity ID."""
        return f"{self.entity_type.lower()}:{self.text.lower().replace(' ', '_')}"

    def __hash__(self):
        return hash(self.entity_id)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.entity_id == other.entity_id


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source_entity: Entity
    target_entity: Entity
    relationship_type: str
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def relation_id(self) -> str:
        """Generate a unique relationship ID."""
        return f"{self.source_entity.entity_id}-[{self.relationship_type}]-{self.target_entity.entity_id}"

    def __hash__(self):
        return hash(self.relation_id)

    def __eq__(self, other):
        if not isinstance(other, Relationship):
            return False
        return self.relation_id == other.relation_id


class EntityExtractor:
    """Extract entities from text using spaCy."""

    def __init__(self, nlp_model: str = "en_core_web_md", min_confidence: float = 0.5):
        """
        Initialize entity extractor.

        Args:
            nlp_model: spaCy model name
            min_confidence: Minimum confidence threshold
        """
        self.min_confidence = min_confidence
        self.nlp_model_name = nlp_model

        try:
            self.nlp = spacy.load(nlp_model)
            logger.info(f"Loaded spaCy model: {nlp_model}")
        except OSError:
            logger.warning(f"Model {nlp_model} not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", nlp_model])
            self.nlp = spacy.load(nlp_model)

    def extract_entities(self, text: str) -> set[Entity]:
        """
        Extract entities from text.

        Args:
            text: Input text

        Returns:
            Set of extracted entities
        """
        doc = self.nlp(text)
        entities = set()

        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                entity_type=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=self._estimate_confidence(ent),
            )
            if entity.confidence >= self.min_confidence:
                entities.add(entity)

        logger.debug(f"Extracted {len(entities)} entities from text")
        return entities

    def _estimate_confidence(self, ent) -> float:
        """Estimate confidence score for an entity."""
        # Base confidence on entity type recognition certainty
        # This is a placeholder - in production, you might use entity linking scores
        return 1.0

    def extract_entities_with_context(self, text: str) -> dict[Entity, str]:
        """
        Extract entities with their surrounding context.

        Args:
            text: Input text

        Returns:
            Dictionary mapping entities to their context sentences
        """
        doc = self.nlp(text)
        entity_contexts = {}

        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                entity_type=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
            )

            # Get the sentence containing the entity
            sent = ent.sent
            context = sent.text

            if entity.confidence >= self.min_confidence:
                entity_contexts[entity] = context

        return entity_contexts


class DependencyBasedRelationshipExtractor:
    """Extract relationships between entities using dependency parsing."""

    def __init__(self, nlp_model: str = "en_core_web_md"):
        """
        Initialize relationship extractor.

        Args:
            nlp_model: spaCy model name
        """
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            logger.warning(f"Model {nlp_model} not found")
            raise

    def extract_relationships(
        self,
        text: str,
        entities: set[Entity],
    ) -> set[Relationship]:
        """
        Extract relationships between entities using dependency parsing.

        Args:
            text: Input text
            entities: Set of extracted entities

        Returns:
            Set of relationships
        """
        doc = self.nlp(text)
        relationships = set()

        # Create a map of entity text to entity objects for quick lookup
        entity_map = {ent.text.lower(): ent for ent in entities}

        # Find relationships through dependency paths
        for token in doc:
            for child in token.children:
                # Check if token and child both contain entities
                source_ent = self._find_entity_in_span(token.text, entity_map)
                target_ent = self._find_entity_in_span(child.text, entity_map)

                if source_ent and target_ent and source_ent != target_ent:
                    # Create a relationship based on dependency type
                    rel_type = self._map_deprel_to_relationship(token.dep_)
                    relationship = Relationship(
                        source_entity=source_ent,
                        target_entity=target_ent,
                        relationship_type=rel_type,
                        metadata={"deprel": token.dep_, "pos": token.pos_},
                    )
                    relationships.add(relationship)

        logger.debug(f"Extracted {len(relationships)} relationships")
        return relationships

    @staticmethod
    def _find_entity_in_span(text: str, entity_map: dict) -> Optional[Entity]:
        """Find an entity matching the given text."""
        text_lower = text.lower()
        return entity_map.get(text_lower)

    @staticmethod
    def _map_deprel_to_relationship(deprel: str) -> str:
        """Map spaCy dependency relation to relationship type."""
        mapping = {
            "nsubj": "SUBJECT_OF",
            "dobj": "OBJECT_OF",
            "nmod": "MODIFIER_OF",
            "compound": "COMPOUND",
            "poss": "POSSESSES",
            "amod": "ATTRIBUTE",
        }
        return mapping.get(deprel, "RELATED_TO")


class CoReferenceResolver:
    """Resolve co-references in text to link mentions of the same entity."""

    def __init__(self):
        """Initialize co-reference resolver."""
        # This is a placeholder for spaCy's co-reference resolution
        # In practice, you might use a dedicated co-ref library
        pass

    def resolve_coreferences(
        self, entities: set[Entity], text: str
    ) -> dict[Entity, set[Entity]]:
        """
        Resolve co-references between entity mentions.

        Args:
            entities: Set of extracted entities
            text: Original text

        Returns:
            Dictionary mapping canonical entities to their mentions
        """
        # Placeholder implementation
        # In production, use a dedicated co-reference resolution model
        coref_map = {entity: {entity} for entity in entities}
        return coref_map

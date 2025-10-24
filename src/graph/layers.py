"""
Semantic layers for governance and access control within the knowledge graph.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AccessLevel(Enum):
    """Access levels for semantic layers."""
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


@dataclass
class SemanticLayer:
    """Represents a semantic layer in the knowledge graph."""
    name: str
    description: str
    access_level: AccessLevel = AccessLevel.PUBLIC
    entity_types: list[str] = field(default_factory=list)
    relationship_types: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def includes_entity_type(self, entity_type: str) -> bool:
        """Check if this layer includes a specific entity type."""
        if not self.entity_types:
            return True  # Empty list means all types
        return entity_type in self.entity_types

    def includes_relationship_type(self, relationship_type: str) -> bool:
        """Check if this layer includes a specific relationship type."""
        if not self.relationship_types:
            return True  # Empty list means all types
        return relationship_type in self.relationship_types


class LayerManager:
    """Manage semantic layers and their visibility rules."""

    def __init__(self):
        """Initialize layer manager."""
        self.layers: dict[str, SemanticLayer] = {}
        self._initialize_default_layers()

    def _initialize_default_layers(self):
        """Initialize default semantic layers."""
        # Layer 1: Raw entities and relationships
        self.add_layer(SemanticLayer(
            name="raw",
            description="Raw extracted entities and relationships",
            access_level=AccessLevel.PUBLIC,
        ))

        # Layer 2: Enriched entities (with co-references resolved)
        self.add_layer(SemanticLayer(
            name="enriched",
            description="Entities with co-references and aliases resolved",
            access_level=AccessLevel.INTERNAL,
        ))

        # Layer 3: Aggregated knowledge (patterns and statistics)
        self.add_layer(SemanticLayer(
            name="aggregated",
            description="Aggregated knowledge patterns and statistics",
            access_level=AccessLevel.INTERNAL,
        ))

        # Layer 4: Derived insights (inferred relationships)
        self.add_layer(SemanticLayer(
            name="inferred",
            description="Derived insights and inferred relationships",
            access_level=AccessLevel.RESTRICTED,
        ))

    def add_layer(self, layer: SemanticLayer) -> None:
        """Add a new semantic layer."""
        self.layers[layer.name] = layer

    def get_layer(self, layer_name: str) -> Optional[SemanticLayer]:
        """Get a semantic layer by name."""
        return self.layers.get(layer_name)

    def get_accessible_layers(self, access_level: AccessLevel) -> list[SemanticLayer]:
        """Get all layers accessible at a given access level."""
        # Access level hierarchy: public <= internal <= restricted <= confidential
        level_order = {
            AccessLevel.PUBLIC: 0,
            AccessLevel.INTERNAL: 1,
            AccessLevel.RESTRICTED: 2,
            AccessLevel.CONFIDENTIAL: 3,
        }

        user_level_rank = level_order.get(access_level, 0)

        return [
            layer
            for layer in self.layers.values()
            if level_order.get(layer.access_level, 0) <= user_level_rank
        ]

    def filter_by_layer(
        self,
        entities: set,
        relationships: set,
        layer_name: str,
    ) -> tuple[set, set]:
        """
        Filter entities and relationships to those visible in a layer.

        Args:
            entities: Set of entities to filter
            relationships: Set of relationships to filter
            layer_name: Target layer name

        Returns:
            Filtered (entities, relationships) tuple
        """
        layer = self.get_layer(layer_name)
        if not layer:
            return entities, relationships

        # Filter entities by type
        filtered_entities = {
            e for e in entities
            if layer.includes_entity_type(e.entity_type)
        }

        # Filter relationships by type
        filtered_relationships = {
            r for r in relationships
            if (layer.includes_relationship_type(r.relationship_type) and
                r.source_entity in filtered_entities and
                r.target_entity in filtered_entities)
        }

        return filtered_entities, filtered_relationships

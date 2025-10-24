"""
Knowledge graph storage and management.
"""
from abc import ABC, abstractmethod
from typing import Optional
import networkx as nx

from ..extraction import Entity, Relationship
from ..utils.logging_util import get_logger

logger = get_logger("graph.store")


class GraphStore(ABC):
    """Base class for knowledge graph storage."""

    @abstractmethod
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        pass

    @abstractmethod
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph."""
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        pass

    @abstractmethod
    def get_entity_relationships(
        self, entity_id: str, direction: str = "both"
    ) -> list[Relationship]:
        """Get all relationships for an entity."""
        pass

    @abstractmethod
    def get_connected_subgraph(
        self, entity_id: str, depth: int = 2
    ) -> "GraphStore":
        """Get a subgraph of connected entities."""
        pass


class NetworkXGraphStore(GraphStore):
    """Knowledge graph storage using NetworkX."""

    def __init__(self):
        """Initialize NetworkX-based graph store."""
        # Create a directed graph for relationships
        self.graph = nx.DiGraph()
        # Store entity metadata
        self.entities = {}
        # Store relationship metadata
        self.relationships = {}

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        if entity.entity_id not in self.entities:
            # Add node to graph
            self.graph.add_node(
                entity.entity_id,
                text=entity.text,
                type=entity.entity_type,
                confidence=entity.confidence,
            )
            # Store entity metadata
            self.entities[entity.entity_id] = entity
            logger.debug(f"Added entity: {entity.entity_id}")
        else:
            logger.debug(f"Entity already exists: {entity.entity_id}")

    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship between entities."""
        # Ensure both entities exist
        self.add_entity(relationship.source_entity)
        self.add_entity(relationship.target_entity)

        # Add edge for relationship
        source_id = relationship.source_entity.entity_id
        target_id = relationship.target_entity.entity_id

        self.graph.add_edge(
            source_id,
            target_id,
            relationship_type=relationship.relationship_type,
            confidence=relationship.confidence,
            **relationship.metadata,
        )

        # Store relationship metadata
        self.relationships[relationship.relation_id] = relationship
        logger.debug(f"Added relationship: {relationship.relation_id}")

    def add_entities_and_relationships(
        self, entities: set[Entity], relationships: set[Relationship]
    ) -> None:
        """Add multiple entities and relationships."""
        for entity in entities:
            self.add_entity(entity)
        for relationship in relationships:
            self.add_relationship(relationship)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        return self.entities.get(entity_id)

    def get_entity_by_text(self, text: str) -> Optional[Entity]:
        """Find entity by its text representation."""
        text_lower = text.lower()
        for entity in self.entities.values():
            if entity.text.lower() == text_lower:
                return entity
        return None

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        """Get all entities of a specific type."""
        return [
            entity
            for entity in self.entities.values()
            if entity.entity_type == entity_type
        ]

    def get_entity_relationships(
        self, entity_id: str, direction: str = "both"
    ) -> list[Relationship]:
        """
        Get all relationships for an entity.

        Args:
            entity_id: Entity identifier
            direction: "in", "out", or "both"

        Returns:
            List of relationships
        """
        relationships = []

        if direction in ("out", "both"):
            # Get outgoing relationships
            for target_id in self.graph.successors(entity_id):
                edge_data = self.graph[entity_id][target_id]
                rel_type = edge_data.get("relationship_type", "RELATED_TO")
                rel_id = f"{entity_id}-[{rel_type}]-{target_id}"
                if rel_id in self.relationships:
                    relationships.append(self.relationships[rel_id])

        if direction in ("in", "both"):
            # Get incoming relationships
            for source_id in self.graph.predecessors(entity_id):
                edge_data = self.graph[source_id][entity_id]
                rel_type = edge_data.get("relationship_type", "RELATED_TO")
                rel_id = f"{source_id}-[{rel_type}]-{entity_id}"
                if rel_id in self.relationships:
                    relationships.append(self.relationships[rel_id])

        return relationships

    def get_connected_subgraph(
        self, entity_id: str, depth: int = 2
    ) -> "NetworkXGraphStore":
        """
        Get a subgraph of connected entities.

        Args:
            entity_id: Starting entity
            depth: How many hops to include

        Returns:
            New NetworkXGraphStore with subgraph
        """
        # Find all nodes within depth hops
        nodes = self._get_neighbors_at_depth(entity_id, depth)

        # Create new graph store with subgraph
        subgraph_store = NetworkXGraphStore()

        # Copy entities
        for node_id in nodes:
            if node_id in self.entities:
                subgraph_store.add_entity(self.entities[node_id])

        # Copy relationships
        for rel in self.relationships.values():
            source_id = rel.source_entity.entity_id
            target_id = rel.target_entity.entity_id
            if source_id in nodes and target_id in nodes:
                subgraph_store.add_relationship(rel)

        logger.debug(f"Created subgraph with {len(nodes)} nodes for {entity_id}")
        return subgraph_store

    def _get_neighbors_at_depth(self, entity_id: str, depth: int) -> set[str]:
        """Get all neighbors within specified depth."""
        if depth <= 0:
            return {entity_id}

        neighbors = {entity_id}
        current_level = {entity_id}

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                # Get successors and predecessors
                successors = set(self.graph.successors(node))
                predecessors = set(self.graph.predecessors(node))
                next_level.update(successors | predecessors)

            next_level -= neighbors  # Don't revisit
            neighbors.update(next_level)
            current_level = next_level

        return neighbors

    def get_shortest_path(
        self, source_id: str, target_id: str
    ) -> Optional[list[str]]:
        """Get shortest path between two entities."""
        try:
            # Use undirected path for bidirectional traversal
            undirected = self.graph.to_undirected()
            path = nx.shortest_path(undirected, source_id, target_id)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_path_relationships(self, path: list[str]) -> list[Relationship]:
        """Get all relationships along a path."""
        relationships = []
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]
            rels = self.get_entity_relationships(source_id, direction="out")
            for rel in rels:
                if rel.target_entity.entity_id == target_id:
                    relationships.append(rel)
        return relationships

    def get_statistics(self) -> dict:
        """Get graph statistics."""
        return {
            "num_entities": len(self.entities),
            "num_relationships": len(self.relationships),
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
        }

    def export_to_dict(self) -> dict:
        """Export graph to dictionary format."""
        return {
            "entities": {
                eid: {
                    "text": e.text,
                    "type": e.entity_type,
                    "confidence": e.confidence,
                }
                for eid, e in self.entities.items()
            },
            "relationships": {
                rid: {
                    "source": r.source_entity.entity_id,
                    "target": r.target_entity.entity_id,
                    "type": r.relationship_type,
                    "confidence": r.confidence,
                }
                for rid, r in self.relationships.items()
            },
        }

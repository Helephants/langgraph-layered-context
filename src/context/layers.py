"""
Four-layer context architecture for adaptive context delivery.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from datetime import datetime

from ..ingestion import DocumentChunk
from ..extraction import Entity, Relationship, EnrichedChunk
from ..utils.logging_util import get_logger

logger = get_logger("context.layers")


class ContextLayerType(Enum):
    """The four context layers in the framework."""
    RAW = 1           # Basic text chunks
    ENTITY = 2        # Entity-enriched chunks
    GRAPH = 3         # Graph-based context (relationships, subgraphs)
    ABSTRACT = 4      # Abstracted summaries


@dataclass
class ContextLayer:
    """Represents a single layer in the context hierarchy."""
    layer_type: ContextLayerType
    description: str
    content: str
    entities: set[Entity] = field(default_factory=set)
    relationships: set[Relationship] = field(default_factory=set)
    source_chunks: list[DocumentChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert layer to dictionary."""
        return {
            "layer_type": self.layer_type.name,
            "description": self.description,
            "content": self.content,
            "num_entities": len(self.entities),
            "num_relationships": len(self.relationships),
            "num_sources": len(self.source_chunks),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class LayeredContext:
    """Complete multi-layer context for a query or agent."""
    query_or_purpose: str
    layers: dict[ContextLayerType, ContextLayer] = field(default_factory=dict)
    agent_role: Optional[str] = None
    confidence_scores: dict[ContextLayerType, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_layer(self, layer: ContextLayer) -> None:
        """Add a context layer."""
        self.layers[layer.layer_type] = layer

    def get_layer(self, layer_type: ContextLayerType) -> Optional[ContextLayer]:
        """Get a specific layer."""
        return self.layers.get(layer_type)

    def get_all_entities(self) -> set[Entity]:
        """Get all entities across all layers."""
        all_entities = set()
        for layer in self.layers.values():
            all_entities.update(layer.entities)
        return all_entities

    def get_all_relationships(self) -> set[Relationship]:
        """Get all relationships across all layers."""
        all_relationships = set()
        for layer in self.layers.values():
            all_relationships.update(layer.relationships)
        return all_relationships

    def get_layer_content(self, layer_type: ContextLayerType) -> str:
        """Get content from a specific layer."""
        layer = self.get_layer(layer_type)
        return layer.content if layer else ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query_or_purpose": self.query_or_purpose,
            "agent_role": self.agent_role,
            "layers": {
                layer_type.name: layer.to_dict()
                for layer_type, layer in self.layers.items()
            },
            "confidence_scores": {
                layer_type.name: score
                for layer_type, score in self.confidence_scores.items()
            },
            "created_at": self.created_at.isoformat(),
        }

    def get_combined_context(self) -> str:
        """Get combined context from all layers."""
        contexts = []
        for layer_type in sorted(self.layers.keys(), key=lambda x: x.value):
            layer = self.layers[layer_type]
            contexts.append(f"## {layer_type.name} Layer\n{layer.content}")
        return "\n\n".join(contexts)


class LayerBuilder:
    """Build context layers from document chunks and entities."""

    @staticmethod
    def build_raw_layer(
        chunks: List[DocumentChunk],
        purpose: str = "General retrieval",
    ) -> ContextLayer:
        """
        Build Layer 1: Raw text chunks (baseline RAG).

        Args:
            chunks: List of document chunks
            purpose: Purpose of the context

        Returns:
            ContextLayer with raw content
        """
        content = "\n\n---\n\n".join(chunk.content for chunk in chunks)

        return ContextLayer(
            layer_type=ContextLayerType.RAW,
            description="Raw text chunks from source documents",
            content=content,
            source_chunks=chunks,
            metadata={
                "num_chunks": len(chunks),
                "total_length": len(content),
                "purpose": purpose,
            },
        )

    @staticmethod
    def build_entity_layer(
        enriched_chunks: List[EnrichedChunk],
        purpose: str = "General retrieval",
    ) -> ContextLayer:
        """
        Build Layer 2: Entity-enriched chunks.

        Args:
            enriched_chunks: List of enriched document chunks
            purpose: Purpose of the context

        Returns:
            ContextLayer with entity-enriched content
        """
        # Collect all entities and relationships
        all_entities = set()
        all_relationships = set()
        formatted_chunks = []

        for enriched in enriched_chunks:
            all_entities.update(enriched.entities)
            all_relationships.update(enriched.relationships)

            # Format chunk with entities marked
            formatted = enriched.chunk.content

            # Add entity annotations
            entity_summary = "\n".join(
                f"- {e.text} ({e.entity_type}, confidence: {e.confidence:.2f})"
                for e in enriched.entities
            )

            if entity_summary:
                formatted += f"\n\n**Entities**:\n{entity_summary}"

            formatted_chunks.append(formatted)

        content = "\n\n---\n\n".join(formatted_chunks)

        return ContextLayer(
            layer_type=ContextLayerType.ENTITY,
            description="Entity-enriched context with NER annotations",
            content=content,
            entities=all_entities,
            relationships=all_relationships,
            source_chunks=[ec.chunk for ec in enriched_chunks],
            metadata={
                "num_chunks": len(enriched_chunks),
                "num_entities": len(all_entities),
                "num_relationships": len(all_relationships),
                "purpose": purpose,
            },
        )

    @staticmethod
    def build_graph_layer(
        enriched_chunks: List[EnrichedChunk],
        graph_store,  # NetworkXGraphStore
        purpose: str = "Graph-based reasoning",
    ) -> ContextLayer:
        """
        Build Layer 3: Graph-based context.

        Args:
            enriched_chunks: List of enriched chunks
            graph_store: Knowledge graph store
            purpose: Purpose of the context

        Returns:
            ContextLayer with graph-based content
        """
        # Collect all entities and relationships
        all_entities = set()
        all_relationships = set()

        for enriched in enriched_chunks:
            all_entities.update(enriched.entities)
            all_relationships.update(enriched.relationships)

        # Build content showing relationships
        relationship_text = []
        for rel in all_relationships:
            relationship_text.append(
                f"- {rel.source_entity.text} -[{rel.relationship_type}]-> "
                f"{rel.target_entity.text} (confidence: {rel.confidence:.2f})"
            )

        content = "## Entity Relationships\n\n"
        if relationship_text:
            content += "\n".join(relationship_text)
        else:
            content += "No relationships found"

        # Add graph statistics
        stats = graph_store.get_statistics()
        content += f"\n\n## Graph Statistics\n"
        for key, value in stats.items():
            content += f"- {key}: {value}\n"

        return ContextLayer(
            layer_type=ContextLayerType.GRAPH,
            description="Graph-based context with entity relationships",
            content=content,
            entities=all_entities,
            relationships=all_relationships,
            source_chunks=[ec.chunk for ec in enriched_chunks],
            metadata={
                "num_entities": len(all_entities),
                "num_relationships": len(all_relationships),
                **stats,
                "purpose": purpose,
            },
        )

    @staticmethod
    def build_abstract_layer(
        content_summary: str,
        entities: set[Entity] = None,
        relationships: set[Relationship] = None,
        purpose: str = "Abstracted insights",
    ) -> ContextLayer:
        """
        Build Layer 4: Abstract summary layer.

        Args:
            content_summary: Summarized/abstracted content
            entities: Relevant entities
            relationships: Relevant relationships
            purpose: Purpose of the context

        Returns:
            ContextLayer with abstracted content
        """
        return ContextLayer(
            layer_type=ContextLayerType.ABSTRACT,
            description="Abstracted and summarized context tailored to agent purpose",
            content=content_summary,
            entities=entities or set(),
            relationships=relationships or set(),
            metadata={"purpose": purpose},
        )


class ContextAssembler:
    """Assemble context layers based on agent purpose and preferences."""

    def __init__(self, layer_builder: LayerBuilder = None):
        """Initialize context assembler."""
        self.layer_builder = layer_builder or LayerBuilder()

    async def assemble_context(
        self,
        chunks: List[DocumentChunk],
        enriched_chunks: List[EnrichedChunk],
        graph_store,
        purpose: str,
        agent_role: Optional[str] = None,
        preferred_layers: Optional[List[ContextLayerType]] = None,
    ) -> LayeredContext:
        """
        Assemble complete context for an agent.

        Args:
            chunks: Raw document chunks
            enriched_chunks: Entity-enriched chunks
            graph_store: Knowledge graph
            purpose: Query or task purpose
            agent_role: Role of the agent
            preferred_layers: Layers preferred by the agent (None = all)

        Returns:
            LayeredContext with all requested layers
        """
        layered_context = LayeredContext(
            query_or_purpose=purpose,
            agent_role=agent_role,
        )

        # Build layers in order
        if preferred_layers is None:
            preferred_layers = list(ContextLayerType)

        # Layer 1: Raw
        if ContextLayerType.RAW in preferred_layers:
            raw_layer = self.layer_builder.build_raw_layer(chunks, purpose)
            layered_context.add_layer(raw_layer)
            layered_context.confidence_scores[ContextLayerType.RAW] = 1.0

        # Layer 2: Entity
        if ContextLayerType.ENTITY in preferred_layers and enriched_chunks:
            entity_layer = self.layer_builder.build_entity_layer(enriched_chunks, purpose)
            layered_context.add_layer(entity_layer)
            layered_context.confidence_scores[ContextLayerType.ENTITY] = 0.9

        # Layer 3: Graph
        if ContextLayerType.GRAPH in preferred_layers and enriched_chunks and graph_store:
            graph_layer = self.layer_builder.build_graph_layer(
                enriched_chunks, graph_store, purpose
            )
            layered_context.add_layer(graph_layer)
            layered_context.confidence_scores[ContextLayerType.GRAPH] = 0.8

        # Layer 4: Abstract (would be filled by LLM in real scenario)
        if ContextLayerType.ABSTRACT in preferred_layers:
            # Placeholder - in production, LLM would summarize
            summary = "Abstract layer would contain LLM-generated summary tailored to agent role"
            abstract_layer = self.layer_builder.build_abstract_layer(
                summary, purpose=purpose
            )
            layered_context.add_layer(abstract_layer)
            layered_context.confidence_scores[ContextLayerType.ABSTRACT] = 0.7

        logger.info(f"Assembled context with {len(layered_context.layers)} layers")
        return layered_context

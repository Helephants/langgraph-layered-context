"""
Context enrichment with entities and relationships.
"""
from dataclasses import dataclass, field
from ..ingestion import DocumentChunk
from .entities import Entity, Relationship, EntityExtractor, DependencyBasedRelationshipExtractor
from ..utils.logging_util import get_logger

logger = get_logger("extraction.enrichment")


@dataclass
class EnrichedChunk:
    """A document chunk enriched with entities and relationships."""
    chunk: DocumentChunk
    entities: set[Entity] = field(default_factory=set)
    relationships: set[Relationship] = field(default_factory=set)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "content": self.chunk.content,
            "source": self.chunk.source,
            "entities": [
                {
                    "text": e.text,
                    "type": e.entity_type,
                    "id": e.entity_id,
                    "confidence": e.confidence,
                }
                for e in self.entities
            ],
            "relationships": [
                {
                    "source": r.source_entity.entity_id,
                    "target": r.target_entity.entity_id,
                    "type": r.relationship_type,
                    "confidence": r.confidence,
                }
                for r in self.relationships
            ],
            "metadata": self.chunk.metadata,
        }


class ChunkEnricher:
    """Enrich document chunks with entities and relationships."""

    def __init__(
        self,
        nlp_model: str = "en_core_web_md",
        min_entity_confidence: float = 0.5,
    ):
        """
        Initialize chunk enricher.

        Args:
            nlp_model: spaCy model name
            min_entity_confidence: Minimum entity confidence threshold
        """
        self.entity_extractor = EntityExtractor(nlp_model, min_entity_confidence)
        self.relationship_extractor = DependencyBasedRelationshipExtractor(nlp_model)

    async def enrich_chunk(self, chunk: DocumentChunk) -> EnrichedChunk:
        """
        Enrich a single chunk with entities and relationships.

        Args:
            chunk: Document chunk to enrich

        Returns:
            EnrichedChunk with extracted entities and relationships
        """
        # Extract entities
        entities = self.entity_extractor.extract_entities(chunk.content)

        # Extract relationships
        relationships = self.relationship_extractor.extract_relationships(
            chunk.content, entities
        )

        enriched = EnrichedChunk(
            chunk=chunk,
            entities=entities,
            relationships=relationships,
        )

        logger.debug(
            f"Enriched chunk {chunk.chunk_id}: "
            f"{len(entities)} entities, {len(relationships)} relationships"
        )

        return enriched

    async def enrich_chunks(
        self, chunks: list[DocumentChunk]
    ) -> list[EnrichedChunk]:
        """
        Enrich multiple chunks.

        Args:
            chunks: List of document chunks

        Returns:
            List of enriched chunks
        """
        enriched_chunks = []
        for chunk in chunks:
            enriched = await self.enrich_chunk(chunk)
            enriched_chunks.append(enriched)
        return enriched_chunks

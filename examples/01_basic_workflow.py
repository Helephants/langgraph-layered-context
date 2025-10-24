"""
Example 1: Basic workflow demonstrating layered context framework.

This example shows:
1. Ingesting documents using glob patterns
2. Extracting entities and relationships
3. Building a knowledge graph
4. Creating layered context
5. Using agents to query context with access control
"""
import asyncio
from pathlib import Path

from src.utils import setup_logger, get_config, set_config, FrameworkConfig
from src.ingestion import DocumentIngestionPipeline
from src.extraction import ChunkEnricher
from src.graph import NetworkXGraphStore
from src.context import ContextAssembler, ContextLayerType
from src.retrieval import ChromaVectorStore
from src.governance import (
    AccessController, User, AuditLogger, ProvenanceTracker
)
from src.graph import AccessLevel
from src.agents import Agent, AgentRole, AgentCapability, AgentOrchestrator


async def main():
    """Run the basic workflow example."""
    # Setup
    setup_logger(level="INFO")
    logger = __import__("loguru").logger

    logger.info("=" * 60)
    logger.info("Layered Context Framework - Basic Workflow Example")
    logger.info("=" * 60)

    # 1. Configure framework
    logger.info("\n1. Configuring framework...")
    config = get_config()
    logger.info(f"Max workers: {config.ingestion.max_workers}")
    logger.info(f"Embedding model: {config.retrieval.embedding_model}")

    # 2. Create sample documents for testing
    logger.info("\n2. Creating sample documents...")
    sample_dir = Path("sample_docs")
    sample_dir.mkdir(exist_ok=True)

    # Create a sample markdown document
    sample_md = sample_dir / "example.md"
    sample_md.write_text("""# Knowledge Graph Example

## Entities and Relationships

Alice and Bob are software engineers at TechCorp. They work on machine learning projects.

Alice specializes in natural language processing, while Bob focuses on computer vision.

The company was founded in 2015 and is located in San Francisco.

Their main product is an AI platform used by enterprises worldwide.
""")

    logger.info(f"Created sample document: {sample_md}")

    # 3. Ingest documents
    logger.info("\n3. Ingesting documents...")
    pipeline = DocumentIngestionPipeline(max_workers=4)
    chunks = []

    async for chunk in pipeline.ingest_from_directory(
        sample_dir,
        file_types=["md", "txt"],
        chunk_size=256,
    ):
        chunks.append(chunk)
        logger.debug(f"Ingested chunk: {chunk.chunk_id}")

    logger.info(f"Total chunks ingested: {len(chunks)}")

    # 4. Extract entities and relationships
    logger.info("\n4. Extracting entities and relationships...")
    enricher = ChunkEnricher(nlp_model="en_core_web_md")
    enriched_chunks = await enricher.enrich_chunks(chunks)

    total_entities = sum(len(ec.entities) for ec in enriched_chunks)
    total_relationships = sum(len(ec.relationships) for ec in enriched_chunks)

    logger.info(f"Extracted entities: {total_entities}")
    logger.info(f"Extracted relationships: {total_relationships}")

    for chunk in enriched_chunks:
        for entity in chunk.entities:
            logger.debug(f"  - {entity.text} ({entity.entity_type})")

    # 5. Build knowledge graph
    logger.info("\n5. Building knowledge graph...")
    graph_store = NetworkXGraphStore()

    for chunk in enriched_chunks:
        graph_store.add_entities_and_relationships(
            chunk.entities,
            chunk.relationships,
        )

    stats = graph_store.get_statistics()
    logger.info(f"Graph statistics: {stats}")

    # 6. Index with vector embeddings
    logger.info("\n6. Indexing documents with vector embeddings...")
    vector_store = ChromaVectorStore(
        collection_name="layered-context-demo",
        persist_dir=str(Path(".cache/chroma")),
    )
    vector_store.add_chunks(chunks)
    logger.info(f"Indexed {len(chunks)} chunks")

    # 7. Set up governance
    logger.info("\n7. Setting up governance and access control...")
    access_controller = AccessController()
    audit_logger = AuditLogger(
        log_file=Path(".cache/audit/audit.log")
    )
    provenance_tracker = ProvenanceTracker()

    # Create users with different access levels
    admin_user = User(
        user_id="admin-001",
        username="admin",
        access_level=AccessLevel.CONFIDENTIAL,
    )
    researcher_user = User(
        user_id="researcher-001",
        username="researcher",
        access_level=AccessLevel.INTERNAL,
    )

    access_controller.register_user(admin_user)
    access_controller.register_user(researcher_user)

    logger.info(f"Registered users: admin, researcher")

    # 8. Assemble layered context
    logger.info("\n8. Assembling layered context...")
    assembler = ContextAssembler()

    context = await assembler.assemble_context(
        chunks=chunks,
        enriched_chunks=enriched_chunks,
        graph_store=graph_store,
        purpose="General knowledge exploration",
        agent_role="researcher",
        preferred_layers=list(ContextLayerType),
    )

    logger.info(f"Context layers created: {len(context.layers)}")
    for layer_type, layer in context.layers.items():
        logger.info(f"  - {layer_type.name}: {len(layer.content)} chars")

    # 9. Set up agents
    logger.info("\n9. Setting up agents...")
    orchestrator = AgentOrchestrator(
        access_controller=access_controller,
        audit_logger=audit_logger,
    )

    # Create researcher agent
    researcher_agent = Agent(
        name="Research Agent",
        role=AgentRole.RESEARCHER,
        capabilities=[
            AgentCapability.RETRIEVE_CONTEXT,
            AgentCapability.ANALYZE_ENTITIES,
        ],
        preferred_layers=[
            ContextLayerType.RAW,
            ContextLayerType.ENTITY,
            ContextLayerType.GRAPH,
        ],
    )

    orchestrator.register_agent(researcher_agent)
    logger.info(f"Registered agent: {researcher_agent.name}")

    # 10. Test access control
    logger.info("\n10. Testing access control...")
    admin_raw_access = access_controller.can_access(
        "admin-001", "raw"
    )
    researcher_abstract_access = access_controller.can_access(
        "researcher-001", "abstract"
    )

    logger.info(f"Admin can access raw layer: {admin_raw_access}")
    logger.info(f"Researcher can access abstract layer: {researcher_abstract_access}")

    # 11. Export results
    logger.info("\n11. Exporting framework data...")

    # Export graph
    graph_export = graph_store.export_to_dict()
    logger.info(f"Graph export: {len(graph_export['entities'])} entities")

    # Export context
    context_export = context.to_dict()
    logger.info(f"Context export: {len(context_export['layers'])} layers")

    logger.info("\n" + "=" * 60)
    logger.info("Example completed successfully!")
    logger.info("=" * 60)

    # Cleanup
    import shutil
    shutil.rmtree(sample_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())

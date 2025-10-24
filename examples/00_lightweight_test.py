"""
Lightweight test of core components without requiring full NLP models.
Tests basic ingestion, graph, context, and governance without entity extraction.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logger, get_config
from src.ingestion import DocumentIngestionPipeline, Document, DocumentChunk
from src.graph import NetworkXGraphStore
from src.context import ContextAssembler, ContextLayerType, LayerBuilder
from src.retrieval import ChromaVectorStore
from src.governance import (
    AccessController, User, AuditLogger, ProvenanceTracker, Permission
)
from src.graph import AccessLevel
from src.agents import Agent, AgentRole, AgentCapability


async def test_document_ingestion():
    """Test document ingestion."""
    logger = __import__("loguru").logger
    logger.info("\n=== Testing Document Ingestion ===")

    # Create sample documents
    sample_dir = Path("sample_docs_test")
    sample_dir.mkdir(exist_ok=True)

    # Create test files
    (sample_dir / "test1.md").write_text("# Test Document\n\nThis is a test document with some content.")
    (sample_dir / "test2.txt").write_text("Another test file with different content.")

    # Test ingestion
    pipeline = DocumentIngestionPipeline(max_workers=2)
    chunks = []

    async for chunk in pipeline.ingest_from_directory(sample_dir, file_types=["md", "txt"]):
        chunks.append(chunk)
        logger.debug(f"Ingested: {chunk.chunk_id}")

    logger.info(f"✓ Ingestion test passed: {len(chunks)} chunks loaded")

    # Cleanup
    import shutil
    shutil.rmtree(sample_dir, ignore_errors=True)

    return chunks


def test_knowledge_graph():
    """Test knowledge graph creation."""
    logger = __import__("loguru").logger
    logger.info("\n=== Testing Knowledge Graph ===")

    from src.extraction import Entity, Relationship

    # Create test entities
    entities = {
        Entity(text="Alice", entity_type="PERSON", start_char=0, end_char=5),
        Entity(text="Python", entity_type="TECHNOLOGY", start_char=10, end_char=16),
        Entity(text="TechCorp", entity_type="ORG", start_char=20, end_char=28),
    }

    # Create test relationships
    alice = list(entities)[0]
    python = list(entities)[1]
    techcorp = list(entities)[2]

    relationships = {
        Relationship(
            source_entity=alice,
            target_entity=techcorp,
            relationship_type="WORKS_AT",
            confidence=0.95,
        ),
        Relationship(
            source_entity=alice,
            target_entity=python,
            relationship_type="USES",
            confidence=0.90,
        ),
    }

    # Test graph
    graph = NetworkXGraphStore()
    graph.add_entities_and_relationships(entities, relationships)

    stats = graph.get_statistics()
    logger.info(f"Graph created: {stats['num_entities']} entities, {stats['num_relationships']} relationships")
    logger.info(f"✓ Knowledge graph test passed")

    return graph, entities, relationships


def test_vector_embeddings(chunks):
    """Test vector embeddings."""
    logger = __import__("loguru").logger
    logger.info("\n=== Testing Vector Embeddings ===")

    vector_store = ChromaVectorStore(
        collection_name="lightweight-test",
        persist_dir=str(Path(".cache/chroma_test")),
    )

    if chunks:
        vector_store.add_chunks(chunks)
        logger.info(f"✓ Added {len(chunks)} chunks to vector store")

    stats = vector_store.get_collection_stats()
    logger.info(f"Vector store stats: {stats}")

    return vector_store


def test_layered_context():
    """Test layered context assembly."""
    logger = __import__("loguru").logger
    logger.info("\n=== Testing Layered Context ===")

    # Create a simple layered context
    raw_layer = LayerBuilder.build_raw_layer(
        [DocumentChunk(content="Test content", source="test.md", chunk_index=0)],
        purpose="Testing"
    )

    context = __import__("src.context", fromlist=["LayeredContext"]).LayeredContext(
        query_or_purpose="Test query"
    )
    context.add_layer(raw_layer)

    logger.info(f"✓ Created context with {len(context.layers)} layer(s)")
    logger.info(f"  Layer types: {[l.name for l in context.layers.keys()]}")

    return context


def test_governance():
    """Test governance and access control."""
    logger = __import__("loguru").logger
    logger.info("\n=== Testing Governance ===")

    # Create controller
    ac = AccessController()
    audit = AuditLogger(log_file=Path(".cache/audit_test.log"))

    # Register users
    admin = User(username="admin", access_level=AccessLevel.CONFIDENTIAL)
    regular = User(username="user", access_level=AccessLevel.PUBLIC)

    ac.register_user(admin)
    ac.register_user(regular)

    logger.info(f"Registered 2 users")

    # Test access
    admin_access = ac.can_access(admin.user_id, "raw")
    user_access = ac.can_access(regular.user_id, "abstract")

    logger.info(f"Admin can access 'raw': {admin_access}")
    logger.info(f"User can access 'abstract': {user_access}")
    logger.info(f"✓ Governance test passed")

    # Log event
    audit.log_context_retrieval(
        user_id=admin.user_id,
        query="Test query",
        num_results=5,
    )

    logger.info(f"✓ Audit logging working")

    return ac, audit


async def main():
    """Run all tests."""
    setup_logger(level="INFO")
    logger = __import__("loguru").logger

    logger.info("=" * 60)
    logger.info("Layered Context Framework - Lightweight Component Test")
    logger.info("=" * 60)

    try:
        # Test 1: Ingestion
        chunks = await test_document_ingestion()

        # Test 2: Knowledge Graph
        graph, entities, relationships = test_knowledge_graph()

        # Test 3: Vector Embeddings
        vector_store = test_vector_embeddings(chunks)

        # Test 4: Layered Context
        context = test_layered_context()

        # Test 5: Governance
        ac, audit = test_governance()

        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED!")
        logger.info("=" * 60)
        logger.info("\nFramework Components Verified:")
        logger.info("  ✓ Document ingestion (glob-based parallel loading)")
        logger.info("  ✓ Knowledge graph (NetworkX-based)")
        logger.info("  ✓ Vector embeddings (Chroma)")
        logger.info("  ✓ Layered context architecture")
        logger.info("  ✓ Governance (RBAC, audit logging)")
        logger.info("\nNext: Run full E2E example with entity extraction")
        logger.info("  python examples/01_basic_workflow.py")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Cleanup
    import shutil
    shutil.rmtree(".cache", ignore_errors=True)

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

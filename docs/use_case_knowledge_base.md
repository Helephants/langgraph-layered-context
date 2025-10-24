# Use Case: Building Searchable Knowledge Bases

## Overview

The Layered Context Framework excels at creating intelligent, searchable knowledge bases from diverse document collections. Unlike traditional full-text search, the framework provides **context-aware retrieval** with semantic understanding, entity-based navigation, and role-based access control.

### Key Benefits

- **Multi-Layer Search**: Users can search at different abstraction levels (raw text, entities, relationships, summaries)
- **Entity-Based Navigation**: Discover related documents through entity relationships
- **Access Control**: Fine-grained permissions for sensitive information
- **Audit Trails**: Complete logging of who accessed what
- **Adaptive Results**: Context layers adapt to user expertise and role

### Ideal For

- Corporate documentation portals
- Research paper management systems
- Technical reference libraries
- Customer knowledge bases
- Internal wikis with governance requirements

---

## Architecture for Knowledge Bases

```
Document Sources
  |
  v
Parallel Ingestion (glob-based)
  |
  v
Layer 1: Raw Text Chunks
  |
  v
Layer 2: Entity-Enriched Context
  |
  v
Layer 3: Relationship Graph
  |
  v
Layer 4: Abstracted Summaries
  |
  +------+------+
         |
  Vector Search + Graph Traversal
         |
  Access Control & Audit
         |
  Agent-Based Query Resolution
```

---

## Step-by-Step Implementation

### Step 1: Setup and Configuration

```python
from pathlib import Path
from src.utils import FrameworkConfig, set_config, setup_logger
from src.governance import AccessController, AuditLogger, User, AccessLevel

# Configure framework
setup_logger(level="INFO")

config = FrameworkConfig(
    ingestion=IngestionConfig(
        max_workers=8,
        chunk_size=512,
        chunk_overlap=50,
    ),
    retrieval=RetrievalConfig(
        embedding_model="all-MiniLM-L6-v2",
        top_k=10,
        use_hybrid_ranking=True,
    ),
    governance=GovernanceConfig(
        enable_audit=True,
        enforce_access_control=True,
    ),
)

set_config(config)

# Setup governance
access_controller = AccessController()
audit_logger = AuditLogger(log_file=Path(".cache/audit.log"))

# Register users with different roles
admin_user = User(
    username="admin",
    access_level=AccessLevel.CONFIDENTIAL,
)
researcher_user = User(
    username="researcher",
    access_level=AccessLevel.INTERNAL,
)

access_controller.register_user(admin_user)
access_controller.register_user(researcher_user)
```

### Step 2: Ingest Documents at Scale

```python
import asyncio
from src.ingestion import DocumentIngestionPipeline

async def ingest_knowledge_base():
    """Ingest all documents from knowledge base."""
    pipeline = DocumentIngestionPipeline(max_workers=8)

    chunks = []

    # Ingest from directory tree
    async for chunk in pipeline.ingest_from_directory(
        directory="knowledge_base/",
        file_types=["pdf", "md", "txt"],
        chunk_size=512,
        chunk_overlap=50,
    ):
        chunks.append(chunk)
        print(f"Loaded: {chunk.source}")

    return chunks

# Run ingestion
chunks = asyncio.run(ingest_knowledge_base())
print(f"Total chunks: {len(chunks)}")
```

### Step 3: Extract Entities and Build Knowledge Graph

```python
from src.extraction import ChunkEnricher
from src.graph import NetworkXGraphStore

async def build_knowledge_graph(chunks):
    """Extract entities and build knowledge graph."""
    enricher = ChunkEnricher(nlp_model="en_core_web_md")

    # Extract entities from all chunks
    enriched = await enricher.enrich_chunks(chunks)

    # Build graph
    graph = NetworkXGraphStore()

    for chunk in enriched:
        graph.add_entities_and_relationships(
            chunk.entities,
            chunk.relationships,
        )

    # Print statistics
    stats = graph.get_statistics()
    print(f"Graph built with:")
    print(f"  - {stats['num_entities']} entities")
    print(f"  - {stats['num_relationships']} relationships")
    print(f"  - {stats['density']:.3f} density")

    return graph, enriched

# Build graph
graph, enriched_chunks = asyncio.run(build_knowledge_graph(chunks))
```

### Step 4: Index with Vector Search

```python
from src.retrieval import ChromaVectorStore

def index_knowledge_base(chunks):
    """Index chunks with vector embeddings."""
    vector_store = ChromaVectorStore(
        collection_name="knowledge-base",
        persist_dir=".cache/chroma",
    )

    vector_store.add_chunks(chunks)

    stats = vector_store.get_collection_stats()
    print(f"Vector store indexed: {stats['num_vectors']} vectors")

    return vector_store

vector_store = index_knowledge_base(chunks)
```

### Step 5: Create Layered Context for Query

```python
from src.context import ContextAssembler, ContextLayerType

async def query_knowledge_base(
    query: str,
    user_id: str,
    vector_store,
    graph,
    chunks,
    enriched_chunks,
):
    """Query knowledge base with layered context."""

    # Check access control
    if not access_controller.can_access(user_id, "raw"):
        print(f"Access denied for user {user_id}")
        return None

    # Vector search for relevant chunks
    search_results = vector_store.search(query, top_k=10)
    retrieved_chunks = [r.chunk for r in search_results]

    # Get enriched versions
    relevant_enriched = [
        ec for ec in enriched_chunks
        if ec.chunk.chunk_id in [c.chunk_id for c in retrieved_chunks]
    ]

    # Assemble layered context
    assembler = ContextAssembler()
    context = await assembler.assemble_context(
        chunks=retrieved_chunks,
        enriched_chunks=relevant_enriched,
        graph_store=graph,
        purpose=query,
        agent_role="researcher",
        preferred_layers=list(ContextLayerType),
    )

    # Audit the access
    audit_logger.log_context_retrieval(
        user_id=user_id,
        query=query,
        num_results=len(retrieved_chunks),
    )

    return context

# Example query
context = asyncio.run(query_knowledge_base(
    query="machine learning frameworks",
    user_id=researcher_user.user_id,
    vector_store=vector_store,
    graph=graph,
    chunks=chunks,
    enriched_chunks=enriched_chunks,
))
```

### Step 6: Present Results by Layer

```python
def present_results(context):
    """Present search results with layer-based information."""

    print("\n" + "="*60)
    print("SEARCH RESULTS - Layered View")
    print("="*60)

    # Layer 1: Raw text (for users who want original context)
    if context.get_layer(ContextLayerType.RAW):
        raw = context.get_layer(ContextLayerType.RAW)
        print(f"\n[Layer 1] Raw Documents ({len(raw.source_chunks)} sources)")
        print(f"Total context: {len(raw.content)} characters")

    # Layer 2: Entity-enriched (for researchers)
    if context.get_layer(ContextLayerType.ENTITY):
        entity = context.get_layer(ContextLayerType.ENTITY)
        print(f"\n[Layer 2] Entity-Enriched View")
        print(f"Key entities found: {len(entity.entities)}")
        for ent in list(entity.entities)[:5]:
            print(f"  - {ent.text} ({ent.entity_type})")

    # Layer 3: Graph relationships (for analysts)
    if context.get_layer(ContextLayerType.GRAPH):
        graph_layer = context.get_layer(ContextLayerType.GRAPH)
        print(f"\n[Layer 3] Relationships & Structure")
        print(f"Relationships found: {len(graph_layer.relationships)}")
        for rel in list(graph_layer.relationships)[:3]:
            print(f"  - {rel.source_entity.text} --[{rel.relationship_type}]--> {rel.target_entity.text}")

    # Layer 4: Abstract summary (for executives)
    if context.get_layer(ContextLayerType.ABSTRACT):
        abstract = context.get_layer(ContextLayerType.ABSTRACT)
        print(f"\n[Layer 4] Executive Summary")
        print(abstract.content[:200] + "...")

present_results(context)
```

---

## Advanced Queries

### Entity-Based Navigation

```python
def find_related_documents(entity_text, graph, chunks):
    """Find documents related to an entity."""

    # Find entity in graph
    entity = graph.get_entity_by_text(entity_text)
    if not entity:
        return []

    # Get all relationships
    rels = graph.get_entity_relationships(entity.entity_id)

    # Find documents mentioning related entities
    related_docs = set()
    for rel in rels:
        # Search for documents with related entities
        for chunk in chunks:
            if (rel.target_entity.text.lower() in chunk.content.lower() or
                rel.source_entity.text.lower() in chunk.content.lower()):
                related_docs.add(chunk.source)

    return list(related_docs)

# Example
related = find_related_documents("Python", graph, chunks)
print(f"Documents related to Python: {len(related)}")
```

### Multi-Agent Queries

```python
from src.agents import Agent, AgentRole, AgentCapability, AgentOrchestrator

# Create specialized agents
researcher_agent = Agent(
    name="Research Assistant",
    role=AgentRole.RESEARCHER,
    capabilities=[
        AgentCapability.RETRIEVE_CONTEXT,
        AgentCapability.ANALYZE_ENTITIES,
        AgentCapability.EXTRACT_RELATIONSHIPS,
    ],
    preferred_layers=[
        ContextLayerType.RAW,
        ContextLayerType.ENTITY,
        ContextLayerType.GRAPH,
    ],
)

executive_agent = Agent(
    name="Executive Summary",
    role=AgentRole.ANALYST,
    capabilities=[AgentCapability.SUMMARIZE],
    preferred_layers=[ContextLayerType.ABSTRACT],
)

# Register agents
orchestrator = AgentOrchestrator(
    access_controller=access_controller,
    audit_logger=audit_logger,
)
orchestrator.register_agent(researcher_agent)
orchestrator.register_agent(executive_agent)

# Execute agent-specific queries
# (Agent selects appropriate layers based on role)
```

---

## Real-World Example: Company Documentation Portal

```python
async def setup_company_kb():
    """Setup knowledge base for company documentation."""

    # Configure for company use
    config = FrameworkConfig(
        ingestion=IngestionConfig(max_workers=16),
        governance=GovernanceConfig(
            enable_audit=True,
            enforce_access_control=True,
        ),
    )
    set_config(config)

    # Ingest all company docs
    pipeline = DocumentIngestionPipeline()
    chunks = []

    async for chunk in pipeline.ingest_from_directory(
        "company_docs/",
        file_types=["pdf", "md", "docx"],
    ):
        chunks.append(chunk)

    # Build knowledge graph
    enricher = ChunkEnricher()
    enriched = await enricher.engest_chunks(chunks)

    graph = NetworkXGraphStore()
    for chunk in enriched:
        graph.add_entities_and_relationships(
            chunk.entities, chunk.relationships
        )

    # Index with vectors
    vector_store = ChromaVectorStore()
    vector_store.add_chunks(chunks)

    print(f"Company KB ready:")
    print(f"  Documents: {len(chunks)} chunks")
    print(f"  Entities: {len(graph.entities)}")
    print(f"  Searchable vectors: {vector_store.get_collection_stats()['num_vectors']}")

    return chunks, graph, vector_store

# Run setup
chunks, graph, vector_store = asyncio.run(setup_company_kb())
```

---

## Performance Optimization

### Caching Strategy

```python
from pathlib import Path
import pickle

def cache_graph(graph, cache_path):
    """Cache knowledge graph for faster startup."""
    with open(cache_path, 'wb') as f:
        pickle.dump(graph.export_to_dict(), f)
    print(f"Graph cached to {cache_path}")

def load_cached_graph(cache_path):
    """Load cached graph."""
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    # Rebuild graph from exported data
    graph = NetworkXGraphStore()
    # ... populate graph from data
    return graph

# Cache large graphs
cache_graph(graph, ".cache/kg_backup.pkl")
```

### Parallel Search

```python
import asyncio

async def parallel_search(query, vector_store, graph, num_sources=3):
    """Search both vector and graph in parallel."""

    # Vector search
    async def vec_search():
        return vector_store.search(query, top_k=10)

    # Graph-based search (simulated)
    async def graph_search():
        # Extract entities from query
        # Find related entities in graph
        # Return weighted results
        return []

    # Run in parallel
    results = await asyncio.gather(vec_search(), graph_search())

    return results[0] + results[1]  # Merge results
```

---

## Access Control Scenarios

### Scenario 1: Research Team Access

```python
# Create research team with internal access
research_team = User(
    username="research_team",
    access_level=AccessLevel.INTERNAL,
)

access_controller.register_user(research_team)

# Can access raw and entity layers
assert access_controller.can_access(
    research_team.user_id,
    resource_type="raw",
)
assert access_controller.can_access(
    research_team.user_id,
    resource_type="entity",
)
# But not confidential summaries
assert not access_controller.can_access(
    research_team.user_id,
    resource_type="abstract",
)
```

### Scenario 2: Confidential Information

```python
from src.governance import AccessRule
from datetime import datetime, timedelta

# Create rule: limit access to confidential documents
rule = AccessRule(
    user_id=researcher_user.user_id,
    resource_type="entity",
    resource_id="SENSITIVE_PROJECT",
    allowed=False,
    expires_at=datetime.utcnow() + timedelta(days=30),
)

access_controller.add_rule(rule)

# This will be denied
assert not access_controller.can_access(
    researcher_user.user_id,
    "entity",
    "SENSITIVE_PROJECT",
)
```

---

## Troubleshooting

### Issue: Slow Vector Search

**Cause**: Large collection with many chunks
**Solution**:
```python
# Use metadata filtering to narrow results
results = vector_store.search(
    query,
    top_k=5,
    where_filter={"source": "2024"}  # Filter by date
)
```

### Issue: Memory Usage Growing

**Cause**: Large knowledge graph in memory
**Solution**:
```python
# Use Neo4j for large graphs
from src.graph import Neo4jGraphStore

graph = Neo4jGraphStore(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
)
```

### Issue: Missing Entities

**Cause**: spaCy model not recognizing domain terms
**Solution**:
```python
# Add custom entities
from spacy.training import Example

# Train custom NER on domain-specific data
# Then use in extraction pipeline
enricher = ChunkEnricher(nlp_model="custom_model")
```

---

## Conclusion

The Layered Context Framework provides a powerful foundation for building production-grade knowledge management systems. By combining vector search, knowledge graphs, and governance capabilities, organizations can build intelligent, auditable knowledge bases that serve users at all levels of expertise.

See also:
- [Code Review Use Case](use_case_code_review.md)
- [Refactoring Use Case](use_case_refactoring.md)
- [Architecture Guide](architecture.md)

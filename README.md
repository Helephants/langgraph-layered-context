# Layered Context Framework for RAG Systems

A production-ready Python framework implementing hierarchical, governed context delivery for Retrieval-Augmented Generation (RAG) systems. Inspired by the "Layered Context" approach from Towards Data Science, this framework goes beyond traditional RAG by organizing information into semantic layers, enabling adaptive context for multi-agent systems.

## Overview

Traditional RAG systems retrieve flat text snippets based on vector similarity. The **Layered Context Framework** structures context hierarchically:

- **Layer 1: Raw** - Basic text chunks (baseline RAG)
- **Layer 2: Entity** - Entity-enriched chunks with NER annotations
- **Layer 3: Graph** - Graph-based context with entity relationships
- **Layer 4: Abstract** - Abstracted summaries tailored to agent purpose

Each layer provides different types of context appropriate for different agent roles, enabling:
- Adaptive context delivery based on agent purpose
- Explainable retrieval with provenance tracking
- Enterprise governance (access control, audit trails)
- Multi-agent orchestration with LangGraph
- High-speed parallel document ingestion

## Architecture

```
AGENTS (LangGraph)
    |
    +----- Layer 4 -----+
    |     (Abstract)    |
    |                   |
    +----- Layer 3 -----+
    |      (Graph)      |
    |                   |
    +----- Layer 2 -----+
    |     (Entity)      |
    |                   |
    +----- Layer 1 -----+
          (Raw)

            |
     +------+------+
     |             |
Knowledge Graph  Vector Search
 (NetworkX)      (Chroma)
     |             |
     +------+------+
            |
   ENTITY EXTRACTION
    - NER Extraction
    - Entity Linking
    - Relationships
            |
   DOCUMENT INGESTION
    - PDF/MD/TXT
    - SQL/MongoDB
    - Parallel Loading
```

## Features

### Document Ingestion
- High-speed parallel file reading using glob patterns
- Support for PDF, Markdown, and plain text files
- SQL and MongoDB connectors for structured data
- Configurable chunking with overlap

### Entity & Relationship Extraction
- spaCy-based Named Entity Recognition (NER)
- Dependency parsing for relationship extraction
- Co-reference resolution support
- Configurable confidence thresholds

### Knowledge Graph
- NetworkX-based graph storage (can extend to Neo4j)
- Entity and relationship management
- Graph traversal and path finding
- Semantic layers for governance

### Layered Context Architecture
- 4-tier context hierarchy
- Adaptive context assembly
- Purpose-driven context selection
- Confidence scoring per layer

### Vector Embeddings & Search
- Sentence-Transformers for embeddings
- Chroma for scalable vector search
- Hybrid ranking (semantic + graph-based)
- Metadata filtering

### Governance & Audit
- Role-based access control (RBAC)
- Fine-grained layer permissions
- Complete audit trail logging
- Provenance tracking for entities
- Session management

### Multi-Agent Orchestration
- LangGraph-based agent framework
- Multiple agent roles (Researcher, Analyzer, Summarizer, etc.)
- Capability-based agent routing
- Autonomous layer selection per agent

## Installation

### Requirements
- Python 3.11+
- Virtual environment (uv recommended)

### Setup

```bash
# Clone and enter directory
cd context-rag

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies with uv
uv pip install -e .

# Download spaCy model
python -m spacy download en_core_web_md
```

## Quick Start

### 1. Basic Document Ingestion

```python
from src.ingestion import DocumentIngestionPipeline

# Create pipeline
pipeline = DocumentIngestionPipeline(max_workers=4)

# Ingest documents from directory
async for chunk in pipeline.ingest_from_directory(
    "documents/",
    file_types=["md", "txt", "pdf"],
    chunk_size=512,
):
    print(f"Chunk: {chunk.chunk_id}")
```

### 2. Extract Entities and Build Graph

```python
from src.extraction import ChunkEnricher
from src.graph import NetworkXGraphStore

# Extract entities
enricher = ChunkEnricher(nlp_model="en_core_web_md")
enriched = await enricher.enrich_chunks(chunks)

# Build graph
graph = NetworkXGraphStore()
for chunk in enriched:
    graph.add_entities_and_relationships(
        chunk.entities,
        chunk.relationships,
    )
```

### 3. Create Layered Context

```python
from src.context import ContextAssembler, ContextLayerType
from src.retrieval import ChromaVectorStore

# Index with vector store
vector_store = ChromaVectorStore()
vector_store.add_chunks(chunks)

# Assemble context
assembler = ContextAssembler()
context = await assembler.assemble_context(
    chunks=chunks,
    enriched_chunks=enriched,
    graph_store=graph,
    purpose="Research on AI applications",
    agent_role="researcher",
    preferred_layers=list(ContextLayerType),
)

# Access layers
raw_layer = context.get_layer(ContextLayerType.RAW)
entity_layer = context.get_layer(ContextLayerType.ENTITY)
```

### 4. Multi-Agent Orchestration

```python
from src.agents import Agent, AgentRole, AgentOrchestrator
from src.governance import AccessController, AuditLogger

# Setup governance
access_controller = AccessController()
audit_logger = AuditLogger()

# Create orchestrator
orchestrator = AgentOrchestrator(
    access_controller=access_controller,
    audit_logger=audit_logger,
)

# Register agents
researcher = Agent(
    name="Research Agent",
    role=AgentRole.RESEARCHER,
    capabilities=[AgentCapability.RETRIEVE_CONTEXT],
)
orchestrator.register_agent(researcher)

# Execute agent
state = await orchestrator.execute_agent(
    agent_id=researcher.agent_id,
    query="What are the key entities in the documents?",
)
```

### 5. Access Control & Audit

```python
from src.governance import User, AccessLevel

# Create users with different access levels
admin = User(
    username="admin",
    access_level=AccessLevel.CONFIDENTIAL,
)

researcher = User(
    username="researcher",
    access_level=AccessLevel.INTERNAL,
)

access_controller.register_user(admin)
access_controller.register_user(researcher)

# Check access
can_access = access_controller.can_access(
    researcher.user_id,
    resource_type="layer",
    resource_id="entity",
)
```

## Project Structure

```
src/
├── ingestion/           # Document loading & parsing
│   ├── loaders.py       # File readers (PDF, MD, TXT)
│   └── connectors.py    # Database connectors
├── extraction/          # Entity & relationship extraction
│   ├── entities.py      # NER and entity linking
│   └── enrichment.py    # Context enrichment
├── graph/               # Knowledge graph management
│   ├── store.py         # NetworkX graph storage
│   └── layers.py        # Semantic layers
├── context/             # Layered context architecture
│   └── layers.py        # 4-layer context system
├── retrieval/           # Retrieval & ranking
│   └── embeddings.py    # Vector embeddings (Chroma)
├── governance/          # Access control & audit
│   ├── access_control.py # RBAC system
│   └── audit.py         # Audit logging & provenance
├── agents/              # Multi-agent orchestration
│   └── orchestrator.py  # LangGraph integration
└── utils/               # Configuration & logging
    ├── config.py        # Configuration management
    └── logging_util.py  # Logging utilities

examples/
└── 01_basic_workflow.py # Complete working example

tests/
└── (Test files for all components)
```

## Configuration

Configure the framework via environment variables or directly:

```python
from src.utils import FrameworkConfig, set_config

config = FrameworkConfig(
    ingestion=IngestionConfig(max_workers=8),
    retrieval=RetrievalConfig(
        embedding_model="all-MiniLM-L6-v2",
        top_k=10,
    ),
    governance=GovernanceConfig(
        enable_audit=True,
        enforce_access_control=True,
    ),
)

set_config(config)
```

## Key Concepts

### Layered Context
- Adaptivity: Context layers are selected based on agent purpose and role
- Explainability: Each layer provides different levels of detail and reasoning
- Scalability: Lightweight layers for fast retrieval, deep layers for analysis
- Governance: Fine-grained access control per layer

### Knowledge Graph
- Captures entity relationships beyond simple text similarity
- Enables path-based reasoning and relationship discovery
- Supports multiple semantic layers (raw, enriched, aggregated, inferred)

### Multi-Agent System
- Agents retrieve only the context layers they need
- Agents have different roles (researcher, analyzer, summarizer, etc.)
- Each agent has specific capabilities and layer preferences
- Orchestrator routes queries to appropriate agents

### Governance
- Access Control: Role-based layer permissions
- Audit Trail: Complete logging of access and operations
- Provenance: Track sources and processing steps
- Session Management: Multi-user concurrent access

## Performance Considerations

- Parallel Ingestion: Uses asyncio for high-speed parallel document loading
- Vector Search: Chroma provides fast similarity search with metadata filtering
- Graph Queries: NetworkX for in-memory graph operations
- Caching: Optional caching of embeddings and context layers
- Rust Acceleration: Optional Rust modules for I/O-intensive operations (future)

## Limitations & Future Work

### Current Limitations
- Graph size limited to available memory (for large datasets, integrate Neo4j)
- No distributed processing (consider Spark for large-scale deployments)
- Basic relationship extraction (rule-based dependency parsing)
- Layer 4 (Abstract) requires external LLM for real summarization

### Future Enhancements
- Neo4j integration for distributed graphs
- Rust modules for high-speed I/O
- LLM-based relationship extraction
- Real-time streaming document ingestion
- Graph visualization and exploration UI
- Advanced co-reference resolution
- Multi-hop reasoning and path planning
- Distributed agent deployment

## Documentation

- README.md - This file
- docs/use_case_knowledge_base.md - Building searchable knowledge bases
- docs/use_case_code_review.md - Code analysis and documentation generation
- docs/use_case_refactoring.md - Understanding codebases for refactoring
- docs/architecture.md - Detailed architecture guide
- docs/api_reference.md - Complete API documentation

## Examples

- examples/01_basic_workflow.py - Basic end-to-end workflow
- examples/02_knowledge_base.py - Knowledge base Q&A system
- examples/03_code_review.py - Code analysis and review
- examples/04_refactoring_assistant.py - Refactoring assistant

## Contributing

Contributions welcome! Areas for improvement:
- More sophisticated entity linking
- Better relationship extraction
- Performance optimizations
- Additional database connectors
- UI/visualization components
- Comprehensive testing

## References

- Beyond RAG - Towards Data Science: https://towardsdatascience.com/beyond-rag/
- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- spaCy NLP: https://spacy.io/
- Chroma Vector Database: https://www.trychroma.com/
- NetworkX Graph Library: https://networkx.org/

## License

MIT License

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check examples/ for working code
- Review documentation in docs/

---

**Built with Python and LangGraph for production-grade RAG systems.**

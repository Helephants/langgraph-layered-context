# Use Case: Code Review & Documentation Generation

## Overview

The Layered Context Framework excels at analyzing code repositories and generating comprehensive documentation. By treating code as a knowledge domain, the framework can:

- **Extract code structure** (functions, classes, imports, dependencies)
- **Map relationships** (call chains, inheritance, module dependencies)
- **Identify patterns** (code smells, architectural issues)
- **Generate documentation** from code + comments automatically
- **Track changes** with audit trails for review accountability

### Key Benefits

- **Comprehensive Code Understanding**: Multiple layers reveal different aspects of code
- **Automated Documentation**: Generate docs from code structure automatically
- **Code Quality Insights**: Identify patterns and potential issues
- **Dependency Analysis**: Understand module and library relationships
- **Change Tracking**: Audit trail of code reviews and modifications
- **Cross-Language Support**: Works with Python, JavaScript, Go, etc.

### Ideal For

- Code review automation
- Documentation generation for legacy systems
- Architecture analysis
- Code quality assessment
- Onboarding documentation
- Technical debt tracking

---

## Architecture for Code Analysis

```
Source Code Repository
  |
  v
Fast Parallel File Loading (glob)
  |
  v
Code Parsing (tree-sitter)
  |
  v
Layer 1: Raw Code Snippets
  |
  v
Layer 2: Extracted Symbols (functions, classes, imports)
  |
  v
Layer 3: Dependency Graph (call chains, relationships)
  |
  v
Layer 4: Architecture Summary (patterns, quality metrics)
  |
  +------+------+
         |
  Vector Search (code similarity)
  |
  Graph Analysis (dependencies)
  |
  Review Agent (issues, suggestions)
```

---

## Step-by-Step Implementation

### Step 1: Setup for Code Analysis

```python
from pathlib import Path
from src.utils import setup_logger, FrameworkConfig
from src.ingestion import DocumentIngestionPipeline

setup_logger(level="INFO")

# Configure for code analysis
config = FrameworkConfig(
    ingestion=IngestionConfig(
        max_workers=16,
        chunk_size=1024,  # Larger chunks for code
        chunk_overlap=100,
        supported_formats=["py", "js", "go", "java", "rs"],
    ),
)

# We'll need to extend the ingestion pipeline for code
# For now, use standard loaders with code-specific chunking
```

### Step 2: Ingest Code Repository

```python
import asyncio
from src.ingestion import DocumentIngestionPipeline, Document, DocumentChunk

async def ingest_codebase(repo_path):
    """Ingest entire codebase."""
    pipeline = DocumentIngestionPipeline(max_workers=16)

    chunks = []
    async for chunk in pipeline.ingest_from_directory(
        directory=repo_path,
        file_types=["py", "js", "ts"],
        chunk_size=1024,
        chunk_overlap=100,
    ):
        # Add code-specific metadata
        chunk.metadata.update({
            "type": "code",
            "language": chunk.source.split(".")[-1],
        })
        chunks.append(chunk)
        print(f"Loaded: {chunk.source}")

    print(f"Total code chunks: {len(chunks)}")
    return chunks

# Ingest codebase
chunks = asyncio.run(ingest_codebase("./src"))
```

### Step 3: Extract Code Symbols and Structure

```python
import re
from src.extraction import Entity, Relationship, ChunkEnricher

class CodeAnalyzer:
    """Analyze code structure without NLP."""

    @staticmethod
    def extract_python_symbols(code):
        """Extract functions, classes, imports from Python code."""
        entities = set()
        relationships = set()

        # Find classes
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, code):
            entity = Entity(
                text=match.group(1),
                entity_type="CLASS",
                start_char=match.start(),
                end_char=match.end(),
            )
            entities.add(entity)

        # Find functions
        func_pattern = r'def\s+(\w+)'
        for match in re.finditer(func_pattern, code):
            entity = Entity(
                text=match.group(1),
                entity_type="FUNCTION",
                start_char=match.start(),
                end_char=match.end(),
            )
            entities.add(entity)

        # Find imports
        import_pattern = r'(?:from|import)\s+([\w.]+)'
        for match in re.finditer(import_pattern, code):
            entity = Entity(
                text=match.group(1),
                entity_type="IMPORT",
                start_char=match.start(),
                end_char=match.end(),
            )
            entities.add(entity)

        return entities

    @staticmethod
    def extract_javascript_symbols(code):
        """Extract functions and classes from JavaScript."""
        entities = set()

        # Find functions
        func_pattern = r'(?:function|const|let)\s+(\w+)\s*(?:\=\s*\(|\()'
        for match in re.finditer(func_pattern, code):
            entity = Entity(
                text=match.group(1),
                entity_type="FUNCTION",
                start_char=match.start(),
                end_char=match.end(),
            )
            entities.add(entity)

        # Find classes
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, code):
            entity = Entity(
                text=match.group(1),
                entity_type="CLASS",
                start_char=match.start(),
                end_char=match.end(),
            )
            entities.add(entity)

        # Find imports
        import_pattern = r'import\s+(?:{.*?}|.*?)\s+from\s+[\'"](.+?)[\'"]'
        for match in re.finditer(import_pattern, code):
            entity = Entity(
                text=match.group(1),
                entity_type="IMPORT",
                start_char=match.start(),
                end_char=match.end(),
            )
            entities.add(entity)

        return entities

# Analyze code
analyzer = CodeAnalyzer()
code_entities = []

for chunk in chunks:
    if "python" in chunk.metadata.get("language", ""):
        entities = analyzer.extract_python_symbols(chunk.content)
    elif "javascript" in chunk.metadata.get("language", ""):
        entities = analyzer.extract_javascript_symbols(chunk.content)
    else:
        entities = set()

    code_entities.append((chunk, entities))

print(f"Extracted symbols from {len(code_entities)} files")
```

### Step 4: Build Code Dependency Graph

```python
from src.graph import NetworkXGraphStore

def build_code_graph(chunks, code_entities):
    """Build graph of code dependencies."""
    graph = NetworkXGraphStore()

    for chunk, entities in code_entities:
        # Add all entities
        graph.add_entities_and_relationships(entities, set())

        # Create relationships between entities in same file
        entity_list = list(entities)
        for i, entity1 in enumerate(entity_list):
            for entity2 in entity_list[i+1:]:
                # Create "DEFINED_IN_SAME_FILE" relationship
                rel = Relationship(
                    source_entity=entity1,
                    target_entity=entity2,
                    relationship_type="CO_LOCATED",
                    confidence=0.95,
                )
                graph.add_relationship(rel)

    # Find cross-file relationships (imports)
    for chunk, entities in code_entities:
        imports = [e for e in entities if e.entity_type == "IMPORT"]
        for imp in imports:
            # Find matching exported symbols
            for other_chunk, other_entities in code_entities:
                if other_chunk.source == chunk.source:
                    continue
                for entity in other_entities:
                    if entity.entity_type in ("FUNCTION", "CLASS"):
                        if imp.text.lower() in other_chunk.source.lower():
                            rel = Relationship(
                                source_entity=entity1,
                                target_entity=entity,
                                relationship_type="EXPORTED_BY",
                                confidence=0.8,
                            )
                            graph.add_relationship(rel)

    stats = graph.get_statistics()
    print(f"Code graph built:")
    print(f"  - {stats['num_entities']} symbols")
    print(f"  - {stats['num_relationships']} relationships")

    return graph

code_graph = build_code_graph(chunks, code_entities)
```

### Step 5: Index Code for Search

```python
from src.retrieval import ChromaVectorStore

def index_codebase(chunks):
    """Index code for similarity search."""
    vector_store = ChromaVectorStore(
        collection_name="codebase",
        persist_dir=".cache/code_vectors",
    )

    # Add code chunks with language metadata
    vector_store.add_chunks(chunks)

    stats = vector_store.get_collection_stats()
    print(f"Code indexed: {stats['num_vectors']} vectors")

    return vector_store

vector_store = index_codebase(chunks)
```

### Step 6: Generate Code Documentation

```python
from src.context import LayerBuilder, ContextAssembler, ContextLayerType

async def generate_code_documentation(chunks, code_entities, code_graph):
    """Generate comprehensive code documentation."""

    # Layer 1: Raw code snippets
    raw_layer = LayerBuilder.build_raw_layer(
        chunks, purpose="Source code reference"
    )

    # Layer 2: Extracted code structure
    symbols_doc = "\n".join([
        f"- {e.text} ({e.entity_type})"
        for chunk, entities in code_entities
        for e in entities
    ])

    from src.context import ContextLayer, ContextLayerType as CLT
    structure_layer = ContextLayer(
        layer_type=CLT.ENTITY,
        description="Extracted code symbols (functions, classes, imports)",
        content=f"## Code Structure\n\n{symbols_doc}",
    )

    # Layer 3: Dependency analysis
    deps_doc = "## Dependencies\n\n"
    imports = set()
    for chunk, entities in code_entities:
        for e in entities:
            if e.entity_type == "IMPORT":
                imports.add(e.text)

    deps_doc += "\n".join([f"- {imp}" for imp in sorted(imports)])

    dependency_layer = ContextLayer(
        layer_type=CLT.GRAPH,
        description="Code dependencies and relationships",
        content=deps_doc,
    )

    # Layer 4: Architecture summary
    summary = f"""
## Architecture Overview

- Total Files: {len(chunks)}
- Total Symbols: {code_graph.get_statistics()['num_entities']}
- Total Relationships: {code_graph.get_statistics()['num_relationships']}
- External Dependencies: {len(imports)}

## Code Quality Observations

- Module Cohesion: Based on symbol co-location
- Coupling: Measured by cross-file imports
- Complexity: Estimated from symbol count and relationships
"""

    summary_layer = ContextLayer(
        layer_type=CLT.ABSTRACT,
        description="High-level architecture and quality assessment",
        content=summary,
    )

    return {
        "raw": raw_layer,
        "structure": structure_layer,
        "dependencies": dependency_layer,
        "summary": summary_layer,
    }

# Generate documentation
docs = asyncio.run(generate_code_documentation(
    chunks, code_entities, code_graph
))

print("\nGenerated Documentation:")
for name, layer in docs.items():
    print(f"  - {name}: {layer.description}")
```

### Step 7: Code Review Analysis

```python
def analyze_code_quality(code_graph, chunks):
    """Analyze code for common issues."""

    issues = []

    stats = code_graph.get_statistics()

    # Check for over-complex modules
    for chunk in chunks:
        # Simple metric: lines per file
        lines = len(chunk.content.split('\n'))
        if lines > 500:
            issues.append({
                "severity": "warning",
                "file": chunk.source,
                "issue": f"Large file ({lines} lines)",
                "suggestion": "Consider breaking into smaller modules",
            })

    # Check for isolated functions (no dependencies)
    isolated = []
    for entity_id, entity in code_graph.entities.items():
        rels = code_graph.get_entity_relationships(entity_id)
        if len(rels) == 0 and entity.entity_type in ("FUNCTION", "CLASS"):
            isolated.append(entity.text)

    if isolated:
        issues.append({
            "severity": "info",
            "issue": f"Found {len(isolated)} isolated symbols",
            "symbols": isolated[:5],
            "suggestion": "Check if these are utility functions or dead code",
        })

    # Check for circular dependencies
    # (simplified - would need cycle detection in graph)

    return issues

issues = analyze_code_quality(code_graph, chunks)

print("\nCode Quality Issues Found:")
for issue in issues:
    print(f"\n[{issue.get('severity').upper()}] {issue['issue']}")
    if 'suggestion' in issue:
        print(f"  Suggestion: {issue['suggestion']}")
```

---

## Real-World Example: Generate README

```python
async def generate_readme(chunks, code_entities, code_graph):
    """Generate README from code analysis."""

    # Extract project info
    readme_content = """# Project Documentation

Generated from automated code analysis.

## Project Structure

"""

    # List main modules
    modules = set()
    for chunk in chunks:
        # Extract directory from path
        parts = chunk.source.split("/")
        if len(parts) > 1:
            modules.add(parts[0])

    readme_content += "\n".join([f"- {m}" for m in sorted(modules)])

    # List main classes/functions
    readme_content += "\n\n## Main Components\n\n"

    classes = [e for chunk, ents in code_entities for e in ents if e.entity_type == "CLASS"]
    for cls in classes[:10]:
        readme_content += f"- `{cls.text}`: [Class]\n"

    functions = [e for chunk, ents in code_entities for e in ents if e.entity_type == "FUNCTION"]
    for func in functions[:10]:
        readme_content += f"- `{func.text}()`: [Function]\n"

    # Add dependency information
    readme_content += "\n## Dependencies\n\n"

    imports = set()
    for chunk, entities in code_entities:
        for e in entities:
            if e.entity_type == "IMPORT":
                imports.add(e.text)

    for imp in sorted(imports)[:15]:
        readme_content += f"- {imp}\n"

    readme_content += "\n---\n*This README was auto-generated from code analysis.*\n"

    return readme_content

# Generate README
readme = asyncio.run(generate_readme(chunks, code_entities, code_graph))

# Save it
with open("generated_README.md", "w") as f:
    f.write(readme)

print("README generated: generated_README.md")
```

---

## Integration with Code Review Workflow

```python
from src.agents import Agent, AgentRole, AgentCapability, AgentOrchestrator
from src.governance import AccessController, AuditLogger

# Create code review agent
reviewer_agent = Agent(
    name="Code Reviewer",
    role=AgentRole.ANALYST,
    capabilities=[
        AgentCapability.RETRIEVE_CONTEXT,
        AgentCapability.ANALYZE_ENTITIES,
        AgentCapability.EXTRACT_RELATIONSHIPS,
    ],
    preferred_layers=[
        ContextLayerType.ENTITY,
        ContextLayerType.GRAPH,
    ],
)

# Setup governance for code review
ac = AccessController()
audit = AuditLogger()

# Register developers
dev1 = User(username="dev1", access_level=AccessLevel.INTERNAL)
ac.register_user(dev1)

# Log code review event
audit.log_context_retrieval(
    user_id=dev1.user_id,
    query="Review module: auth.py",
    num_results=len(chunks),
    session_id="review_001",
)

print("Code review workflow initialized")
```

---

## Advanced Patterns

### Detecting Code Clones

```python
def find_similar_code(query_code, vector_store, top_k=5):
    """Find similar code patterns using vector search."""

    results = vector_store.search(query_code, top_k=top_k)

    print(f"Found {len(results)} similar code blocks:")
    for r in results:
        print(f"  {r.chunk.source}: {r.similarity_score:.2%} similar")

    return results

# Find similar patterns
similar = find_similar_code("def process_data(data):", vector_store)
```

### Change Impact Analysis

```python
def analyze_change_impact(modified_file, code_graph):
    """Analyze impact of changes to a file."""

    # Find all entities in modified file
    affected_entities = [e for e in code_graph.entities.values()
                        if e.metadata.get('source') == modified_file]

    # Find all dependent entities
    dependent = set()
    for entity in affected_entities:
        rels = code_graph.get_entity_relationships(entity.entity_id)
        for rel in rels:
            dependent.add(rel.target_entity.entity_id)

    print(f"Changes to {modified_file} affect:")
    print(f"  - {len(affected_entities)} entities directly")
    print(f"  - {len(dependent)} dependent entities")

    return affected_entities, dependent

# Analyze impact
modified = "src/core/auth.py"
affected, dependent = analyze_change_impact(modified, code_graph)
```

---

## Troubleshooting

### Issue: Missing Symbols

**Cause**: Language-specific patterns not recognized
**Solution**:
```python
# Add custom pattern for your language
CUSTOM_PATTERNS = {
    "go": r'func\s+(\w+)',
    "rust": r'fn\s+(\w+)',
    "java": r'(public|private)\s+(class|interface)\s+(\w+)',
}
```

### Issue: Incomplete Dependency Graph

**Cause**: Complex import patterns not recognized
**Solution**:
```python
# Use tree-sitter for better parsing
from tree_sitter import Language, Parser

PYTHON = Language("path/to/tree-sitter-python")
parser = Parser()
parser.set_language(PYTHON)

tree = parser.parse(code.encode('utf8'))
# Extract symbols from tree
```

---

## Conclusion

The Layered Context Framework makes it practical to analyze and document large codebases automatically. By combining code parsing, symbol extraction, dependency graphs, and vector search, development teams can:

- Maintain comprehensive, accurate documentation
- Perform thorough code reviews efficiently
- Understand complex codebases quickly
- Track code quality and dependencies
- Make informed refactoring decisions

See also:
- [Knowledge Base Use Case](use_case_knowledge_base.md)
- [Refactoring Use Case](use_case_refactoring.md)
- [Architecture Guide](architecture.md)

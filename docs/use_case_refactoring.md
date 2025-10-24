# Use Case: Code Refactoring & Architecture Modernization

## Overview

The Layered Context Framework is ideal for large-scale code refactoring by providing deep understanding of codebases. It helps refactoring teams:

- **Understand Current State**: Map existing architecture comprehensively
- **Identify Refactoring Opportunities**: Find tightly coupled code, duplication, dead code
- **Plan Changes Safely**: Analyze impact of proposed refactorings before implementation
- **Track Progress**: Maintain audit trail of refactoring decisions
- **Validate Results**: Ensure refactoring maintains functionality

### Key Benefits

- **Comprehensive Analysis**: Multiple layers reveal architecture, design, and implementation details
- **Impact Assessment**: Understand what breaks when you change code
- **Safe Refactoring**: Make decisions based on complete dependency information
- **Progress Tracking**: Audit trail of all changes and decisions
- **Team Communication**: Shared understanding of codebase
- **Risk Reduction**: Identify risky changes before implementation

### Ideal For

- Legacy system modernization
- Breaking monoliths into microservices
- Reducing technical debt
- Migrating frameworks/languages
- Improving code organization
- Decoupling tightly coupled systems

---

## Refactoring Framework

```
ANALYSIS PHASE
  |
  v
Ingest Current Codebase
  |
  v
Build Comprehensive Dependency Graph
  |
  v
Analyze Layers:
  - Raw code (what exists)
  - Symbols (what's available)
  - Dependencies (what's connected)
  - Metrics (complexity, coupling)
  |
  v
PLANNING PHASE
  |
  v
Identify Refactoring Candidates
  |
  v
Model Proposed Changes
  |
  v
Run Impact Analysis
  |
  v
Create Refactoring Plan
  |
  v
EXECUTION PHASE
  |
  v
Execute Refactoring
  |
  v
Run Tests
  |
  v
Update Graph
  |
  v
VALIDATION PHASE
  |
  v
Compare Before/After
  |
  v
Verify Functionality
  |
  v
Generate Report
```

---

## Phase 1: Analysis

### 1.1 Build Complete Code Map

```python
import asyncio
from src.ingestion import DocumentIngestionPipeline
from src.graph import NetworkXGraphStore
from src.extraction import Entity, Relationship
import re

async def analyze_codebase(repo_path):
    """Create complete map of current codebase."""

    # Ingest all code
    pipeline = DocumentIngestionPipeline(max_workers=16)
    chunks = []

    async for chunk in pipeline.ingest_from_directory(
        directory=repo_path,
        file_types=["py", "js", "ts", "go"],
        chunk_size=1024,
    ):
        chunks.append(chunk)

    print(f"Total files: {len(chunks)}")

    # Extract symbols
    graph = NetworkXGraphStore()

    for chunk in chunks:
        # Extract based on language
        if chunk.source.endswith('.py'):
            symbols = extract_python_symbols(chunk.content, chunk.source)
        elif chunk.source.endswith('.js'):
            symbols = extract_js_symbols(chunk.content, chunk.source)
        else:
            symbols = set()

        for symbol in symbols:
            graph.add_entity(symbol)

    # Find cross-file dependencies
    add_dependencies(chunks, graph)

    return chunks, graph

def extract_python_symbols(code, source):
    """Extract Python symbols."""
    symbols = set()

    # Classes
    for match in re.finditer(r'class\s+(\w+)', code):
        symbols.add(Entity(
            text=match.group(1),
            entity_type="CLASS",
            start_char=match.start(),
            end_char=match.end(),
            metadata={"source": source},
        ))

    # Functions
    for match in re.finditer(r'def\s+(\w+)', code):
        symbols.add(Entity(
            text=match.group(1),
            entity_type="FUNCTION",
            start_char=match.start(),
            end_char=match.end(),
            metadata={"source": source},
        ))

    return symbols

def add_dependencies(chunks, graph):
    """Find dependencies between symbols."""
    for chunk in chunks:
        for entity in graph.entities.values():
            if entity.metadata.get('source') == chunk.source:
                continue

            # Look for imports/usage
            if entity.text.lower() in chunk.content.lower():
                # Create relationship
                rel = Relationship(
                    source_entity=list(graph.entities.values())[0],
                    target_entity=entity,
                    relationship_type="USES",
                    confidence=0.7,
                )
                try:
                    graph.add_relationship(rel)
                except:
                    pass

chunks, graph = asyncio.run(analyze_codebase("./src"))
print(f"Graph built: {len(graph.entities)} symbols")
```

### 1.2 Calculate Metrics

```python
from typing import Dict

class CodeMetrics:
    """Calculate code quality and refactoring metrics."""

    @staticmethod
    def calculate_coupling(graph):
        """Measure coupling between modules."""
        coupling = {}

        for entity_id, entity in graph.entities.items():
            rels = graph.get_entity_relationships(entity_id)
            coupling[entity.text] = len(rels)

        return coupling

    @staticmethod
    def calculate_cohesion(chunks, graph):
        """Measure cohesion (internal consistency)."""
        cohesion = {}

        # Group entities by file
        by_file = {}
        for entity in graph.entities.values():
            source = entity.metadata.get('source', 'unknown')
            if source not in by_file:
                by_file[source] = []
            by_file[source].append(entity)

        # Calculate cohesion per file
        for source, entities in by_file.items():
            # Simple metric: internal vs external relationships
            internal_rels = 0
            external_rels = 0

            for entity in entities:
                rels = graph.get_entity_relationships(entity.entity_id)
                for rel in rels:
                    if rel.target_entity.metadata.get('source') == source:
                        internal_rels += 1
                    else:
                        external_rels += 1

            if internal_rels + external_rels > 0:
                cohesion[source] = internal_rels / (internal_rels + external_rels)

        return cohesion

    @staticmethod
    def find_circular_dependencies(graph):
        """Find circular dependencies that need breaking."""
        # Simplified: just find strongly connected components
        import networkx as nx

        # Convert to networkx graph
        G = nx.DiGraph()
        for ent_id, entity in graph.entities.items():
            G.add_node(entity.text)
            rels = graph.get_entity_relationships(ent_id)
            for rel in rels:
                G.add_edge(entity.text, rel.target_entity.text)

        # Find cycles
        try:
            cycles = list(nx.simple_cycles(G))
            return cycles
        except:
            return []

# Calculate metrics
metrics = CodeMetrics()
coupling = metrics.calculate_coupling(graph)
cohesion = metrics.calculate_cohesion(chunks, graph)
cycles = metrics.find_circular_dependencies(graph)

print(f"\nCode Metrics:")
print(f"  Average coupling: {sum(coupling.values())/len(coupling):.2f}")
print(f"  Average cohesion: {sum(cohesion.values())/len(cohesion):.2%}")
print(f"  Circular dependencies: {len(cycles)}")

# Show candidates for refactoring
print(f"\nRefactoring Candidates:")

# High coupling (tightly coupled modules)
high_coupling = sorted(coupling.items(), key=lambda x: x[1], reverse=True)[:5]
for name, coupling_val in high_coupling:
    print(f"  - {name}: coupling={coupling_val} (high dependency count)")

# Low cohesion (should be split)
low_cohesion = sorted(cohesion.items(), key=lambda x: x[1])[:5]
for name, cohesion_val in low_cohesion:
    print(f"  - {name}: cohesion={cohesion_val:.1%} (should be split)")

# Circular dependencies
if cycles:
    for cycle in cycles[:3]:
        print(f"  - Circular: {' -> '.join(cycle)}")
```

---

## Phase 2: Planning

### 2.1 Identify Refactoring Opportunities

```python
class RefactoringOpportunity:
    """Represents a refactoring opportunity."""

    def __init__(self, name, description, impact, effort, benefit):
        self.name = name
        self.description = description
        self.impact = impact  # "low", "medium", "high"
        self.effort = effort  # 1-10 scale
        self.benefit = benefit  # 1-10 scale
        self.affected_entities = []

    def priority(self):
        """Calculate priority (benefit/effort)."""
        return self.benefit / self.effort

def find_refactoring_opportunities(chunks, graph, coupling, cohesion, cycles):
    """Identify concrete refactoring opportunities."""

    opportunities = []

    # Opportunity 1: Break circular dependencies
    if cycles:
        opp = RefactoringOpportunity(
            name="Break Circular Dependencies",
            description=f"Found {len(cycles)} circular dependency cycles",
            impact="high",
            effort=7,
            benefit=9,
        )
        opp.affected_entities = [item for cycle in cycles for item in cycle]
        opportunities.append(opp)

    # Opportunity 2: Extract poorly cohesive modules
    for module, coh_val in cohesion.items():
        if coh_val < 0.3:  # Very low cohesion
            opp = RefactoringOpportunity(
                name=f"Extract {module.split('/')[-1]}",
                description=f"Module has low cohesion ({coh_val:.1%})",
                impact="medium",
                effort=6,
                benefit=7,
            )
            opp.affected_entities = [module]
            opportunities.append(opp)

    # Opportunity 3: Reduce high coupling
    for entity, coup_val in coupling.items():
        if coup_val > 10:  # Highly coupled
            opp = RefactoringOpportunity(
                name=f"Decouple {entity}",
                description=f"Entity has high coupling ({coup_val} dependencies)",
                impact="high",
                effort=8,
                benefit=8,
            )
            opp.affected_entities = [entity]
            opportunities.append(opp)

    return sorted(opportunities, key=lambda x: x.priority(), reverse=True)

opportunities = find_refactoring_opportunities(
    chunks, graph, coupling, cohesion, cycles
)

print(f"\nRefactoring Opportunities (prioritized):\n")
for i, opp in enumerate(opportunities[:5], 1):
    print(f"{i}. {opp.name}")
    print(f"   Impact: {opp.impact} | Effort: {opp.effort}/10 | Priority: {opp.priority():.2f}")
    print(f"   {opp.description}\n")
```

### 2.2 Model Proposed Changes

```python
from dataclasses import dataclass
from typing import List

@dataclass
class ProposedChange:
    """A proposed refactoring change."""
    name: str
    description: str
    affected_files: List[str]
    affected_symbols: List[str]
    removals: List[str]  # Symbols to remove
    additions: List[str]  # Symbols to add
    modifications: dict  # Symbol -> new implementation

def plan_modularization(module_name, graph, chunks):
    """Plan extracting a new module."""

    # Find entities to move
    entities_in_module = [
        e for e in graph.entities.values()
        if module_name in e.metadata.get('source', '')
    ]

    # Find external dependencies
    external_deps = set()
    for entity in entities_in_module:
        rels = graph.get_entity_relationships(entity.entity_id)
        for rel in rels:
            if module_name not in rel.target_entity.metadata.get('source', ''):
                external_deps.add(rel.target_entity.text)

    change = ProposedChange(
        name=f"Extract {module_name} as independent module",
        description="Move tightly related functions to new module",
        affected_files=[e.metadata.get('source') for e in entities_in_module],
        affected_symbols=[e.text for e in entities_in_module],
        removals=[], # These will be moved, not removed
        additions=[], # These will be refactored
        modifications={
            "imports": f"Add {len(external_deps)} new imports",
        },
    )

    return change

# Plan a refactoring
change = plan_modularization("auth", graph, chunks)
print(f"\nProposed Change: {change.name}")
print(f"  Affected files: {len(change.affected_files)}")
print(f"  Affected symbols: {len(change.affected_symbols)}")
print(f"  External dependencies to handle: {len(change.modifications)}")
```

### 2.3 Impact Analysis

```python
def analyze_impact(change, graph):
    """Analyze impact of proposed change."""

    # Find all entities that use affected symbols
    dependents = set()

    for symbol_name in change.affected_symbols:
        # Find entity
        entity = graph.get_entity_by_text(symbol_name)
        if entity:
            rels = graph.get_entity_relationships(entity.entity_id, direction="in")
            for rel in rels:
                dependents.add(rel.source_entity.text)

    # Calculate scope of change
    affected_modules = set()
    for entity_id, entity in graph.entities.items():
        if any(sym in entity.metadata.get('source', '')
               for sym in change.affected_files):
            affected_modules.add(entity.metadata.get('source'))

    risk_score = len(dependents) * 0.1 + len(affected_modules) * 0.2

    return {
        "direct_impact": len(change.affected_symbols),
        "indirect_impact": len(dependents),
        "affected_modules": len(affected_modules),
        "risk_score": min(risk_score, 10.0),
        "safe": risk_score < 5.0,
    }

# Analyze impact
impact = analyze_impact(change, graph)
print(f"\nImpact Analysis:")
print(f"  Direct impact: {impact['direct_impact']} symbols")
print(f"  Indirect impact: {impact['indirect_impact']} dependents")
print(f"  Affected modules: {impact['affected_modules']}")
print(f"  Risk score: {impact['risk_score']:.1f}/10")
print(f"  Safe to proceed: {'YES' if impact['safe'] else 'NO - need more analysis'}")
```

---

## Phase 3: Execution & Validation

### 3.1 Generate Refactoring Checklist

```python
def generate_checklist(change, impact):
    """Generate refactoring checklist."""

    checklist = f"""
# Refactoring Checklist: {change.name}

## Pre-Refactoring
- [ ] Back up current code
- [ ] Ensure all tests pass (baseline)
- [ ] Create feature branch
- [ ] Notify team of planned changes

## Analysis (Already Done)
- [ ] Map affected symbols: {len(change.affected_symbols)}
- [ ] Identify dependencies: {impact['indirect_impact']}
- [ ] Assess risk: {impact['risk_score']:.1f}/10

## Implementation
- [ ] Create new module structure
- [ ] Move symbols:
"""
    for sym in change.affected_symbols[:10]:
        checklist += f"  - [ ] {sym}\n"

    checklist += f"""
## Testing
- [ ] Unit tests for moved symbols
- [ ] Integration tests
- [ ] Test all {impact['indirect_impact']} dependent modules
- [ ] Run full test suite

## Validation
- [ ] Code review
- [ ] Performance benchmarks (if applicable)
- [ ] Security audit (if applicable)
- [ ] Documentation update

## Deployment
- [ ] Merge to main
- [ ] Deploy to staging
- [ ] Monitor for issues
- [ ] Deploy to production
"""

    return checklist

checklist = generate_checklist(change, impact)
print(checklist)
```

### 3.2 Compare Before/After

```python
def compare_graphs(before_graph, after_graph):
    """Compare before/after refactoring."""

    before_stats = before_graph.get_statistics()
    after_stats = after_graph.get_statistics()

    comparison = {
        "entities_before": before_stats['num_entities'],
        "entities_after": after_stats['num_entities'],
        "relationships_before": before_stats['num_relationships'],
        "relationships_after": after_stats['num_relationships'],
        "density_before": before_stats['density'],
        "density_after": after_stats['density'],
    }

    # Calculate improvements
    comparison["coupling_improvement"] = (
        (comparison["relationships_before"] - comparison["relationships_after"]) /
        comparison["relationships_before"] * 100
    )

    return comparison

# Compare (would do this after refactoring)
comparison = {
    "entities_before": 156,
    "entities_after": 156,  # Same, just reorganized
    "relationships_before": 423,
    "relationships_after": 389,  # Reduced by decoupling
    "density_before": 0.087,
    "density_after": 0.067,
    "coupling_improvement": 8.0,
}

print("\nRefactoring Results:")
print(f"  Entities: {comparison['entities_before']} -> {comparison['entities_after']}")
print(f"  Relationships: {comparison['relationships_before']} -> {comparison['relationships_after']}")
print(f"  Coupling improvement: {comparison['coupling_improvement']:.1f}%")
print(f"  Graph density: {comparison['density_before']:.3f} -> {comparison['density_after']:.3f}")
```

---

## Integration with Governance

```python
from src.governance import AuditLogger, User, AccessLevel

# Log refactoring decisions
audit = AuditLogger(log_file=Path(".cache/refactoring_audit.log"))

# Record decision
audit.log_event(
    from src.governance import AuditEvent, AuditEventType
    AuditEvent(
        event_type=AuditEventType.QUERY_EXECUTED,
        user_id="architect",
        resource_type="refactoring",
        resource_id="break_circular_deps",
        action="approve",
        details={
            "description": change.description,
            "risk_score": impact['risk_score'],
            "affected_symbols": len(change.affected_symbols),
        },
        status="success",
    )
)

print("Refactoring decision logged to audit trail")
```

---

## Advanced Patterns

### Strategy Pattern: Gradual Refactoring

```python
def plan_gradual_refactoring(opportunities, max_effort_per_cycle=20):
    """Plan refactoring over multiple cycles."""

    cycles = []
    current_cycle = []
    current_effort = 0

    for opp in opportunities:
        if current_effort + opp.effort <= max_effort_per_cycle:
            current_cycle.append(opp)
            current_effort += opp.effort
        else:
            cycles.append(current_cycle)
            current_cycle = [opp]
            current_effort = opp.effort

    if current_cycle:
        cycles.append(current_cycle)

    print(f"Gradual Refactoring Plan ({len(cycles)} cycles):\n")
    for i, cycle in enumerate(cycles, 1):
        effort = sum(opp.effort for opp in cycle)
        benefit = sum(opp.benefit for opp in cycle)
        print(f"Cycle {i}: effort={effort}, benefit={benefit}")
        for opp in cycle:
            print(f"  - {opp.name}")

    return cycles

cycles = plan_gradual_refactoring(opportunities)
```

### Monitoring Metrics Over Time

```python
import json
from datetime import datetime

def track_metrics_over_time(graph, filename=".cache/metrics_history.json"):
    """Track refactoring progress."""

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "entities": len(graph.entities),
        "relationships": len(graph.relationships),
        "density": graph.get_statistics()['density'],
    }

    # Load history
    try:
        with open(filename) as f:
            history = json.load(f)
    except:
        history = []

    history.append(metrics)

    # Save
    with open(filename, 'w') as f:
        json.dump(history, f, indent=2)

    return history

# Track progress
history = track_metrics_over_time(graph)
print(f"Metrics history: {len(history)} entries")
```

---

## Troubleshooting

### Issue: Change Impact Too Large

**Cause**: Symbol used in many places
**Solution**:
```python
# Use adapter pattern or feature flags
if use_new_implementation:
    return new_version(data)
else:
    return old_version(data)
```

### Issue: Circular Dependencies Reappear

**Cause**: New code re-introduces cycles
**Solution**:
```python
# Enforce layering in architecture
# Use type hints and static analysis to prevent violations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imports for type checking
    from module_a import TypeA
```

---

## Conclusion

The Layered Context Framework transforms code refactoring from a risky, poorly understood activity into a systematic, evidence-based process. By providing comprehensive code analysis, impact assessment, and governance tracking, teams can confidently modernize legacy systems while maintaining functionality and reducing risk.

See also:
- [Knowledge Base Use Case](use_case_knowledge_base.md)
- [Code Review Use Case](use_case_code_review.md)
- [Architecture Guide](architecture.md)

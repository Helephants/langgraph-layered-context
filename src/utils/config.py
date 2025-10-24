"""
Configuration management for the Layered Context Framework.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    """Configuration for document ingestion."""
    max_workers: int = Field(default=4, description="Max concurrent file readers")
    chunk_size: int = Field(default=512, description="Character chunk size")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    supported_formats: list[str] = Field(
        default=["pdf", "md", "txt"],
        description="Supported file formats"
    )


class ExtractionConfig(BaseModel):
    """Configuration for entity and relationship extraction."""
    nlp_model: str = Field(default="en_core_web_md", description="spaCy model")
    min_entity_score: float = Field(default=0.5, description="Minimum entity confidence")
    max_relationships_per_entity: int = Field(default=10)


class GraphConfig(BaseModel):
    """Configuration for knowledge graph."""
    graph_type: str = Field(default="networkx", description="'networkx' or 'neo4j'")
    enable_deep_analysis: bool = Field(default=False, description="Enable deep graph analysis")
    neo4j_uri: Optional[str] = Field(default=None)
    neo4j_user: Optional[str] = Field(default=None)
    neo4j_password: Optional[str] = Field(default=None)


class ContextConfig(BaseModel):
    """Configuration for layered context system."""
    num_layers: int = Field(default=4, description="Number of context layers")
    enable_caching: bool = Field(default=True)
    cache_dir: Path = Field(default=Path(".cache/context"))


class RetrievalConfig(BaseModel):
    """Configuration for retrieval system."""
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model"
    )
    vector_db_path: Path = Field(default=Path(".cache/chroma"))
    top_k: int = Field(default=5, description="Default number of results")
    use_hybrid_ranking: bool = Field(default=True)


class GovernanceConfig(BaseModel):
    """Configuration for governance and audit."""
    enable_audit: bool = Field(default=True)
    audit_log_path: Path = Field(default=Path(".cache/audit"))
    enable_provenance: bool = Field(default=True)
    enforce_access_control: bool = Field(default=False)


class FrameworkConfig(BaseModel):
    """Main configuration for the entire framework."""
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)

    # Global settings
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    def ensure_paths(self):
        """Ensure all configured paths exist."""
        self.context.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retrieval.vector_db_path.mkdir(parents=True, exist_ok=True)
        self.governance.audit_log_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "FrameworkConfig":
        """Load configuration from environment variables."""
        return cls(
            ingestion=IngestionConfig(
                max_workers=int(os.getenv("CONTEXT_MAX_WORKERS", "4")),
            ),
            retrieval=RetrievalConfig(
                embedding_model=os.getenv("CONTEXT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            ),
            debug=os.getenv("CONTEXT_DEBUG", "false").lower() == "true",
        )


# Global configuration instance
_global_config: Optional[FrameworkConfig] = None


def get_config() -> FrameworkConfig:
    """Get the global framework configuration."""
    global _global_config
    if _global_config is None:
        _global_config = FrameworkConfig.from_env()
        _global_config.ensure_paths()
    return _global_config


def set_config(config: FrameworkConfig) -> None:
    """Set the global framework configuration."""
    global _global_config
    config.ensure_paths()
    _global_config = config

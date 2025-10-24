"""
Audit logging and provenance tracking.
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from enum import Enum

from ..utils.logging_util import get_logger

logger = get_logger("governance.audit")


class AuditEventType(Enum):
    """Types of audit events."""
    CONTEXT_RETRIEVED = "context_retrieved"
    ENTITY_ACCESSED = "entity_accessed"
    RELATIONSHIP_ACCESSED = "relationship_accessed"
    USER_ACCESS_DENIED = "access_denied"
    LAYER_ACCESSED = "layer_accessed"
    CHUNK_RETRIEVED = "chunk_retrieved"
    QUERY_EXECUTED = "query_executed"


@dataclass
class AuditEvent:
    """Represents an audit log entry."""
    event_type: AuditEventType
    user_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resource_type: str = ""
    resource_id: str = ""
    action: str = ""
    status: str = "success"  # success, denied, error
    details: dict = field(default_factory=dict)
    ip_address: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class Provenance:
    """Track the provenance of a context or entity."""
    entity_id: str
    source_chunks: list[str] = field(default_factory=list)
    extracted_from: str = ""  # Document source
    created_at: datetime = field(default_factory=datetime.utcnow)
    processing_steps: list[str] = field(default_factory=list)
    confidence_score: float = 1.0

    def add_step(self, step: str) -> None:
        """Add a processing step."""
        self.processing_steps.append(f"{datetime.utcnow().isoformat()}: {step}")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity_id,
            "source_chunks": self.source_chunks,
            "extracted_from": self.extracted_from,
            "created_at": self.created_at.isoformat(),
            "processing_steps": self.processing_steps,
            "confidence_score": self.confidence_score,
        }


class AuditLogger:
    """Log and manage audit events."""

    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize audit logger.

        Args:
            log_file: Optional file path for persisting audit logs
        """
        self.log_file = log_file
        self.events: list[AuditEvent] = []

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Audit log file: {log_file}")

    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        self.events.append(event)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(event.to_json() + "\n")

        logger.debug(f"Audit event: {event.event_type.value} - {event.resource_id}")

    def log_context_retrieval(
        self,
        user_id: str,
        query: str,
        num_results: int,
        session_id: Optional[str] = None,
    ) -> None:
        """Log context retrieval."""
        event = AuditEvent(
            event_type=AuditEventType.CONTEXT_RETRIEVED,
            user_id=user_id,
            action="retrieve",
            details={"query": query, "num_results": num_results},
            session_id=session_id,
        )
        self.log_event(event)

    def log_access_denied(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        reason: str,
        session_id: Optional[str] = None,
    ) -> None:
        """Log denied access attempt."""
        event = AuditEvent(
            event_type=AuditEventType.USER_ACCESS_DENIED,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            status="denied",
            details={"reason": reason},
            session_id=session_id,
        )
        self.log_event(event)

    def get_events_for_user(self, user_id: str) -> list[AuditEvent]:
        """Get all events for a user."""
        return [event for event in self.events if event.user_id == user_id]

    def get_events_by_type(self, event_type: AuditEventType) -> list[AuditEvent]:
        """Get all events of a specific type."""
        return [event for event in self.events if event.event_type == event_type]

    def export_events(self, output_file: Path) -> None:
        """Export all events to a JSON file."""
        with open(output_file, "w") as f:
            events_data = [event.to_dict() for event in self.events]
            json.dump(events_data, f, indent=2, default=str)
        logger.info(f"Exported {len(self.events)} audit events to {output_file}")


class ProvenanceTracker:
    """Track provenance of entities and contexts."""

    def __init__(self):
        """Initialize provenance tracker."""
        self.provenance_map: dict[str, Provenance] = {}

    def track_entity(
        self,
        entity_id: str,
        source_chunks: list[str],
        extracted_from: str,
        confidence: float = 1.0,
    ) -> Provenance:
        """
        Track provenance for an entity.

        Args:
            entity_id: Entity identifier
            source_chunks: List of chunk IDs this entity came from
            extracted_from: Document/source name
            confidence: Confidence score for the entity

        Returns:
            Provenance object
        """
        provenance = Provenance(
            entity_id=entity_id,
            source_chunks=source_chunks,
            extracted_from=extracted_from,
            confidence_score=confidence,
        )
        provenance.add_step("entity_created")
        self.provenance_map[entity_id] = provenance
        return provenance

    def add_processing_step(self, entity_id: str, step: str) -> None:
        """Add a processing step to entity provenance."""
        if entity_id in self.provenance_map:
            self.provenance_map[entity_id].add_step(step)

    def get_provenance(self, entity_id: str) -> Optional[Provenance]:
        """Get provenance for an entity."""
        return self.provenance_map.get(entity_id)

    def export_provenance(self, output_file: Path) -> None:
        """Export all provenance information to JSON."""
        with open(output_file, "w") as f:
            data = {
                entity_id: prov.to_dict()
                for entity_id, prov in self.provenance_map.items()
            }
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Exported provenance for {len(self.provenance_map)} entities")

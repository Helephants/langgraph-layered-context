"""
Access control and permission management.
"""
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Set, Optional
from uuid import uuid4

from ..graph import AccessLevel
from ..utils.logging_util import get_logger

logger = get_logger("governance.access_control")


class Permission(Enum):
    """Permissions for accessing context layers."""
    READ_RAW = "read_raw"
    READ_ENTITY = "read_entity"
    READ_GRAPH = "read_graph"
    READ_ABSTRACT = "read_abstract"
    MODIFY_GRAPH = "modify_graph"
    ADMIN = "admin"


@dataclass
class User:
    """Represents a user with access rights."""
    user_id: str = field(default_factory=lambda: str(uuid4()))
    username: str = ""
    access_level: AccessLevel = AccessLevel.PUBLIC
    permissions: Set[Permission] = field(default_factory=set)
    metadata: dict = field(default_factory=dict)


@dataclass
class AccessRule:
    """Represents an access control rule."""
    rule_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    resource_type: str = ""  # "layer", "entity", "relationship"
    resource_id: Optional[str] = None  # None means all resources of type
    allowed: bool = True
    conditions: dict = field(default_factory=dict)  # Additional conditions
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def is_valid(self) -> bool:
        """Check if this rule is still valid."""
        if self.expires_at is None:
            return True
        return datetime.utcnow() < self.expires_at


class AccessController:
    """Manage access control rules and permissions."""

    def __init__(self):
        """Initialize access controller."""
        self.users: dict[str, User] = {}
        self.rules: dict[str, AccessRule] = {}

    def register_user(self, user: User) -> None:
        """Register a new user."""
        self.users[user.user_id] = user
        logger.info(f"Registered user: {user.username} ({user.access_level.value})")

    def add_rule(self, rule: AccessRule) -> None:
        """Add an access control rule."""
        self.rules[rule.rule_id] = rule
        logger.debug(f"Added access rule: {rule.rule_id}")

    def can_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        action: str = "read",
    ) -> bool:
        """
        Check if a user can access a resource.

        Args:
            user_id: User identifier
            resource_type: Type of resource (layer, entity, relationship)
            resource_id: ID of specific resource (None = all resources of type)
            action: Action being attempted (read, write, delete)

        Returns:
            True if access is allowed
        """
        user = self.users.get(user_id)
        if not user:
            logger.warning(f"Unknown user: {user_id}")
            return False

        # Check if user has explicit permission
        if action == "read":
            # Check based on resource type
            permission_map = {
                "raw": Permission.READ_RAW,
                "entity": Permission.READ_ENTITY,
                "graph": Permission.READ_GRAPH,
                "abstract": Permission.READ_ABSTRACT,
            }
            required_permission = permission_map.get(resource_type)
            if required_permission and required_permission in user.permissions:
                return True

        # Check access control rules
        applicable_rules = [
            rule
            for rule in self.rules.values()
            if rule.user_id == user_id
            and rule.resource_type == resource_type
            and (rule.resource_id is None or rule.resource_id == resource_id)
            and rule.is_valid()
        ]

        # If any rule denies access, deny
        for rule in applicable_rules:
            if not rule.allowed:
                return False

        # If any rule allows access, allow
        for rule in applicable_rules:
            if rule.allowed:
                return True

        # Default: check access level
        # Higher access levels have more permissions
        level_hierarchy = {
            AccessLevel.PUBLIC: 0,
            AccessLevel.INTERNAL: 1,
            AccessLevel.RESTRICTED: 2,
            AccessLevel.CONFIDENTIAL: 3,
        }

        level_to_resource = {
            "raw": AccessLevel.PUBLIC,
            "entity": AccessLevel.INTERNAL,
            "graph": AccessLevel.RESTRICTED,
            "abstract": AccessLevel.RESTRICTED,
        }

        required_level_value = level_hierarchy.get(level_to_resource.get(resource_type, AccessLevel.PUBLIC))
        user_level_value = level_hierarchy.get(user.access_level)

        return user_level_value >= required_level_value

    def get_accessible_layers(self, user_id: str) -> list[str]:
        """Get list of layers accessible to a user."""
        layers = []
        for layer_type in ["raw", "entity", "graph", "abstract"]:
            if self.can_access(user_id, layer_type):
                layers.append(layer_type)
        return layers

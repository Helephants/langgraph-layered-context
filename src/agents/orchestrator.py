"""
Multi-agent orchestration with LangGraph.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from uuid import uuid4

from langgraph.graph import StateGraph, END

from ..context import LayeredContext, ContextLayerType
from ..governance import AccessController, AuditLogger
from ..utils.logging_util import get_logger

logger = get_logger("agents.orchestrator")


class AgentRole(Enum):
    """Types of agents in the framework."""
    RESEARCHER = "researcher"
    SUMMARIZER = "summarizer"
    ANALYST = "analyst"
    RETRIEVER = "retriever"
    VALIDATOR = "validator"


class AgentCapability(Enum):
    """Capabilities of agents."""
    RETRIEVE_CONTEXT = "retrieve_context"
    ANALYZE_ENTITIES = "analyze_entities"
    EXTRACT_RELATIONSHIPS = "extract_relationships"
    SUMMARIZE = "summarize"
    VALIDATE = "validate"
    PLAN = "plan"


@dataclass
class Agent:
    """Represents an agent in the system."""
    agent_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    role: AgentRole = AgentRole.RESEARCHER
    capabilities: List[AgentCapability] = field(default_factory=list)
    preferred_layers: List[ContextLayerType] = field(default_factory=lambda: list(ContextLayerType))
    metadata: dict = field(default_factory=dict)

    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has a capability."""
        return capability in self.capabilities


@dataclass
class AgentState:
    """State for LangGraph agent execution."""
    agent_id: str
    query: str
    context: Optional[LayeredContext] = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class ContextRetrievalTool:
    """Tool for agents to retrieve layered context."""

    def __init__(
        self,
        access_controller: AccessController,
        audit_logger: AuditLogger,
    ):
        """
        Initialize retrieval tool.

        Args:
            access_controller: Access control manager
            audit_logger: Audit logging system
        """
        self.access_controller = access_controller
        self.audit_logger = audit_logger

    async def retrieve_context(
        self,
        user_id: str,
        query: str,
        preferred_layers: Optional[List[ContextLayerType]] = None,
    ) -> Optional[LayeredContext]:
        """
        Retrieve context for a query with access control.

        Args:
            user_id: User/agent identifier
            query: Query or purpose
            preferred_layers: Preferred context layers

        Returns:
            LayeredContext if access allowed, None otherwise
        """
        # Check access for each layer
        accessible_layers = []
        if preferred_layers:
            for layer in preferred_layers:
                resource_id = layer.name if hasattr(layer, 'name') else str(layer.value)
                if self.access_controller.can_access(user_id, "layer", resource_id):
                    accessible_layers.append(layer)
        else:
            # Get all accessible layers
            for layer in ContextLayerType:
                resource_id = layer.name if hasattr(layer, 'name') else str(layer.value)
                if self.access_controller.can_access(user_id, "layer", resource_id):
                    accessible_layers.append(layer)

        if not accessible_layers:
            logger.warning(f"No accessible layers for user {user_id}")
            self.audit_logger.log_access_denied(
                user_id, "context", "all_layers", "No accessible layers"
            )
            return None

        # Log successful retrieval
        self.audit_logger.log_context_retrieval(
            user_id, query, len(accessible_layers)
        )

        # In a real system, context would be built here
        # For now, return a template
        context = LayeredContext(query_or_purpose=query)
        logger.info(f"Retrieved context for user {user_id}")
        return context


class AgentOrchestrator:
    """Orchestrate multiple agents using LangGraph."""

    def __init__(
        self,
        access_controller: AccessController,
        audit_logger: AuditLogger,
    ):
        """
        Initialize orchestrator.

        Args:
            access_controller: Access control manager
            audit_logger: Audit logging
        """
        self.agents: Dict[str, Agent] = {}
        self.access_controller = access_controller
        self.audit_logger = audit_logger
        self.retrieval_tool = ContextRetrievalTool(
            access_controller, audit_logger
        )

        # Build LangGraph workflow
        self._build_workflow()

    def register_agent(self, agent: Agent) -> None:
        """Register an agent."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.role.value})")

    def _build_workflow(self) -> None:
        """Build the LangGraph state machine."""
        # Create state graph
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("retrieve", self._node_retrieve)
        workflow.add_node("process", self._node_process)
        workflow.add_node("validate", self._node_validate)
        workflow.add_node("respond", self._node_respond)

        # Define edges
        workflow.add_edge("retrieve", "process")
        workflow.add_edge("process", "validate")
        workflow.add_conditional_edges(
            "validate",
            self._should_reprocess,
            {
                True: "process",
                False: "respond",
            }
        )
        workflow.add_edge("respond", END)

        # Set entry point
        workflow.set_entry_point("retrieve")

        # Compile workflow
        self.workflow = workflow.compile()

    async def _node_retrieve(self, state: AgentState) -> AgentState:
        """Retrieve context node."""
        agent = self.agents.get(state.agent_id)
        if not agent:
            logger.error(f"Unknown agent: {state.agent_id}")
            return state

        # Retrieve context with access control
        context = await self.retrieval_tool.retrieve_context(
            state.agent_id,
            state.query,
            agent.preferred_layers,
        )

        state.context = context
        state.messages.append({
            "role": "system",
            "content": f"Retrieved context with {len(context.layers) if context else 0} layers"
        })

        return state

    async def _node_process(self, state: AgentState) -> AgentState:
        """Process context node."""
        agent = self.agents.get(state.agent_id)
        if not agent or not state.context:
            return state

        # Process based on agent role
        if agent.role == AgentRole.ANALYZER:
            state.results["analysis"] = self._analyze_context(state.context)
        elif agent.role == AgentRole.SUMMARIZER:
            state.results["summary"] = self._summarize_context(state.context)
        elif agent.role == AgentRole.RETRIEVER:
            state.results["retrieved"] = True

        state.messages.append({
            "role": "assistant",
            "content": f"Processing complete for {agent.role.value}"
        })

        return state

    async def _node_validate(self, state: AgentState) -> AgentState:
        """Validate results node."""
        if not state.results:
            state.metadata["validation_status"] = "no_results"
            return state

        # Simple validation - check if results are empty
        state.metadata["validation_status"] = "valid"
        state.messages.append({
            "role": "system",
            "content": "Validation passed"
        })

        return state

    async def _node_respond(self, state: AgentState) -> AgentState:
        """Generate response node."""
        agent = self.agents.get(state.agent_id)
        if not agent:
            return state

        response = {
            "agent": agent.name,
            "query": state.query,
            "results": state.results,
            "messages": state.messages,
        }

        state.metadata["response"] = response
        return state

    def _should_reprocess(self, state: AgentState) -> bool:
        """Determine if processing should be repeated."""
        # Simple heuristic: reprocess if validation failed
        return state.metadata.get("validation_status") == "invalid"

    @staticmethod
    def _analyze_context(context: LayeredContext) -> dict:
        """Analyze context."""
        return {
            "num_layers": len(context.layers),
            "num_entities": len(context.get_all_entities()),
            "num_relationships": len(context.get_all_relationships()),
        }

    @staticmethod
    def _summarize_context(context: LayeredContext) -> str:
        """Summarize context."""
        return f"Context contains {len(context.layers)} layers with {len(context.get_all_entities())} entities"

    async def execute_agent(
        self,
        agent_id: str,
        query: str,
    ) -> Optional[AgentState]:
        """
        Execute an agent against a query.

        Args:
            agent_id: Agent identifier
            query: Query or task

        Returns:
            Final agent state
        """
        agent = self.agents.get(agent_id)
        if not agent:
            logger.error(f"Unknown agent: {agent_id}")
            return None

        # Create initial state
        state = AgentState(
            agent_id=agent_id,
            query=query,
        )

        # Execute workflow
        # Note: LangGraph execution details depend on specific version
        # This is a template that would be adapted to actual LangGraph API
        logger.info(f"Executing agent {agent.name} with query: {query}")

        return state

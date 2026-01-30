"""
Multi-Agent Core Framework

Enterprise-grade multi-agent system with message passing, similar to AutoGen/CrewAI.
Supports:
- Agent registration and lifecycle management
- Async message passing between agents
- Multiple orchestration modes (sequential, parallel, hierarchical, dynamic)
- Agent reflection and debate capabilities
- Consensus mechanisms
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Set, Type, Union,
    AsyncGenerator, Awaitable
)

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Message Protocol
# =============================================================================

class MessageType(str, Enum):
    """Types of messages that can be passed between agents."""
    REQUEST = "request"           # Request for action/information
    RESPONSE = "response"         # Response to a request
    BROADCAST = "broadcast"       # Message to all agents
    DELEGATE = "delegate"         # Delegate task to another agent
    RESULT = "result"             # Final result
    ERROR = "error"               # Error message
    REFLECTION = "reflection"     # Self-reflection output
    DEBATE = "debate"             # Debate/challenge message
    CONSENSUS = "consensus"       # Consensus request/response
    TERMINATE = "terminate"       # Signal to stop processing
    HEARTBEAT = "heartbeat"       # Health check
    TOOL_REQUEST = "tool_request" # Request to use a tool
    TOOL_RESULT = "tool_result"   # Result from tool execution
    MEMORY_QUERY = "memory_query" # Query memory
    MEMORY_STORE = "memory_store" # Store in memory


class MessagePriority(int, Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class AgentMessage(BaseModel):
    """
    Message structure for inter-agent communication.
    
    Follows a protocol similar to Actor model messaging.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    sender: str
    receiver: str  # Agent ID or "broadcast" for all
    content: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    
    # Tracking
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None  # Links related messages
    reply_to: Optional[str] = None  # ID of message being replied to
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    ttl_seconds: Optional[int] = None  # Time to live
    
    class Config:
        use_enum_values = True


class MessageEnvelope(BaseModel):
    """Wrapper for message with delivery metadata."""
    message: AgentMessage
    attempts: int = 0
    max_attempts: int = 3
    delivered: bool = False
    acknowledged: bool = False
    delivery_time: Optional[datetime] = None


# =============================================================================
# Agent State Management
# =============================================================================

class AgentState(str, Enum):
    """Agent lifecycle states."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    REFLECTING = "reflecting"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class AgentContext:
    """
    Context passed to agents during execution.
    Contains shared state and utilities.
    """
    query_id: str
    original_query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Accumulated state
    messages: List[AgentMessage] = field(default_factory=list)
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    memory_context: Dict[str, Any] = field(default_factory=dict)
    
    # Agent outputs
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    
    # Control flow
    iteration: int = 0
    max_iterations: int = 10
    should_terminate: bool = False
    termination_reason: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.utcnow)
    
    def add_message(self, message: AgentMessage) -> None:
        """Add message to context history."""
        self.messages.append(message)
    
    def get_messages_for_agent(self, agent_id: str) -> List[AgentMessage]:
        """Get messages sent to a specific agent."""
        return [
            m for m in self.messages 
            if m.receiver == agent_id or m.receiver == "broadcast"
        ]
    
    def get_messages_from_agent(self, agent_id: str) -> List[AgentMessage]:
        """Get messages sent by a specific agent."""
        return [m for m in self.messages if m.sender == agent_id]


# =============================================================================
# Base Agent Class
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    Each agent:
    - Has a unique ID and role
    - Can send/receive messages
    - Has a defined capability set
    - Can use tools
    - Supports reflection
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        capabilities: List[str],
        llm_client: Optional[Any] = None,
        tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.llm_client = llm_client
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.state = AgentState.INITIALIZING
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._inbox: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._outbox: asyncio.Queue[AgentMessage] = asyncio.Queue()
        
        # Register default handlers
        self._register_default_handlers()
        
        logger.info(f"Agent initialized: {self.agent_id} ({self.name})")
    
    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self._message_handlers[MessageType.REQUEST] = self._handle_request
        self._message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self._message_handlers[MessageType.TERMINATE] = self._handle_terminate
    
    def register_handler(
        self, 
        message_type: MessageType, 
        handler: Callable[[AgentMessage, AgentContext], Awaitable[Optional[AgentMessage]]]
    ) -> None:
        """Register a custom message handler."""
        self._message_handlers[message_type] = handler
    
    async def receive_message(self, message: AgentMessage) -> None:
        """Receive a message into the inbox."""
        await self._inbox.put(message)
        logger.debug(f"Agent {self.agent_id} received message: {message.type}")
    
    async def send_message(self, message: AgentMessage) -> None:
        """Send a message to the outbox."""
        message.sender = self.agent_id
        await self._outbox.put(message)
        logger.debug(f"Agent {self.agent_id} sent message to {message.receiver}: {message.type}")
    
    async def process_messages(self, context: AgentContext) -> List[AgentMessage]:
        """Process all pending messages in inbox."""
        responses = []
        
        while not self._inbox.empty():
            try:
                message = await asyncio.wait_for(
                    self._inbox.get(), 
                    timeout=0.1
                )
                
                handler = self._message_handlers.get(message.type)
                if handler:
                    response = await handler(message, context)
                    if response:
                        responses.append(response)
                else:
                    logger.warning(
                        f"No handler for message type {message.type} "
                        f"in agent {self.agent_id}"
                    )
            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.error(f"Error processing message in {self.agent_id}: {e}")
        
        return responses
    
    async def get_outgoing_messages(self) -> List[AgentMessage]:
        """Retrieve all outgoing messages."""
        messages = []
        while not self._outbox.empty():
            try:
                message = await asyncio.wait_for(self._outbox.get(), timeout=0.1)
                messages.append(message)
            except asyncio.TimeoutError:
                break
        return messages
    
    # Default handlers
    async def _handle_request(
        self, 
        message: AgentMessage, 
        context: AgentContext
    ) -> Optional[AgentMessage]:
        """Handle incoming request - to be overridden by subclasses."""
        result = await self.execute(message.content, context)
        
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            content=result,
            correlation_id=message.correlation_id,
            reply_to=message.id
        )
    
    async def _handle_heartbeat(
        self, 
        message: AgentMessage, 
        context: AgentContext
    ) -> AgentMessage:
        """Respond to heartbeat."""
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            content={
                "status": self.state.value,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            reply_to=message.id
        )
    
    async def _handle_terminate(
        self, 
        message: AgentMessage, 
        context: AgentContext
    ) -> None:
        """Handle termination signal."""
        self.state = AgentState.TERMINATED
        logger.info(f"Agent {self.agent_id} terminated")
        return None
    
    @abstractmethod
    async def execute(
        self, 
        input_data: Dict[str, Any], 
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Execute the agent's main logic.
        
        Args:
            input_data: Input data for the agent
            context: Shared agent context
            
        Returns:
            Output data from the agent
        """
        pass
    
    async def reflect(self, context: AgentContext) -> Dict[str, Any]:
        """
        Perform self-reflection on agent's output.
        
        Override in subclasses for custom reflection logic.
        """
        return {
            "agent_id": self.agent_id,
            "reflection": "No reflection implemented",
            "confidence": 1.0
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "state": self.state.value
        }


# =============================================================================
# Agent Registry
# =============================================================================

class AgentRegistry:
    """
    Registry for managing agents.
    
    Provides:
    - Agent registration and lookup
    - Capability-based agent discovery
    - Lifecycle management
    """
    
    _instance: Optional['AgentRegistry'] = None
    
    def __new__(cls) -> 'AgentRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents: Dict[str, BaseAgent] = {}
            cls._instance._capability_index: Dict[str, Set[str]] = {}
        return cls._instance
    
    def register(self, agent: BaseAgent) -> None:
        """Register an agent."""
        self._agents[agent.agent_id] = agent
        
        # Index by capabilities
        for capability in agent.capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = set()
            self._capability_index[capability].add(agent.agent_id)
        
        logger.info(f"Registered agent: {agent.agent_id}")
    
    def unregister(self, agent_id: str) -> None:
        """Unregister an agent."""
        if agent_id in self._agents:
            agent = self._agents[agent_id]
            
            # Remove from capability index
            for capability in agent.capabilities:
                if capability in self._capability_index:
                    self._capability_index[capability].discard(agent_id)
            
            del self._agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
    
    def get(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID."""
        return self._agents.get(agent_id)
    
    def get_by_capability(self, capability: str) -> List[BaseAgent]:
        """Get all agents with a specific capability."""
        agent_ids = self._capability_index.get(capability, set())
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]
    
    def get_all(self) -> List[BaseAgent]:
        """Get all registered agents."""
        return list(self._agents.values())
    
    def clear(self) -> None:
        """Clear all registered agents."""
        self._agents.clear()
        self._capability_index.clear()


# =============================================================================
# Message Router
# =============================================================================

class MessageRouter:
    """
    Routes messages between agents.
    
    Supports:
    - Direct messaging
    - Broadcast messaging
    - Priority-based delivery
    - Message persistence (optional)
    """
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._message_history: List[AgentMessage] = []
        self._pending_messages: asyncio.Queue[MessageEnvelope] = asyncio.Queue()
    
    async def route(self, message: AgentMessage) -> bool:
        """
        Route a message to its recipient(s).
        
        Returns:
            True if message was delivered successfully
        """
        self._message_history.append(message)
        
        if message.receiver == "broadcast":
            return await self._broadcast(message)
        else:
            return await self._direct_send(message)
    
    async def _direct_send(self, message: AgentMessage) -> bool:
        """Send message to a specific agent."""
        agent = self.registry.get(message.receiver)
        
        if agent is None:
            logger.warning(f"Agent not found: {message.receiver}")
            return False
        
        await agent.receive_message(message)
        return True
    
    async def _broadcast(self, message: AgentMessage) -> bool:
        """Broadcast message to all agents except sender."""
        agents = self.registry.get_all()
        
        for agent in agents:
            if agent.agent_id != message.sender:
                await agent.receive_message(message)
        
        return True
    
    async def collect_responses(
        self, 
        timeout: float = 30.0
    ) -> List[AgentMessage]:
        """Collect all outgoing messages from agents."""
        responses = []
        agents = self.registry.get_all()
        
        for agent in agents:
            agent_responses = await agent.get_outgoing_messages()
            responses.extend(agent_responses)
        
        return responses
    
    def get_history(
        self, 
        correlation_id: Optional[str] = None
    ) -> List[AgentMessage]:
        """Get message history, optionally filtered by correlation ID."""
        if correlation_id:
            return [
                m for m in self._message_history 
                if m.correlation_id == correlation_id
            ]
        return self._message_history.copy()
    
    def clear_history(self) -> None:
        """Clear message history."""
        self._message_history.clear()


# =============================================================================
# Orchestrator Base
# =============================================================================

class OrchestratorMode(str, Enum):
    """Orchestration modes."""
    SEQUENTIAL = "sequential"     # Agents execute one after another
    PARALLEL = "parallel"         # Agents execute simultaneously
    HIERARCHICAL = "hierarchical" # Tree-like execution
    DYNAMIC = "dynamic"           # LLM decides execution order


class BaseOrchestrator(ABC):
    """
    Base class for agent orchestration.
    
    The orchestrator:
    - Manages agent execution flow
    - Routes messages between agents
    - Handles consensus and termination
    - Tracks overall progress
    """
    
    def __init__(
        self,
        registry: AgentRegistry,
        mode: OrchestratorMode = OrchestratorMode.DYNAMIC,
        max_iterations: int = 10,
        max_agent_calls: int = 20
    ):
        self.registry = registry
        self.router = MessageRouter(registry)
        self.mode = mode
        self.max_iterations = max_iterations
        self.max_agent_calls = max_agent_calls
        
        self._agent_call_count = 0
        self._iteration_count = 0
    
    @abstractmethod
    async def orchestrate(
        self, 
        query: str, 
        context: Optional[AgentContext] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate agent execution for a query.
        
        Args:
            query: User query
            context: Optional pre-existing context
            
        Returns:
            Final result from agent collaboration
        """
        pass
    
    async def _check_termination(self, context: AgentContext) -> bool:
        """Check if orchestration should terminate."""
        if context.should_terminate:
            return True
        
        if self._iteration_count >= self.max_iterations:
            context.termination_reason = "max_iterations_reached"
            return True
        
        if self._agent_call_count >= self.max_agent_calls:
            context.termination_reason = "max_agent_calls_reached"
            return True
        
        return False
    
    async def _execute_agent(
        self, 
        agent: BaseAgent, 
        input_data: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """Execute a single agent with tracking."""
        self._agent_call_count += 1
        agent.state = AgentState.PROCESSING
        
        try:
            result = await agent.execute(input_data, context)
            agent.state = AgentState.IDLE
            return result
        except Exception as e:
            agent.state = AgentState.ERROR
            logger.error(f"Agent {agent.agent_id} execution failed: {e}")
            raise
    
    def reset_counters(self) -> None:
        """Reset iteration and call counters."""
        self._agent_call_count = 0
        self._iteration_count = 0


# =============================================================================
# Factory Functions
# =============================================================================

def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    return AgentRegistry()


def create_context(
    query: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    max_iterations: int = 10,
    metadata: Optional[Dict[str, Any]] = None
) -> AgentContext:
    """Create a new agent context."""
    return AgentContext(
        query_id=str(uuid.uuid4()),
        original_query=query,
        session_id=session_id,
        user_id=user_id,
        max_iterations=max_iterations,
        metadata=metadata or {}
    )

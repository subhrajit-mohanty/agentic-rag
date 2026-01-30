"""
Tool Framework for Agentic RAG

Provides a pluggable tool system that agents can use to perform actions.
Supports:
- Tool registration and discovery
- Input/output validation with Pydantic
- Async execution
- Tool chaining
- Error handling and retries
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Types and Status
# =============================================================================

class ToolCategory(str, Enum):
    """Categories of tools."""
    SEARCH = "search"
    CALCULATION = "calculation"
    DATA = "data"
    CODE = "code"
    COMMUNICATION = "communication"
    MEMORY = "memory"
    EXTERNAL = "external"


class ToolStatus(str, Enum):
    """Tool execution status."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    NOT_AVAILABLE = "not_available"


# =============================================================================
# Tool Input/Output Models
# =============================================================================

class ToolInput(BaseModel):
    """Base input for tools."""
    pass


class ToolOutput(BaseModel):
    """Base output from tools."""
    status: ToolStatus
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    """Definition of a tool for LLM consumption."""
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Any]
    required_parameters: List[str]
    returns: str
    examples: List[Dict[str, Any]] = []


# =============================================================================
# Base Tool Class
# =============================================================================

class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    Each tool:
    - Has a unique name and description
    - Defines its input/output schema
    - Can be discovered and used by agents
    - Supports async execution
    """
    
    name: str
    description: str
    category: ToolCategory
    
    def __init__(
        self,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._enabled = True
        self._call_count = 0
        self._error_count = 0
        
        logger.info(f"Tool initialized: {self.name}")
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled
    
    def enable(self) -> None:
        self._enabled = True
    
    def disable(self) -> None:
        self._enabled = False
    
    @abstractmethod
    def get_input_schema(self) -> Type[BaseModel]:
        """Return the Pydantic model for input validation."""
        pass
    
    @abstractmethod
    async def _execute(self, validated_input: BaseModel) -> Any:
        """Execute the tool with validated input."""
        pass
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        retry: bool = True
    ) -> ToolOutput:
        """
        Execute the tool with input validation and error handling.
        
        Args:
            input_data: Raw input dictionary
            retry: Whether to retry on failure
            
        Returns:
            ToolOutput with result or error
        """
        if not self._enabled:
            return ToolOutput(
                status=ToolStatus.NOT_AVAILABLE,
                error=f"Tool '{self.name}' is not available"
            )
        
        start_time = datetime.utcnow()
        
        # Validate input
        try:
            input_schema = self.get_input_schema()
            validated_input = input_schema.model_validate(input_data)
        except Exception as e:
            logger.error(f"Tool {self.name} input validation failed: {e}")
            return ToolOutput(
                status=ToolStatus.INVALID_INPUT,
                error=f"Invalid input: {str(e)}",
                execution_time_ms=self._calc_time(start_time)
            )
        
        # Execute with retries
        last_error = None
        attempts = self.max_retries if retry else 1
        
        for attempt in range(attempts):
            try:
                self._call_count += 1
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute(validated_input),
                    timeout=self.timeout_seconds
                )
                
                return ToolOutput(
                    status=ToolStatus.SUCCESS,
                    result=result,
                    execution_time_ms=self._calc_time(start_time),
                    metadata={
                        "attempt": attempt + 1,
                        "tool": self.name
                    }
                )
                
            except asyncio.TimeoutError:
                last_error = "Execution timeout"
                logger.warning(f"Tool {self.name} timeout (attempt {attempt + 1})")
                
            except Exception as e:
                last_error = str(e)
                self._error_count += 1
                logger.error(f"Tool {self.name} error (attempt {attempt + 1}): {e}")
            
            if attempt < attempts - 1:
                await asyncio.sleep(self.retry_delay)
        
        return ToolOutput(
            status=ToolStatus.FAILURE if last_error != "Execution timeout" else ToolStatus.TIMEOUT,
            error=last_error,
            execution_time_ms=self._calc_time(start_time),
            metadata={"attempts": attempts}
        )
    
    def _calc_time(self, start: datetime) -> float:
        """Calculate execution time in milliseconds."""
        return (datetime.utcnow() - start).total_seconds() * 1000
    
    def get_definition(self) -> ToolDefinition:
        """Get tool definition for LLM consumption."""
        schema = self.get_input_schema().model_json_schema()
        
        return ToolDefinition(
            name=self.name,
            description=self.description,
            category=self.category,
            parameters=schema.get("properties", {}),
            required_parameters=schema.get("required", []),
            returns=self._get_return_description(),
            examples=self._get_examples()
        )
    
    def _get_return_description(self) -> str:
        """Override to provide return description."""
        return "Result of tool execution"
    
    def _get_examples(self) -> List[Dict[str, Any]]:
        """Override to provide usage examples."""
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            "name": self.name,
            "enabled": self._enabled,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._call_count, 1)
        }


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """
    Registry for managing tools.
    
    Provides:
    - Tool registration and lookup
    - Category-based discovery
    - Tool definitions for LLM
    """
    
    _instance: Optional['ToolRegistry'] = None
    
    def __new__(cls) -> 'ToolRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, BaseTool] = {}
            cls._instance._category_index: Dict[ToolCategory, List[str]] = {}
        return cls._instance
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        
        # Index by category
        if tool.category not in self._category_index:
            self._category_index[tool.category] = []
        
        if tool.name not in self._category_index[tool.category]:
            self._category_index[tool.category].append(tool.name)
        
        logger.info(f"Registered tool: {tool.name} ({tool.category})")
    
    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            tool = self._tools[name]
            
            # Remove from category index
            if tool.category in self._category_index:
                if name in self._category_index[tool.category]:
                    self._category_index[tool.category].remove(name)
            
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get tool by name."""
        return self._tools.get(name)
    
    def get_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Get all tools in a category."""
        names = self._category_index.get(category, [])
        return [self._tools[name] for name in names if name in self._tools]
    
    def get_all(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_enabled(self) -> List[BaseTool]:
        """Get all enabled tools."""
        return [t for t in self._tools.values() if t.is_enabled]
    
    def get_definitions(self, enabled_only: bool = True) -> List[ToolDefinition]:
        """Get definitions for all tools (for LLM consumption)."""
        tools = self.get_enabled() if enabled_only else self.get_all()
        return [tool.get_definition() for tool in tools]
    
    def get_definitions_json(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Get definitions as JSON-serializable dicts."""
        definitions = self.get_definitions(enabled_only)
        return [d.model_dump() for d in definitions]
    
    async def execute(
        self,
        tool_name: str,
        input_data: Dict[str, Any]
    ) -> ToolOutput:
        """Execute a tool by name."""
        tool = self.get(tool_name)
        
        if tool is None:
            return ToolOutput(
                status=ToolStatus.NOT_AVAILABLE,
                error=f"Tool '{tool_name}' not found"
            )
        
        return await tool.execute(input_data)
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._category_index.clear()


# =============================================================================
# Tool Executor
# =============================================================================

class ToolExecutor:
    """
    Executes tools with agent context awareness.
    
    Features:
    - Tool selection based on task
    - Parallel execution of independent tools
    - Result aggregation
    - Execution logging
    """
    
    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or ToolRegistry()
        self._execution_log: List[Dict[str, Any]] = []
    
    async def execute_single(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolOutput:
        """Execute a single tool."""
        output = await self.registry.execute(tool_name, input_data)
        
        # Log execution
        self._execution_log.append({
            "tool": tool_name,
            "input": input_data,
            "output_status": output.status,
            "execution_time_ms": output.execution_time_ms,
            "timestamp": datetime.utcnow().isoformat(),
            "context": context
        })
        
        return output
    
    async def execute_parallel(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolOutput]:
        """
        Execute multiple tools in parallel.
        
        Args:
            tool_calls: List of {"tool": "name", "input": {...}}
            
        Returns:
            List of ToolOutputs in same order
        """
        tasks = [
            self.execute_single(call["tool"], call.get("input", {}))
            for call in tool_calls
        ]
        
        return await asyncio.gather(*tasks)
    
    async def execute_chain(
        self,
        tool_chain: List[Dict[str, Any]],
        initial_input: Dict[str, Any] = {}
    ) -> List[ToolOutput]:
        """
        Execute tools in sequence, passing output as input to next.
        
        Args:
            tool_chain: List of {"tool": "name", "input_mapping": {"key": "prev.result.field"}}
            initial_input: Initial input for first tool
            
        Returns:
            List of all ToolOutputs
        """
        outputs = []
        current_input = initial_input.copy()
        
        for step in tool_chain:
            tool_name = step["tool"]
            input_mapping = step.get("input_mapping", {})
            
            # Build input from mapping
            step_input = {}
            for key, value in input_mapping.items():
                if isinstance(value, str) and value.startswith("prev."):
                    # Reference to previous output
                    if outputs:
                        path = value.split(".")[1:]
                        ref_value = outputs[-1].result
                        for p in path:
                            if isinstance(ref_value, dict):
                                ref_value = ref_value.get(p)
                        step_input[key] = ref_value
                else:
                    step_input[key] = value
            
            # Merge with current input
            step_input = {**current_input, **step_input, **step.get("input", {})}
            
            # Execute
            output = await self.execute_single(tool_name, step_input)
            outputs.append(output)
            
            # Stop chain on failure
            if output.status != ToolStatus.SUCCESS:
                break
            
            # Update current input with result
            if isinstance(output.result, dict):
                current_input.update(output.result)
        
        return outputs
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self._execution_log.copy()
    
    def clear_log(self) -> None:
        """Clear execution log."""
        self._execution_log.clear()


# =============================================================================
# Factory Functions
# =============================================================================

def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return ToolRegistry()


def create_tool_executor(registry: Optional[ToolRegistry] = None) -> ToolExecutor:
    """Create a new tool executor."""
    return ToolExecutor(registry or get_tool_registry())

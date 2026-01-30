"""
Tool Framework for Agentic RAG

Pluggable tool system for agent actions.
"""

from .base import (
    BaseTool,
    ToolCategory,
    ToolStatus,
    ToolInput,
    ToolOutput,
    ToolDefinition,
    ToolRegistry,
    ToolExecutor,
    get_tool_registry,
    create_tool_executor
)

from .specialized import (
    CalculatorTool,
    CodeExecutorTool,
    WebSearchTool,
    DatabaseTool,
    SearchTool,
    create_calculator_tool,
    create_code_executor_tool,
    create_web_search_tool,
    create_database_tool,
    create_search_tool
)

__all__ = [
    # Base
    "BaseTool",
    "ToolCategory",
    "ToolStatus",
    "ToolInput",
    "ToolOutput",
    "ToolDefinition",
    "ToolRegistry",
    "ToolExecutor",
    "get_tool_registry",
    "create_tool_executor",
    
    # Specialized Tools
    "CalculatorTool",
    "CodeExecutorTool",
    "WebSearchTool",
    "DatabaseTool",
    "SearchTool",
    "create_calculator_tool",
    "create_code_executor_tool",
    "create_web_search_tool",
    "create_database_tool",
    "create_search_tool"
]

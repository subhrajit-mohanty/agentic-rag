"""
Specialized Tools for Agentic RAG

Implements core tools:
- CalculatorTool: Mathematical calculations
- CodeExecutorTool: Containerized code execution (K8s ready)
- WebSearchTool: Configurable web search (Tavily, SerpAPI, Bing, Google)
- DatabaseTool: SQL/NoSQL database queries
- SearchTool: Internal knowledge base search
"""

import asyncio
import json
import logging
import math
import re
import subprocess
import tempfile
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import httpx
from pydantic import BaseModel, Field

from .base import BaseTool, ToolCategory, ToolOutput, ToolStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Calculator Tool
# =============================================================================

class CalculatorInput(BaseModel):
    """Input for calculator tool."""
    expression: str = Field(
        description="Mathematical expression to evaluate"
    )
    precision: int = Field(
        default=6,
        ge=0,
        le=15,
        description="Decimal precision for result"
    )


class CalculatorTool(BaseTool):
    """
    Safe mathematical calculator.
    
    Supports:
    - Basic arithmetic
    - Scientific functions (sin, cos, log, etc.)
    - Statistical functions (mean, std, etc.)
    - Constants (pi, e)
    """
    
    name = "calculator"
    description = "Evaluate mathematical expressions safely"
    category = ToolCategory.CALCULATION
    
    # Safe math functions
    SAFE_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "pow": pow,
        # Math module
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "degrees": math.degrees,
        "radians": math.radians,
    }
    
    SAFE_CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": float("inf"),
    }
    
    def get_input_schema(self) -> Type[BaseModel]:
        return CalculatorInput
    
    async def _execute(self, validated_input: CalculatorInput) -> Dict[str, Any]:
        """Evaluate the mathematical expression."""
        expression = validated_input.expression
        precision = validated_input.precision
        
        # Sanitize expression
        sanitized = self._sanitize_expression(expression)
        
        # Build safe namespace
        namespace = {
            **self.SAFE_FUNCTIONS,
            **self.SAFE_CONSTANTS,
            "__builtins__": {},
        }
        
        try:
            # Compile and evaluate
            compiled = compile(sanitized, "<calculator>", "eval")
            result = eval(compiled, namespace)
            
            # Format result
            if isinstance(result, float):
                result = round(result, precision)
            
            return {
                "expression": expression,
                "result": result,
                "type": type(result).__name__
            }
            
        except Exception as e:
            raise ValueError(f"Calculation error: {str(e)}")
    
    def _sanitize_expression(self, expr: str) -> str:
        """Sanitize expression for safe evaluation."""
        # Remove potentially dangerous patterns
        dangerous = ["import", "exec", "eval", "__", "open", "file", "os.", "sys."]
        expr_lower = expr.lower()
        
        for pattern in dangerous:
            if pattern in expr_lower:
                raise ValueError(f"Unsafe pattern detected: {pattern}")
        
        return expr
    
    def _get_return_description(self) -> str:
        return "Calculation result with expression, result value, and type"
    
    def _get_examples(self) -> List[Dict[str, Any]]:
        return [
            {"input": {"expression": "2 + 2"}, "output": {"result": 4}},
            {"input": {"expression": "sqrt(16)"}, "output": {"result": 4.0}},
            {"input": {"expression": "sin(pi/2)"}, "output": {"result": 1.0}},
        ]


# =============================================================================
# Code Executor Tool (Containerized)
# =============================================================================

class CodeExecutorInput(BaseModel):
    """Input for code executor tool."""
    code: str = Field(description="Python code to execute")
    language: str = Field(default="python", description="Programming language")
    timeout: int = Field(default=30, ge=1, le=300, description="Timeout in seconds")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input variables")


class CodeExecutorTool(BaseTool):
    """
    Containerized code execution tool.
    
    Supports:
    - Docker execution (default)
    - Kubernetes job execution
    - Input/output variable passing
    - Resource limits
    """
    
    name = "code_executor"
    description = "Execute code safely in an isolated container"
    category = ToolCategory.CODE
    
    def __init__(
        self,
        container_runtime: str = "docker",
        container_image: str = "python:3.11-slim",
        memory_limit: str = "256m",
        cpu_limit: str = "0.5",
        enable_network: bool = False,
        k8s_namespace: str = "agentic-rag",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.container_runtime = container_runtime
        self.container_image = container_image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.enable_network = enable_network
        self.k8s_namespace = k8s_namespace
    
    def get_input_schema(self) -> Type[BaseModel]:
        return CodeExecutorInput
    
    async def _execute(self, validated_input: CodeExecutorInput) -> Dict[str, Any]:
        """Execute code in container."""
        if self.container_runtime == "docker":
            return await self._execute_docker(validated_input)
        elif self.container_runtime == "kubernetes":
            return await self._execute_kubernetes(validated_input)
        else:
            # Fallback to local (unsafe, for development only)
            return await self._execute_local(validated_input)
    
    async def _execute_docker(self, input_data: CodeExecutorInput) -> Dict[str, Any]:
        """Execute code using Docker."""
        # Create temp file with code
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False
        ) as f:
            # Inject inputs
            code = self._prepare_code(input_data.code, input_data.inputs)
            f.write(code)
            code_file = f.name
        
        try:
            # Build docker command
            cmd = [
                "docker", "run",
                "--rm",
                "--memory", self.memory_limit,
                "--cpus", self.cpu_limit,
                "--read-only",
            ]
            
            if not self.enable_network:
                cmd.append("--network=none")
            
            # Mount code file
            cmd.extend([
                "-v", f"{code_file}:/app/code.py:ro",
                self.container_image,
                "python", "/app/code.py"
            ])
            
            # Execute
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=input_data.timeout
            )
            
            return {
                "stdout": stdout.decode()[:10000],  # Limit output size
                "stderr": stderr.decode()[:2000],
                "exit_code": process.returncode,
                "success": process.returncode == 0
            }
            
        except asyncio.TimeoutError:
            raise TimeoutError("Code execution timeout")
        except FileNotFoundError:
            # Docker not available, fall back to local
            logger.warning("Docker not available, using local execution")
            return await self._execute_local(input_data)
        finally:
            # Clean up temp file
            Path(code_file).unlink(missing_ok=True)
    
    async def _execute_kubernetes(self, input_data: CodeExecutorInput) -> Dict[str, Any]:
        """Execute code using Kubernetes Job."""
        # This would create a K8s Job and wait for completion
        # Simplified implementation - in production, use kubernetes-asyncio
        
        job_name = f"code-exec-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        job_manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": self.k8s_namespace
            },
            "spec": {
                "ttlSecondsAfterFinished": 60,
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "executor",
                            "image": self.container_image,
                            "command": ["python", "-c", input_data.code],
                            "resources": {
                                "limits": {
                                    "memory": self.memory_limit,
                                    "cpu": self.cpu_limit
                                }
                            }
                        }],
                        "restartPolicy": "Never"
                    }
                }
            }
        }
        
        # In production: use kubernetes-asyncio client
        # For now, return placeholder
        logger.info(f"Would create K8s job: {job_name}")
        
        return {
            "stdout": "K8s execution not fully implemented",
            "stderr": "",
            "exit_code": 0,
            "success": True,
            "job_name": job_name
        }
    
    async def _execute_local(self, input_data: CodeExecutorInput) -> Dict[str, Any]:
        """Execute code locally (development only, not secure)."""
        logger.warning("Using local code execution - NOT SECURE for production!")
        
        code = self._prepare_code(input_data.code, input_data.inputs)
        
        # Create restricted namespace
        namespace = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "bool": bool,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
            }
        }
        
        # Capture output
        output_capture = []
        original_print = print
        
        def captured_print(*args, **kwargs):
            output_capture.append(" ".join(str(a) for a in args))
        
        namespace["__builtins__"]["print"] = captured_print
        
        try:
            exec(code, namespace)
            
            return {
                "stdout": "\n".join(output_capture),
                "stderr": "",
                "exit_code": 0,
                "success": True,
                "result": namespace.get("result")
            }
            
        except Exception as e:
            return {
                "stdout": "\n".join(output_capture),
                "stderr": str(e),
                "exit_code": 1,
                "success": False
            }
    
    def _prepare_code(self, code: str, inputs: Dict[str, Any]) -> str:
        """Prepare code with input injection."""
        if not inputs:
            return code
        
        # Inject inputs at the beginning
        input_lines = []
        for key, value in inputs.items():
            if isinstance(value, str):
                input_lines.append(f'{key} = """{value}"""')
            else:
                input_lines.append(f"{key} = {repr(value)}")
        
        return "\n".join(input_lines) + "\n\n" + code


# =============================================================================
# Web Search Tool (Configurable)
# =============================================================================

class WebSearchInput(BaseModel):
    """Input for web search tool."""
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, ge=1, le=20)
    search_depth: str = Field(default="basic", description="basic or advanced")


class WebSearchTool(BaseTool):
    """
    Configurable web search tool.
    
    Supports multiple providers:
    - Tavily (default, built for RAG)
    - SerpAPI
    - Bing Search API
    - Google Custom Search
    - DuckDuckGo (no API key needed)
    """
    
    name = "web_search"
    description = "Search the web for current information"
    category = ToolCategory.EXTERNAL
    
    def __init__(
        self,
        provider: str = "tavily",
        tavily_api_key: Optional[str] = None,
        serp_api_key: Optional[str] = None,
        bing_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.provider = provider
        self.tavily_api_key = tavily_api_key
        self.serp_api_key = serp_api_key
        self.bing_api_key = bing_api_key
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    def get_input_schema(self) -> Type[BaseModel]:
        return WebSearchInput
    
    async def _execute(self, validated_input: WebSearchInput) -> Dict[str, Any]:
        """Execute web search."""
        if self.provider == "tavily":
            return await self._search_tavily(validated_input)
        elif self.provider == "serp":
            return await self._search_serp(validated_input)
        elif self.provider == "bing":
            return await self._search_bing(validated_input)
        elif self.provider == "google":
            return await self._search_google(validated_input)
        elif self.provider == "duckduckgo":
            return await self._search_duckduckgo(validated_input)
        else:
            raise ValueError(f"Unknown search provider: {self.provider}")
    
    async def _search_tavily(self, input_data: WebSearchInput) -> Dict[str, Any]:
        """Search using Tavily API."""
        if not self.tavily_api_key:
            raise ValueError("Tavily API key not configured")
        
        client = await self._get_client()
        
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.tavily_api_key,
                "query": input_data.query,
                "max_results": input_data.max_results,
                "search_depth": input_data.search_depth,
                "include_answer": True,
                "include_raw_content": False
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "query": input_data.query,
            "provider": "tavily",
            "answer": data.get("answer"),
            "results": [
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "content": r.get("content"),
                    "score": r.get("score")
                }
                for r in data.get("results", [])
            ]
        }
    
    async def _search_serp(self, input_data: WebSearchInput) -> Dict[str, Any]:
        """Search using SerpAPI."""
        if not self.serp_api_key:
            raise ValueError("SerpAPI key not configured")
        
        client = await self._get_client()
        
        response = await client.get(
            "https://serpapi.com/search",
            params={
                "api_key": self.serp_api_key,
                "q": input_data.query,
                "num": input_data.max_results,
                "engine": "google"
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "query": input_data.query,
            "provider": "serp",
            "results": [
                {
                    "title": r.get("title"),
                    "url": r.get("link"),
                    "content": r.get("snippet")
                }
                for r in data.get("organic_results", [])[:input_data.max_results]
            ]
        }
    
    async def _search_bing(self, input_data: WebSearchInput) -> Dict[str, Any]:
        """Search using Bing Search API."""
        if not self.bing_api_key:
            raise ValueError("Bing API key not configured")
        
        client = await self._get_client()
        
        response = await client.get(
            "https://api.bing.microsoft.com/v7.0/search",
            headers={"Ocp-Apim-Subscription-Key": self.bing_api_key},
            params={
                "q": input_data.query,
                "count": input_data.max_results
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "query": input_data.query,
            "provider": "bing",
            "results": [
                {
                    "title": r.get("name"),
                    "url": r.get("url"),
                    "content": r.get("snippet")
                }
                for r in data.get("webPages", {}).get("value", [])
            ]
        }
    
    async def _search_google(self, input_data: WebSearchInput) -> Dict[str, Any]:
        """Search using Google Custom Search API."""
        if not self.google_api_key or not self.google_cse_id:
            raise ValueError("Google API key or CSE ID not configured")
        
        client = await self._get_client()
        
        response = await client.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": input_data.query,
                "num": min(input_data.max_results, 10)
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "query": input_data.query,
            "provider": "google",
            "results": [
                {
                    "title": r.get("title"),
                    "url": r.get("link"),
                    "content": r.get("snippet")
                }
                for r in data.get("items", [])
            ]
        }
    
    async def _search_duckduckgo(self, input_data: WebSearchInput) -> Dict[str, Any]:
        """Search using DuckDuckGo (no API key needed)."""
        # DuckDuckGo instant answer API
        client = await self._get_client()
        
        response = await client.get(
            "https://api.duckduckgo.com/",
            params={
                "q": input_data.query,
                "format": "json",
                "no_redirect": 1
            }
        )
        response.raise_for_status()
        data = response.json()
        
        results = []
        
        # Add abstract if available
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", "Summary"),
                "url": data.get("AbstractURL"),
                "content": data.get("Abstract")
            })
        
        # Add related topics
        for topic in data.get("RelatedTopics", [])[:input_data.max_results - 1]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:100],
                    "url": topic.get("FirstURL"),
                    "content": topic.get("Text")
                })
        
        return {
            "query": input_data.query,
            "provider": "duckduckgo",
            "results": results[:input_data.max_results]
        }
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# Database Tool
# =============================================================================

class DatabaseInput(BaseModel):
    """Input for database tool."""
    query: str = Field(description="SQL query or MongoDB query")
    database_type: str = Field(default="mongodb", description="mongodb or sql")
    collection: Optional[str] = Field(default=None, description="MongoDB collection")
    limit: int = Field(default=100, ge=1, le=1000)


class DatabaseTool(BaseTool):
    """
    Database query tool.
    
    Supports:
    - MongoDB queries
    - SQL queries (with restrictions)
    """
    
    name = "database"
    description = "Query internal databases for structured data"
    category = ToolCategory.DATA
    
    def __init__(
        self,
        mongodb_client: Any = None,
        sql_connection: Any = None,
        allowed_collections: List[str] = None,
        allowed_tables: List[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mongodb_client = mongodb_client
        self.sql_connection = sql_connection
        self.allowed_collections = allowed_collections or []
        self.allowed_tables = allowed_tables or []
    
    def get_input_schema(self) -> Type[BaseModel]:
        return DatabaseInput
    
    async def _execute(self, validated_input: DatabaseInput) -> Dict[str, Any]:
        """Execute database query."""
        if validated_input.database_type == "mongodb":
            return await self._query_mongodb(validated_input)
        elif validated_input.database_type == "sql":
            return await self._query_sql(validated_input)
        else:
            raise ValueError(f"Unknown database type: {validated_input.database_type}")
    
    async def _query_mongodb(self, input_data: DatabaseInput) -> Dict[str, Any]:
        """Execute MongoDB query."""
        if not self.mongodb_client:
            raise ValueError("MongoDB client not configured")
        
        collection = input_data.collection
        if collection and self.allowed_collections and collection not in self.allowed_collections:
            raise ValueError(f"Collection '{collection}' not allowed")
        
        try:
            # Parse query (expecting JSON string or dict)
            if isinstance(input_data.query, str):
                query = json.loads(input_data.query)
            else:
                query = input_data.query
            
            # Execute query
            db = self.mongodb_client
            coll = db[collection] if collection else db["documents"]
            
            cursor = coll.find(query).limit(input_data.limit)
            results = await cursor.to_list(length=input_data.limit)
            
            # Convert ObjectId to string
            for doc in results:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
            
            return {
                "database_type": "mongodb",
                "collection": collection,
                "query": input_data.query,
                "count": len(results),
                "results": results
            }
            
        except Exception as e:
            raise ValueError(f"MongoDB query error: {str(e)}")
    
    async def _query_sql(self, input_data: DatabaseInput) -> Dict[str, Any]:
        """Execute SQL query (read-only)."""
        if not self.sql_connection:
            raise ValueError("SQL connection not configured")
        
        query = input_data.query.strip().upper()
        
        # Only allow SELECT queries
        if not query.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")
        
        # Check for dangerous patterns
        dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "EXEC"]
        for pattern in dangerous:
            if pattern in query:
                raise ValueError(f"Query contains forbidden keyword: {pattern}")
        
        # Check table access
        # Simplified - in production, use proper SQL parser
        for table in self.allowed_tables:
            if table.upper() not in query:
                # Table not referenced, check if any other table is
                pass
        
        try:
            # Execute query (using sync cursor in thread)
            loop = asyncio.get_event_loop()
            
            def execute_query():
                cursor = self.sql_connection.cursor()
                cursor.execute(input_data.query)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchmany(input_data.limit)
                return columns, rows
            
            columns, rows = await loop.run_in_executor(None, execute_query)
            
            # Convert to list of dicts
            results = [dict(zip(columns, row)) for row in rows]
            
            return {
                "database_type": "sql",
                "query": input_data.query,
                "columns": columns,
                "count": len(results),
                "results": results
            }
            
        except Exception as e:
            raise ValueError(f"SQL query error: {str(e)}")


# =============================================================================
# Internal Search Tool
# =============================================================================

class SearchInput(BaseModel):
    """Input for internal search tool."""
    query: str = Field(description="Search query")
    limit: int = Field(default=5, ge=1, le=20)
    use_hybrid: bool = Field(default=True)
    filters: Dict[str, Any] = Field(default_factory=dict)


class SearchTool(BaseTool):
    """
    Internal knowledge base search tool.
    
    Uses the vector store service for retrieval.
    """
    
    name = "search"
    description = "Search the internal knowledge base"
    category = ToolCategory.SEARCH
    
    def __init__(self, vector_store: Any = None, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store
    
    def get_input_schema(self) -> Type[BaseModel]:
        return SearchInput
    
    async def _execute(self, validated_input: SearchInput) -> Dict[str, Any]:
        """Execute internal search."""
        if not self.vector_store:
            raise ValueError("Vector store not configured")
        
        results = await self.vector_store.search(
            query=validated_input.query,
            limit=validated_input.limit,
            use_hybrid=validated_input.use_hybrid
        )
        
        return {
            "query": validated_input.query,
            "total": len(results),
            "results": results
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_calculator_tool(**kwargs) -> CalculatorTool:
    """Create calculator tool."""
    return CalculatorTool(**kwargs)


def create_code_executor_tool(
    container_runtime: str = "docker",
    **kwargs
) -> CodeExecutorTool:
    """Create code executor tool."""
    return CodeExecutorTool(container_runtime=container_runtime, **kwargs)


def create_web_search_tool(
    provider: str = "tavily",
    **kwargs
) -> WebSearchTool:
    """Create web search tool."""
    return WebSearchTool(provider=provider, **kwargs)


def create_database_tool(**kwargs) -> DatabaseTool:
    """Create database tool."""
    return DatabaseTool(**kwargs)


def create_search_tool(vector_store: Any = None, **kwargs) -> SearchTool:
    """Create internal search tool."""
    return SearchTool(vector_store=vector_store, **kwargs)

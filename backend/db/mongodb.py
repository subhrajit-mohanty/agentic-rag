"""
MongoDB Database Module

Async MongoDB connection management using Motor driver with Beanie ODM.
Supports connection pooling, health checks, and graceful shutdown.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional, Type

from beanie import Document, init_beanie
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from backend.core.config import get_settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    MongoDB connection manager with async support.
    
    Handles connection lifecycle, pooling, and health monitoring.
    """
    
    def __init__(self):
        self._client: Optional[AsyncIOMotorClient] = None
        self._database: Optional[AsyncIOMotorDatabase] = None
        self._initialized: bool = False
    
    @property
    def client(self) -> AsyncIOMotorClient:
        if self._client is None:
            raise RuntimeError("Database not initialized. Call connect() first.")
        return self._client
    
    @property
    def database(self) -> AsyncIOMotorDatabase:
        if self._database is None:
            raise RuntimeError("Database not initialized. Call connect() first.")
        return self._database
    
    async def connect(self, document_models: Optional[List[Type[Document]]] = None) -> None:
        """
        Initialize MongoDB connection and Beanie ODM.
        
        Args:
            document_models: List of Beanie Document classes to initialize
        """
        if self._initialized:
            logger.warning("Database already initialized")
            return
        
        settings = get_settings()
        
        try:
            logger.info(f"Connecting to MongoDB at {settings.mongodb.uri[:30]}...")
            
            self._client = AsyncIOMotorClient(
                settings.mongodb.uri,
                maxPoolSize=settings.mongodb.max_pool_size,
                minPoolSize=settings.mongodb.min_pool_size,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
            )
            
            # Verify connection
            await self._client.admin.command("ping")
            logger.info("MongoDB ping successful")
            
            self._database = self._client[settings.mongodb.database]
            
            # Initialize Beanie ODM if document models provided
            if document_models:
                await init_beanie(
                    database=self._database,
                    document_models=document_models
                )
                logger.info(f"Beanie initialized with {len(document_models)} document models")
            
            self._initialized = True
            logger.info(f"Connected to MongoDB database: {settings.mongodb.database}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close MongoDB connection gracefully."""
        if self._client:
            logger.info("Closing MongoDB connection...")
            self._client.close()
            self._client = None
            self._database = None
            self._initialized = False
            logger.info("MongoDB connection closed")
    
    async def health_check(self) -> dict:
        """
        Check MongoDB connection health.
        
        Returns:
            dict: Health status with details
        """
        try:
            if not self._client:
                return {"status": "unhealthy", "message": "Not connected"}
            
            # Ping database
            await self._client.admin.command("ping")
            
            # Get server info
            server_info = await self._client.server_info()
            
            return {
                "status": "healthy",
                "message": "Connected",
                "version": server_info.get("version", "unknown"),
                "database": get_settings().mongodb.database
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": str(e)
            }
    
    def get_collection(self, name: str):
        """Get a MongoDB collection by name."""
        return self.database[name]


# Global database manager instance
db_manager = DatabaseManager()


async def get_database() -> AsyncIOMotorDatabase:
    """Dependency for FastAPI to get database instance."""
    return db_manager.database


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncIOMotorDatabase, None]:
    """Context manager for database sessions."""
    try:
        yield db_manager.database
    finally:
        pass  # Motor handles connection pooling automatically


# Convenience functions
async def init_database(document_models: Optional[List[Type[Document]]] = None) -> None:
    """Initialize database connection."""
    await db_manager.connect(document_models)


async def close_database() -> None:
    """Close database connection."""
    await db_manager.disconnect()

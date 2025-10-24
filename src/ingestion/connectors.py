"""
Database connectors for structured data ingestion.
"""
from typing import AsyncIterator, Optional, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import motor.motor_asyncio

from ..utils.logging_util import get_logger
from .loaders import DocumentChunk

logger = get_logger("ingestion.connectors")


@dataclass
class StructuredData:
    """Represents structured data from a database."""
    data: dict[str, Any]
    source: str
    table_name: str
    metadata: dict = None


class DatabaseConnector(ABC):
    """Base class for database connectors."""

    @abstractmethod
    async def query(self, query_str: str) -> AsyncIterator[StructuredData]:
        """Execute a query and yield results."""
        pass

    @abstractmethod
    async def close(self):
        """Close database connection."""
        pass


class SQLConnector(DatabaseConnector):
    """Connector for SQL databases."""

    def __init__(self, connection_string: str):
        """
        Initialize SQL connector.

        Args:
            connection_string: SQLAlchemy connection string
        """
        self.connection_string = connection_string
        self.engine = None
        self.SessionLocal = None

    async def connect(self):
        """Establish database connection."""
        self.engine = create_async_engine(self.connection_string)
        self.SessionLocal = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        logger.info(f"Connected to SQL database: {self.connection_string}")

    async def query(
        self,
        query_str: str,
        table_name: str = "unknown",
    ) -> AsyncIterator[StructuredData]:
        """
        Execute a SQL query and yield results.

        Args:
            query_str: SQL query string
            table_name: Name of the source table

        Yields:
            StructuredData objects
        """
        if not self.engine:
            await self.connect()

        async with self.SessionLocal() as session:
            try:
                result = await session.execute(query_str)
                rows = result.fetchall()

                for row in rows:
                    row_dict = dict(row._mapping) if hasattr(row, "_mapping") else dict(row)
                    yield StructuredData(
                        data=row_dict,
                        source=self.connection_string,
                        table_name=table_name,
                        metadata={"query": query_str},
                    )
            except Exception as e:
                logger.error(f"Error executing query: {e}")

    async def close(self):
        """Close database connection."""
        if self.engine:
            await self.engine.dispose()
            logger.info("SQL connection closed")


class MongoConnector(DatabaseConnector):
    """Connector for MongoDB."""

    def __init__(self, connection_string: str, database_name: str):
        """
        Initialize MongoDB connector.

        Args:
            connection_string: MongoDB connection string
            database_name: Database name
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None

    async def connect(self):
        """Establish MongoDB connection."""
        self.client = motor.motor_asyncio.AsyncMongoClient(self.connection_string)
        self.db = self.client[self.database_name]
        logger.info(f"Connected to MongoDB: {self.database_name}")

    async def query(
        self,
        collection_name: str,
        filter_dict: dict = None,
        projection: dict = None,
    ) -> AsyncIterator[StructuredData]:
        """
        Query a MongoDB collection.

        Args:
            collection_name: Name of the collection
            filter_dict: MongoDB filter dictionary
            projection: Fields to include/exclude

        Yields:
            StructuredData objects
        """
        if not self.db:
            await self.connect()

        try:
            collection = self.db[collection_name]
            cursor = collection.find(filter_dict or {}, projection)

            async for doc in cursor:
                # Convert MongoDB ObjectId to string
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])

                yield StructuredData(
                    data=doc,
                    source=self.connection_string,
                    table_name=collection_name,
                    metadata={"filter": filter_dict},
                )
        except Exception as e:
            logger.error(f"Error querying MongoDB collection {collection_name}: {e}")

    async def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


class StructuredDataToChunks:
    """Convert structured data to document chunks for indexing."""

    @staticmethod
    def structured_to_chunks(
        structured_data: StructuredData,
        chunk_size: int = 512,
    ) -> list[DocumentChunk]:
        """
        Convert structured data to document chunks.

        Args:
            structured_data: Input structured data
            chunk_size: Target chunk size

        Returns:
            List of DocumentChunk objects
        """
        # Format structured data as text
        text_lines = []
        for key, value in structured_data.data.items():
            text_lines.append(f"{key}: {value}")

        content = "\n".join(text_lines)

        # Create metadata chunk
        chunk = DocumentChunk(
            content=content,
            source=f"{structured_data.table_name}:{structured_data.source}",
            chunk_index=0,
            metadata={
                "type": "structured",
                "table": structured_data.table_name,
                **structured_data.metadata,
            },
        )

        return [chunk]

"""
Document loaders for various file formats.
"""
import asyncio
import glob
from pathlib import Path
from typing import Optional, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

import aiofiles
from pypdf import PdfReader

from ..utils.logging_util import get_logger

logger = get_logger("ingestion.loaders")


@dataclass
class Document:
    """Represents a loaded document."""
    content: str
    source: str
    file_type: str = ""
    metadata: dict = field(default_factory=dict)
    loaded_at: datetime = field(default_factory=datetime.utcnow)

    def chunk(self, chunk_size: int = 512, overlap: int = 50) -> list["DocumentChunk"]:
        """Split document into overlapping chunks."""
        chunks = []
        content = self.content

        for i in range(0, len(content), chunk_size - overlap):
            chunk_text = content[i : i + chunk_size]
            if chunk_text.strip():
                chunk = DocumentChunk(
                    content=chunk_text,
                    source=self.source,
                    chunk_index=len(chunks),
                    metadata=self.metadata.copy(),
                )
                chunks.append(chunk)

        return chunks


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    content: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Unique identifier for this chunk."""
        return f"{self.source}::{self.chunk_index}"


class DocumentLoader(ABC):
    """Base class for document loaders."""

    @abstractmethod
    async def load(self, file_path: Path) -> Optional[Document]:
        """Load a document from a file."""
        pass


class TextLoader(DocumentLoader):
    """Loader for plain text files."""

    async def load(self, file_path: Path) -> Optional[Document]:
        """Load text file."""
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            return Document(
                content=content,
                source=str(file_path),
                file_type="txt",
                metadata={"file_size": file_path.stat().st_size},
            )
        except Exception as e:
            logger.error(f"Failed to load text file {file_path}: {e}")
            return None


class MarkdownLoader(DocumentLoader):
    """Loader for Markdown files."""

    async def load(self, file_path: Path) -> Optional[Document]:
        """Load markdown file."""
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
            return Document(
                content=content,
                source=str(file_path),
                file_type="md",
                metadata={"file_size": file_path.stat().st_size},
            )
        except Exception as e:
            logger.error(f"Failed to load markdown file {file_path}: {e}")
            return None


class PDFLoader(DocumentLoader):
    """Loader for PDF files."""

    async def load(self, file_path: Path) -> Optional[Document]:
        """Load PDF file."""
        try:
            # PDF reading must be done synchronously (pypdf is not async)
            # So we run it in a thread executor
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, self._read_pdf, file_path)
            return Document(
                content=content,
                source=str(file_path),
                file_type="pdf",
                metadata={"file_size": file_path.stat().st_size},
            )
        except Exception as e:
            logger.error(f"Failed to load PDF file {file_path}: {e}")
            return None

    @staticmethod
    def _read_pdf(file_path: Path) -> str:
        """Read PDF content synchronously."""
        try:
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return ""


class DocumentIngestionPipeline:
    """High-speed parallel document ingestion using glob and concurrent reading."""

    def __init__(self, max_workers: int = 4):
        """
        Initialize the ingestion pipeline.

        Args:
            max_workers: Maximum concurrent file readers
        """
        self.max_workers = max_workers
        self.loaders = {
            "txt": TextLoader(),
            "md": MarkdownLoader(),
            "pdf": PDFLoader(),
        }

    async def ingest_from_glob(
        self,
        pattern: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> AsyncIterator[DocumentChunk]:
        """
        Ingest documents matching a glob pattern with parallel reading.

        Args:
            pattern: Glob pattern (e.g., "data/**/*.md")
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks

        Yields:
            DocumentChunk objects
        """
        # Use glob to find matching files
        files = glob.glob(pattern, recursive=True)
        logger.info(f"Found {len(files)} files matching pattern: {pattern}")

        if not files:
            return

        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.max_workers)

        async def load_with_semaphore(file_path: str) -> list[DocumentChunk]:
            async with semaphore:
                return await self._load_and_chunk(
                    file_path, chunk_size, chunk_overlap
                )

        # Load all files concurrently
        tasks = [load_with_semaphore(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Yield chunks from all documents
        for result in results:
            if isinstance(result, list):
                for chunk in result:
                    yield chunk

    async def ingest_from_directory(
        self,
        directory: str | Path,
        file_types: list[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> AsyncIterator[DocumentChunk]:
        """
        Ingest all supported documents from a directory.

        Args:
            directory: Directory path
            file_types: List of file types to ingest (txt, md, pdf)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks

        Yields:
            DocumentChunk objects
        """
        if file_types is None:
            file_types = ["txt", "md", "pdf"]

        directory = Path(directory)
        patterns = [str(directory / f"**/*.{ft}") for ft in file_types]

        for pattern in patterns:
            async for chunk in self.ingest_from_glob(
                pattern, chunk_size, chunk_overlap
            ):
                yield chunk

    async def _load_and_chunk(
        self,
        file_path: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[DocumentChunk]:
        """Load a single file and chunk it."""
        file_path_obj = Path(file_path)
        suffix = file_path_obj.suffix.lstrip(".")

        loader = self.loaders.get(suffix)
        if not loader:
            logger.warning(f"No loader for file type: {suffix}")
            return []

        document = await loader.load(file_path_obj)
        if not document:
            return []

        logger.info(f"Loaded document: {file_path} ({len(document.content)} chars)")
        return document.chunk(chunk_size, chunk_overlap)

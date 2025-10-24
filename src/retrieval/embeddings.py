"""
Vector embeddings and similarity search using Chroma.
"""
from typing import Optional, List
from dataclasses import dataclass
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

from ..ingestion import DocumentChunk
from ..utils.config import get_config
from ..utils.logging_util import get_logger

logger = get_logger("retrieval.embeddings")


@dataclass
class SearchResult:
    """Result from a similarity search."""
    chunk: DocumentChunk
    similarity_score: float
    metadata: dict = None


class EmbeddingModel:
    """Wrapper around sentence transformer for embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.

        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.model.encode([text], convert_to_numpy=True)[0]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts."""
        return self.model.encode(texts, convert_to_numpy=True)

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


class ChromaVectorStore:
    """Vector storage and search using Chroma."""

    def __init__(
        self,
        collection_name: str = "context-chunks",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir: Optional[str] = None,
    ):
        """
        Initialize Chroma vector store.

        Args:
            collection_name: Name of the Chroma collection
            embedding_model: Embedding model to use
            persist_dir: Directory for persistent storage
        """
        self.collection_name = collection_name
        self.embedding_model = EmbeddingModel(embedding_model)

        # Initialize Chroma client
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.EphemeralClient()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Initialized Chroma collection: {collection_name}")

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of document chunks to add
        """
        if not chunks:
            return

        # Prepare data for Chroma
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Embed and add to collection
        embeddings = self.embedding_model.embed_texts(documents).tolist()

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(chunks)} chunks to vector store")

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        where_filter: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query_text: Query text
            top_k: Number of results to return
            where_filter: Optional Chroma metadata filter

        Returns:
            List of SearchResult objects
        """
        # Embed query
        query_embedding = self.embedding_model.embed_text(query_text).tolist()

        # Search in Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        if results["ids"] and len(results["ids"]) > 0:
            for i, chunk_id in enumerate(results["ids"][0]):
                # Convert distance to similarity score (Chroma returns distances)
                distance = results["distances"][0][i]
                similarity_score = 1 - (distance / 2)  # Normalize cosine distance

                chunk = DocumentChunk(
                    content=results["documents"][0][i],
                    source=results["metadatas"][0][i].get("source", "unknown"),
                    chunk_index=results["metadatas"][0][i].get("chunk_index", -1),
                    metadata=results["metadatas"][0][i],
                )

                search_results.append(SearchResult(
                    chunk=chunk,
                    similarity_score=similarity_score,
                ))

        logger.debug(f"Found {len(search_results)} results for query")
        return search_results

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete chunks from the vector store."""
        self.collection.delete(ids=chunk_ids)
        logger.info(f"Deleted {len(chunk_ids)} chunks from vector store")

    def update_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Update chunks in the vector store."""
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        embeddings = self.embedding_model.embed_texts(documents).tolist()

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Updated {len(chunks)} chunks in vector store")

    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        return {
            "collection_name": self.collection_name,
            "num_vectors": self.collection.count(),
            "embedding_model": self.embedding_model.model_name,
        }

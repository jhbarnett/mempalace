"""
client.py — Singleton ChromaDB client for MemPalace
====================================================

Caches the PersistentClient so the ONNX embedding model is loaded once
at process start, not on every MCP tool call.
"""

import logging

import chromadb

from .config import MempalaceConfig

logger = logging.getLogger("mempalace_mcp")

_client = None
_client_path = None


def get_client(palace_path: str = None) -> chromadb.ClientAPI:
    """Return a cached ChromaDB PersistentClient.

    The client is created on first call and reused for the lifetime of the
    process.  If ``palace_path`` changes between calls the cached client is
    replaced (unlikely in normal MCP operation).
    """
    global _client, _client_path

    if palace_path is None:
        palace_path = MempalaceConfig().palace_path

    if _client is None or _client_path != palace_path:
        logger.info("Creating ChromaDB client for %s", palace_path)
        _client = chromadb.PersistentClient(path=palace_path)
        _client_path = palace_path

    return _client


def get_collection(palace_path: str = None, collection_name: str = None, create: bool = False):
    """Return a ChromaDB collection via the cached client.

    Args:
        palace_path: Override palace path (default from config).
        collection_name: Override collection name (default from config).
        create: If True, use get_or_create_collection.

    Returns:
        The collection, or None on failure.
    """
    config = MempalaceConfig()
    path = palace_path or config.palace_path
    name = collection_name or config.collection_name

    try:
        client = get_client(path)
        if create:
            return client.get_or_create_collection(name)
        return client.get_collection(name)
    except Exception:
        return None


def warmup():
    """Pre-load the ONNX model by touching the collection once at startup."""
    config = MempalaceConfig()
    col = get_collection()
    if col is not None:
        try:
            col.count()
            logger.info("ChromaDB warmup complete (%s)", config.palace_path)
        except Exception as e:
            logger.warning("ChromaDB warmup failed: %s", e)
    else:
        logger.info("No palace found at %s — skipping warmup", config.palace_path)

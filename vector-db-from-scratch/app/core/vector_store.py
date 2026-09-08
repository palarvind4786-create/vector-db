"""
Central vector storage, document metadata mapping, tombstone tracking,
and NumPy/JSON disk persistence.
"""

from typing import Any
import numpy as np


class VectorStore:
    """
    Central storage manager storing vectors in a contiguous NumPy array
    and mapping external string doc_ids to internal sequential integer IDs.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.doc_id_to_internal: dict[str, int] = {}
        self.internal_to_doc_id: dict[int, str] = {}
        self.metadata: dict[int, dict[str, Any]] = {}
        self.deleted_mask: np.ndarray = np.empty(0, dtype=bool)

    def add_document(self, doc_id: str, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> int:
        """Append a document vector and register metadata. Returns internal_id."""
        raise NotImplementedError("Skeleton only.")

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Retrieve document metadata and vector by doc_id."""
        raise NotImplementedError("Skeleton only.")

    def delete_document(self, doc_id: str) -> bool:
        """Perform lazy deletion by toggling tombstone mask."""
        raise NotImplementedError("Skeleton only.")

    def get_active_vectors(self) -> tuple[list[int], np.ndarray]:
        """Return non-tombstoned internal IDs and their vectors."""
        raise NotImplementedError("Skeleton only.")

    def compact(self) -> None:
        """Physically drop deleted vectors, re-pack continuous array and re-index IDs."""
        raise NotImplementedError("Skeleton only.")

    def save(self, directory: str) -> None:
        """Persist state: vectors as .npz and metadata/mappings as .json."""
        raise NotImplementedError("Skeleton only.")

    def load(self, directory: str) -> None:
        """Restore state from disk."""
        raise NotImplementedError("Skeleton only.")

import json
import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MemoryItem:
    """Single memory item in ReasoningBank"""
    title: str
    description: str
    content: str
    source_problem_id: str
    success: bool
    created_at: str
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryItem":
        return cls(**data)

class ReasoningBank:
    """Persistent storage for ReasoningBank memory items"""

    def __init__(self, storage_path: str = "memory_bank.json"):
        self.storage_path = storage_path
        self.memories: list[MemoryItem] = []
        self.load()

    def add_memory(self, memory: MemoryItem) -> None:
        """Add a single memory item and persist"""
        self.memories.append(memory)
        self.save()

    def add_memories(self, memories: list[MemoryItem]) -> None:
        """Add multiple memory items and trigger consolidation if budget exceeded"""
        self.memories.extend(memories)
        if len(self.memories) > 20:
            self.consolidate()
        self.save()

    def consolidate(self, budget: int = 20) -> None:
        """
        Consolidate memories to fit within budget.
        Prioritizes successful memories and newer items.
        """
        # Sort by success (True first) and then by creation time (Newer first)
        self.memories.sort(key=lambda x: (x.success, x.created_at), reverse=True)
        self.memories = self.memories[:budget]

    def get_all(self) -> list[MemoryItem]:
        """Return all memories"""
        return self.memories

    def save(self) -> None:
        """Persist memories to JSON file"""
        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump([m.to_dict() for m in self.memories], f, indent=2)

    def load(self) -> None:
        """Load memories from JSON file if it exists"""
        if os.path.exists(self.storage_path):
            with open(self.storage_path) as f:
                data = json.load(f)
                self.memories = [MemoryItem.from_dict(m) for m in data]
        else:
            self.memories = []

    def clear(self) -> None:
        """Clear all memories and persist"""
        self.memories = []
        self.save()

    def __len__(self) -> int:
        return len(self.memories)

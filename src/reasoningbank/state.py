from typing import Any, TypedDict

from .memory import MemoryItem


class AgentState(TypedDict):
    """State for the ReasoningBank agent"""
    problem_id: str
    question: str
    expected_answer: str | None
    retrieved_memories: list[MemoryItem]
    trajectories: list[str]  # Parallel MATTS
    refinement_steps: int    # Sequential MATTS
    solution: str
    success: bool
    evaluation: dict[str, Any]
    new_memories: list[MemoryItem]

class EvalState(TypedDict):
    """State for the evaluation workflow"""
    problems: list[dict[str, Any]]
    results: list[dict[str, Any]]
    index: int

import re

import numpy as np
from langchain_openai import OpenAIEmbeddings

from .memory import MemoryItem


class MemoryRetriever:
    """Retrieve relevant memories using semantic search with answer leak protection"""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=model_name)

    def _has_answer_leak(self, memory: MemoryItem, expected_answer: str) -> bool:
        """
        Check if memory contains the expected answer in a result-like context.
        """
        if not expected_answer:
            return False

        memory_text = f"{memory.title} {memory.description} {memory.content}".lower()
        expected_answer = str(expected_answer).lower()

        # Simple heuristic: if the exact answer appears near result keywords
        result_keywords = ["answer", "result", "total", "=", "is", "solution", "final"]

        # Extract numbers from expected answer
        expected_numbers = re.findall(r"\d+\.?\d*", expected_answer)
        if not expected_numbers:
            return False

        for num in expected_numbers:
            if num in memory_text:
                # Check context around the number
                for match in re.finditer(re.escape(num), memory_text):
                    start, end = match.start(), match.end()
                    context = memory_text[max(0, start-30):min(len(memory_text), end+30)]
                    if any(kw in context for kw in result_keywords):
                        return True
        return False

    async def retrieve(
        self,
        query: str,
        memories: list[MemoryItem],
        top_k: int = 1,
        expected_answer: str | None = None
    ) -> list[tuple[MemoryItem, float]]:
        """Retrieve top-k most relevant memories"""
        if not memories:
            return []

        # Ensure all memories have embeddings
        memories_to_embed = [m for m in memories if m.embedding is None]
        if memories_to_embed:
            texts = [f"{m.title}: {m.description}" for m in memories_to_embed]
            embeddings = await self.embeddings.aembed_documents(texts)
            for m, emb in zip(memories_to_embed, embeddings):
                m.embedding = emb

        # Embed query
        query_embedding = await self.embeddings.aembed_query(query)

        # Calculate cosine similarity
        results = []
        for m in memories:
            if self._has_answer_leak(m, expected_answer):
                continue

            similarity = np.dot(m.embedding, query_embedding) / (
                np.linalg.norm(m.embedding) * np.linalg.norm(query_embedding)
            )
            results.append((m, float(similarity)))

        # Sort and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def format_memories(self, retrieved: list[tuple[MemoryItem, float]]) -> str:
        """Format retrieved memories for a prompt"""
        if not retrieved:
            return "No relevant past strategies found."

        formatted = "## Relevant Past Strategies\n\n"
        for i, (m, score) in enumerate(retrieved, 1):
            status = "Success" if m.success else "Failure Lesson"
            formatted += f"### {i}. {m.title} ({status})\n"
            formatted += f"**Description**: {m.description}\n"
            formatted += f"**Strategy**: {m.content}\n\n"
        return formatted

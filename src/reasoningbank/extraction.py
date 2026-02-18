from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .memory import MemoryItem


class MemoryExtractor:
    """Extract generalizable strategies from problem-solving trajectories"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=1.0)

    async def extract(
        self,
        problem_id: str,
        question: str,
        trajectory: str,
        success: bool,
        expected_answer: str = ""
    ) -> list[MemoryItem]:
        """Extract 1-3 memory items from a trajectory"""

        if success:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert reasoning distiller. Extract 1-2 generalizable reasoning strategies from a successful problem-solving trajectory."),
                ("user", f"""PROBLEM: {question}
                
TRAJECTORY: {trajectory}

Extract strategies that helped solve this problem and can be applied to other similar problems.
Format each strategy exactly as follows:

STRATEGY 1
TITLE: <concise name>
DESCRIPTION: <one sentence summary>
CONTENT: <detailed transferable strategy steps>

STRATEGY 2
...""")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert reasoning distiller. Extract 1-2 lessons about what went wrong from a failed problem-solving trajectory."),
                ("user", f"""PROBLEM: {question}
                
TRAJECTORY: {trajectory}

EXPECTED ANSWER: {expected_answer}

Extract lessons or preventive strategies to avoid these mistakes in the future.
Format each lesson exactly as follows:

STRATEGY 1
TITLE: <concise name>
DESCRIPTION: <one sentence summary>
CONTENT: <detailed preventive steps or check list>

STRATEGY 2
...""")
            ])

        response = await self.llm.ainvoke(prompt.format())
        return self._parse_response(response.content, problem_id, success)

    def _parse_response(self, text: str, problem_id: str, success: bool) -> list[MemoryItem]:
        """Parse structured text into MemoryItem objects"""
        memories = []
        parts = text.split("STRATEGY ")

        for part in parts[1:]:
            try:
                lines = part.strip().split("\n")
                title = ""
                description = ""
                content = []

                collecting_content = False
                for line in lines:
                    if line.startswith("TITLE:"):
                        title = line.replace("TITLE:", "").strip()
                    elif line.startswith("DESCRIPTION:"):
                        description = line.replace("DESCRIPTION:", "").strip()
                    elif line.startswith("CONTENT:"):
                        content.append(line.replace("CONTENT:", "").strip())
                        collecting_content = True
                    elif collecting_content:
                        content.append(line.strip())

                if title and description and content:
                    memories.append(MemoryItem(
                        title=title,
                        description=description,
                        content="\n".join(content),
                        source_problem_id=problem_id,
                        success=success,
                        created_at=datetime.now().isoformat()
                    ))
            except Exception:
                continue
        return memories

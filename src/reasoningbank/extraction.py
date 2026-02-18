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
                ("system", "You are an expert in web navigation and reasoning distillation. You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory. You must first think why the trajectory is successful, and then summarize the insights."),
                ("user", f"""PROBLEM: {question}
                
TRAJECTORY: {trajectory}

Extract generalizable reasoning strategies. Format each memory item using Markdown:
# Memory Item 1
## Title: <concise name>
## Description: <one sentence summary>
## Content: <detailed transferable strategy steps>

# Memory Item 2
...""")
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert in reasoning distillation. You will be given a user query and a trajectory that failed. You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future."),
                ("user", f"""PROBLEM: {question}
                
TRAJECTORY: {trajectory}

EXPECTED ANSWER: {expected_answer}

Extract lessons or preventive strategies. Format each memory item using Markdown:
# Memory Item 1
## Title: <concise name>
## Description: <one sentence summary>
## Content: <detailed preventive steps or checklist>

# Memory Item 2
...""")
            ])

        response = await self.llm.ainvoke(prompt.format())
        return self._parse_response(response.content, problem_id, success)

    def _parse_response(self, text: str, problem_id: str, success: bool) -> list[MemoryItem]:
        """Parse Markdown-style text into MemoryItem objects"""
        memories = []
        # Split by # Memory Item header
        items = text.split("# Memory Item")

        for item in items[1:]:
            try:
                title = ""
                description = ""
                content = []

                lines = item.strip().split("\n")
                collecting_content = False
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("## Title:"):
                        title = line.replace("## Title:", "").strip()
                    elif line.startswith("## Description:"):
                        description = line.replace("## Description:", "").strip()
                    elif line.startswith("## Content:"):
                        content.append(line.replace("## Content:", "").strip())
                        collecting_content = True
                    elif collecting_content and line:
                        # Append line to content if it's not another header
                        if not line.startswith("##"):
                            content.append(line)
                        else:
                            collecting_content = False

                if title and description and content:
                    memories.append(MemoryItem(
                        title=title,
                        description=description,
                        content="\n".join(content).strip(),
                        source_problem_id=problem_id,
                        success=success,
                        created_at=datetime.now().isoformat()
                    ))
            except Exception:
                continue
        return memories

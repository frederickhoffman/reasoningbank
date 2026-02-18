import asyncio
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .extraction import MemoryExtractor
from .judge import MathJudge, BaseJudge, get_judge
from .memory import ReasoningBank
from .retrieval import MemoryRetriever
from .state import AgentState


class AgentGraph:
    """LangGraph agent for ReasoningBank with MATTS scaling"""

    def __init__(
        self,
        bank: ReasoningBank,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        k: int = 3,
        dataset: str = "gsm8k"
    ):
        self.bank = bank
        self.retriever = MemoryRetriever(embedding_model)
        self.extractor = MemoryExtractor(model_name)
        self.judge: BaseJudge = get_judge(dataset)
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.k = k
        self.N = k
        self.max_refinements = k
        self.dataset = dataset

        # Build the graph
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("solve_parallel", self.solve_parallel_node)
        workflow.add_node("refine_sequential", self.refine_sequential_node)
        workflow.add_node("select_best", self.select_best_node)
        workflow.add_node("extract", self.extract_node)
        workflow.add_node("consolidate", self.consolidate_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "solve_parallel")
        workflow.add_edge("solve_parallel", "refine_sequential")
        
        # Router for sequential refinement
        workflow.add_conditional_edges(
            "refine_sequential",
            self.should_refine,
            {
                "refine": "refine_sequential",
                "select": "select_best"
            }
        )
        
        workflow.add_edge("select_best", "extract")
        workflow.add_edge("extract", "consolidate")
        workflow.add_edge("consolidate", END)

        self.graph = workflow.compile()

    def _get_system_message(self, phase: str = "solve") -> str:
        """Get dataset-specific system messages"""
        ds = self.dataset.lower()
        if "webarena" in ds:
            role = "expert web browsing assistant"
            instruction = "Navigate the web to achieve the user's intent. Provide your actions in a clear, executable format."
        elif "mind2web" in ds:
            role = "web operation expert"
            instruction = "Identify the correct element and action to perform based on the task intent and page state."
        elif "swe" in ds:
            role = "senior software engineer"
            instruction = "Resolve the issue described in the problem statement. Provide a patch or a sequence of bash commands to fix the bug."
        else:
            role = "expert math problem solver"
            instruction = "Solve the problem step-by-step. End your response with '#### <numeric_answer>'."

        if phase == "refine":
            return f"You are an {role} in correction mode. Your previous attempt was incorrect. Review your strategy, identify the mistake, and provide a corrected solution. {instruction}"
        
        return f"You are an {role}. {instruction}"

    async def retrieve_node(self, state: AgentState) -> dict[str, Any]:
        """Fetch top-1 relevant memory from ReasoningBank"""
        memories = self.bank.get_all()
        retrieved = await self.retriever.retrieve(
            query=state["question"],
            memories=memories,
            top_k=1,
            expected_answer=state.get("expected_answer")
        )
        state_memories = [m for m, score in retrieved]
        return {"retrieved_memories": state_memories, "refinement_steps": 0}

    async def solve_parallel_node(self, state: AgentState) -> dict[str, Any]:
        """Generate N independent trajectories (Parallel MATTS)"""
        formatted_memories = self.retriever.format_memories(
            [(m, 1.0) for m in state["retrieved_memories"]]
        )
        
        system_msg = self._get_system_message("solve")
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{system_msg}
            
Use the following past strategy hints if helpful:
            
{formatted_memories}"""),
            ("user", state["question"])
        ])
        
        # Generate N trajectories in parallel
        tasks = [self.llm.ainvoke(prompt.format()) for _ in range(self.N)]
        responses = await asyncio.gather(*tasks)
        trajectories = [r.content for r in responses]
        
        return {"trajectories": trajectories}

    async def refine_sequential_node(self, state: AgentState) -> dict[str, Any]:
        """Iteratively refine failed trajectories (Sequential MATTS)"""
        new_trajectories = []
        system_msg = self._get_system_message("refine")
        
        for traj in state["trajectories"]:
            # Check if current trajectory is correct
            is_correct = self.judge.is_correct(traj, state["expected_answer"] or "")
            
            if is_correct or state["refinement_steps"] >= self.max_refinements:
                new_trajectories.append(traj)
            else:
                # Refine
                refine_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_msg),
                    ("user", f"PROBLEM: {state['question']}\n\nPREVIOUS ATTEMPT: {traj}\n\nREVISED SOLUTION:")
                ])
                response = await self.llm.ainvoke(refine_prompt.format())
                new_trajectories.append(response.content)
                
        return {
            "trajectories": new_trajectories, 
            "refinement_steps": state["refinement_steps"] + 1
        }

    def should_refine(self, state: AgentState) -> str:
        """Check if any trajectory still needs refinement and we have budget"""
        if state["refinement_steps"] >= self.max_refinements:
            return "select"
            
        # If all are correct, stop refining
        all_correct = all(
            self.judge.is_correct(t, state["expected_answer"] or "") 
            for t in state["trajectories"]
        )
        return "select" if all_correct else "refine"

    async def select_best_node(self, state: AgentState) -> dict[str, Any]:
        """Select the best trajectory for final output and extraction"""
        best_traj = state["trajectories"][0]
        success = False
        
        # Prioritize any correct trajectory
        for traj in state["trajectories"]:
            if self.judge.is_correct(traj, state["expected_answer"] or ""):
                best_traj = traj
                success = True
                break
        
        evaluation = self.judge.evaluate(best_traj, state["expected_answer"] or "")
        return {"solution": best_traj, "success": success, "evaluation": evaluation}

    async def extract_node(self, state: AgentState) -> dict[str, Any]:
        """Distill memories from the best trajectory"""
        new_memories = await self.extractor.extract(
            problem_id=state["problem_id"],
            question=state["question"],
            trajectory=state["solution"],
            success=state["success"],
            expected_answer=state["expected_answer"] or ""
        )
        return {"new_memories": new_memories}

    async def consolidate_node(self, state: AgentState) -> dict[str, Any]:
        """Save new memories to the bank"""
        if state["new_memories"]:
            self.bank.add_memories(state["new_memories"])
        return {}

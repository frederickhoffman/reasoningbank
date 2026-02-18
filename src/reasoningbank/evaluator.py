from typing import Any

import wandb
from datasets import load_dataset
from tqdm import tqdm

from .agent import AgentGraph
from .state import AgentState


class ReasoningBankEvaluator:
    """Orchestrate evaluation of ReasoningBank across datasets"""

    def __init__(self, agent_graph: AgentGraph, project_name: str = "reasoning-bank"):
        self.agent_graph = agent_graph
        self.project_name = project_name

    async def evaluate_dataset(
        self,
        dataset_name: str = "gsm8k",
        split: str = "test",
        limit: int = 10,
        config: dict[str, Any] = None
    ) -> list[dict[str, Any]]:
        """Run evaluation on a dataset and log to W&B"""

        # Initialize wandb
        wandb.init(
            project=self.project_name,
            config=config or {"dataset": dataset_name, "split": split, "limit": limit}
        )

        # Load dataset
        if dataset_name == "gsm8k":
            ds = load_dataset("gsm8k", "main", split=split)
            question_key = "question"
            answer_key = "answer"
        elif dataset_name == "math":
            ds = load_dataset("competition_math", split=split)
            question_key = "problem"
            answer_key = "solution"
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        results = []
        success_count = 0

        # Run agent on dataset
        for i in tqdm(range(min(limit, len(ds))), desc=f"Evaluating {dataset_name}"):
            item = ds[i]
            question = item[question_key]
            expected = item[answer_key]

            initial_state: AgentState = {
                "problem_id": f"{dataset_name}_{i}",
                "question": question,
                "expected_answer": expected,
                "retrieved_memories": [],
                "trajectories": [],
                "refinement_steps": 0,
                "solution": "",
                "success": False,
                "evaluation": {},
                "new_memories": []
            }

            # Execute agent graph
            final_state = await self.agent_graph.graph.ainvoke(initial_state)

            results.append(final_state)
            if final_state["success"]:
                success_count += 1

            # Log detailed metrics to wandb
            extraction_success = len(final_state["new_memories"]) > 0
            wandb.log({
                "problem_id": final_state["problem_id"],
                "success": int(final_state["success"]),
                "refinement_steps_used": final_state["refinement_steps"],
                "num_trajectories_parallel": len(final_state["trajectories"]),
                "num_memories_retrieved": len(final_state["retrieved_memories"]),
                "num_memories_extracted": len(final_state["new_memories"]),
                "extraction_success": int(extraction_success),
                "total_memories_in_bank": len(self.agent_graph.bank)
            })

        # Log summary metrics
        accuracy = success_count / limit if limit > 0 else 0
        wandb.log({
            "final_accuracy": accuracy,
            "total_success": success_count,
            "total_limit": limit,
            "final_memory_bank_size": len(self.agent_graph.bank),
            "avg_refinement_steps": sum(r["refinement_steps"] for r in results) / limit if limit > 0 else 0
        })

        wandb.finish()
        return results

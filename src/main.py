import argparse
import asyncio

from dotenv import load_dotenv

from reasoningbank.agent import AgentGraph
from reasoningbank.evaluator import ReasoningBankEvaluator
from reasoningbank.memory import ReasoningBank


async def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run ReasoningBank evaluation")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset (gsm8k/math)")
    parser.add_argument("--limit", type=int, default=10, help="Number of problems to test")
    parser.add_argument("--bank", type=str, default="memory_bank.json", help="Memory bank file")
    parser.add_argument("--clear-bank", action="store_true", help="Clear memory bank before starting")

    parser.add_argument("--N", type=int, default=3, help="Parallel trajectories (N)")
    parser.add_argument("--max-refinements", type=int, default=5, help="Max sequential refinements")

    args = parser.parse_args()

    # Initialize components
    bank = ReasoningBank(args.bank)
    if args.clear_bank:
        bank.clear()

    agent = AgentGraph(bank, N=args.N, max_refinements=args.max_refinements)
    evaluator = ReasoningBankEvaluator(agent)

    # Run evaluation
    print(f"Starting evaluation on {args.dataset} (limit={args.limit})...")
    await evaluator.evaluate_dataset(
        dataset_name=args.dataset,
        limit=args.limit
    )
    print("Evaluation completed.")

if __name__ == "__main__":
    asyncio.run(main())

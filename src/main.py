import argparse
import asyncio

from dotenv import load_dotenv

from reasoningbank.agent import AgentGraph
from reasoningbank.evaluator import ReasoningBankEvaluator
from reasoningbank.memory import ReasoningBank


async def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run ReasoningBank evaluation")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset (gsm8k/math/webarena/mind2web/swebench)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--limit", type=int, default=10, help="Number of problems to test")
    parser.add_argument("--bank", type=str, default="memory_bank.json", help="Memory bank file")
    parser.add_argument("--clear-bank", action="store_true", help="Clear memory bank before starting")

    parser.add_argument("-k", "--k", type=int, default=3, help="MaTTS scaling factor (both parallel and sequential)")

    args = parser.parse_args()

    # Initialize components
    bank = ReasoningBank(args.bank)
    if args.clear_bank:
        bank.clear()

    agent = AgentGraph(bank, k=args.k, dataset=args.dataset, model_name=args.model)
    evaluator = ReasoningBankEvaluator(agent)

    # Run evaluation
    print(f"Starting evaluation on {args.dataset} (limit={args.limit})...")
    await evaluator.evaluate_dataset(
        dataset_name=args.dataset,
        limit=args.limit,
        config={
            "dataset": args.dataset,
            "limit": args.limit,
            "k": args.k
        }
    )
    print("Evaluation completed.")

if __name__ == "__main__":
    asyncio.run(main())

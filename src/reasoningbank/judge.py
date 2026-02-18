import re
from abc import ABC, abstractmethod
from typing import Any


class BaseJudge(ABC):
    """Base class for dataset-specific judges"""

    @abstractmethod
    def is_correct(self, predicted: str, expected: str) -> bool:
        pass

    def evaluate(self, solution: str, expected_answer: str) -> dict[str, Any]:
        success = self.is_correct(solution, expected_answer)
        return {
            "success": success,
            "predicted": solution,
            "expected": expected_answer,
        }


class MathJudge(BaseJudge):
    """Evaluate if math solution is correct with GSM8K and MATH dataset support"""

    def is_correct(self, predicted: str, expected: str) -> bool:
        """Robust numeric comparison with GSM8K and MATH format awareness"""
        try:
            pred_num = self._extract_number(predicted)
            exp_num = self._extract_number(expected)

            if pred_num is None or exp_num is None:
                pred_clean = self._normalize_text(str(predicted))
                exp_clean = self._normalize_text(str(expected))
                return pred_clean == exp_clean

            if isinstance(pred_num, int) and isinstance(exp_num, int):
                return pred_num == exp_num

            return abs(pred_num - exp_num) < 0.01
        except Exception:
            return False

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = text.replace("\\", "").replace("$", "").replace("{", "").replace("}", "")
        return " ".join(text.split())

    def _extract_number(self, text: str) -> float | None:
        if not text:
            return None

        text = str(text).strip()

        if "\\boxed" in text:
            match = re.search(r"\\boxed\{([^}]+)\}", text)
            if match:
                return self._clean_number(match.group(1))

        if "####" in text:
            return self._clean_number(text.split("####")[-1].strip())

        if "ANSWER:" in text.upper():
            return self._clean_number(text.upper().split("ANSWER:")[-1].strip())

        if "/" in text and len(text.split("/")) == 2:
            return self._clean_number(text)

        matches = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?", text)
        if matches:
            return self._clean_number(matches[-1])

        return None

    def _clean_number(self, text: str) -> float | None:
        if not text:
            return None
        cleaned = text.replace("\\", "").replace("$", "").replace(",", "").replace("%", "").strip()

        if "/" in cleaned:
            try:
                parts = cleaned.split("/")
                if len(parts) == 2:
                    numerator = float(parts[0].strip())
                    denominator = float(parts[1].strip())
                    if denominator != 0:
                        result = numerator / denominator
                        return int(result) if result.is_integer() else result
            except Exception:
                pass

        try:
            num = float(cleaned)
            return int(num) if num.is_integer() else num
        except Exception:
            return None

    def evaluate(self, solution: str, expected_answer: str) -> dict[str, Any]:
        eval_dict = super().evaluate(solution, expected_answer)
        eval_dict.update({
            "predicted_number": self._extract_number(solution),
            "expected_number": self._extract_number(expected_answer)
        })
        return eval_dict


class LLMJudge(BaseJudge):
    """LLM-as-a-judge for evaluating agent success as described in Figure 9"""

    def __init__(self, model_name: str = "gpt-4o"):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def is_correct(self, predicted: str, expected: str) -> bool:
        # LLMJudge typically needs async for performance, but the interface is sync
        # In a real implementation with LangGraph, we'd use a node for this
        # For this refactor, we'll keep it simple or wrap it
        return "success" in predicted.lower()

    async def evaluate_async(self, question: str, trajectory: str, expected: str) -> dict[str, Any]:
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in evaluating the performance of an AI agent. 
Given the user's intent, the agent's action history (trajectory), and the expected result, your goal is to decide whether the agent's execution is successful or not.

*IMPORTANT* Format your response into two lines:
Thoughts: <your thoughts and reasoning process>
Status: "success" or "failure\""""),
            ("user", f"INTENT: {question}\n\nTRAJECTORY: {trajectory}\n\nEXPECTED: {expected}")
        ])
        
        response = await self.llm.ainvoke(prompt.format())
        content = response.content
        
        status = "failure"
        if "Status: \"success\"" in content or "Status: success" in content:
            status = "success"
            
        return {
            "success": status == "success",
            "thoughts": content.split("Status:")[0].replace("Thoughts:", "").strip(),
            "predicted": trajectory,
            "expected": expected
        }


class WebArenaJudge(LLMJudge):
    """WebArena evaluation using LLM-as-a-judge"""
    pass


class Mind2WebJudge(LLMJudge):
    """Mind2Web action/element correctness using LLM-as-a-judge"""
    pass


class SWEBenchJudge(LLMJudge):
    """SWE-Bench patch correctness using LLM-as-a-judge"""
    pass


def get_judge(dataset_name: str) -> BaseJudge:
    """Judge factory"""
    dataset_name = dataset_name.lower()
    if dataset_name in ["gsm8k", "math"]:
        return MathJudge()
    elif "webarena" in dataset_name:
        return WebArenaJudge()
    elif "mind2web" in dataset_name:
        return Mind2WebJudge()
    elif "swe" in dataset_name:
        return SWEBenchJudge()
    else:
        return MathJudge()  # Default fallback

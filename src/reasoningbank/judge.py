import re
from typing import Any


class MathJudge:
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
        success = self.is_correct(solution, expected_answer)
        return {
            "success": success,
            "predicted": solution,
            "expected": expected_answer,
            "predicted_number": self._extract_number(solution),
            "expected_number": self._extract_number(expected_answer)
        }

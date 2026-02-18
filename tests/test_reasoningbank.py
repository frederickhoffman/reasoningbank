import pytest
import os
from reasoningbank.memory import ReasoningBank, MemoryItem
from reasoningbank.judge import MathJudge
from datetime import datetime

@pytest.fixture
def temp_bank():
    bank_path = "test_bank.json"
    bank = ReasoningBank(bank_path)
    yield bank
    if os.path.exists(bank_path):
        os.remove(bank_path)

def test_memory_bank_save_load(temp_bank):
    item = MemoryItem(
        title="Test Strategy",
        description="A test strategy",
        content="Step 1, Step 2",
        source_problem_id="prob_1",
        success=True,
        created_at=datetime.now().isoformat()
    )
    temp_bank.add_memory(item)
    
    # Reload
    new_bank = ReasoningBank("test_bank.json")
    assert len(new_bank) == 1
    assert new_bank.get_all()[0].title == "Test Strategy"

def test_math_judge():
    judge = MathJudge()
    
    # Test GSM8K format
    assert judge.is_correct("The answer is #### 42", "42")
    assert judge.is_correct("#### 42", "The solution is 42")
    
    # Test MATH format
    assert judge.is_correct("Therefore \\boxed{1/2}", "0.5")
    
    # Test numeric cleanup
    assert judge.is_correct("The result is 1,234.5", "1234.5")
    
    # Test negative match
    assert not judge.is_correct("#### 42", "43")

def test_math_judge_complex():
    judge = MathJudge()
    assert judge.is_correct("The final answer is #### 3/4", "0.75")
    assert judge.is_correct("Final Answer: $100", "100")
    assert judge.is_correct("2,500", "2500")

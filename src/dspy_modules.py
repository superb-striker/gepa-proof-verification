import dspy

from dspy_signatures import (
    AssessApproximation,
    AssessComputation,
    AssessEquivalence,
    AssessLogicalGap,
    AssessToyCase,
    ExtractFinalAnswer,
)


def parse_boolean(output_str: str) -> bool:
    """Helper to safely parse the string output from DSPy into a Python boolean."""
    if not output_str:
        return False
    cleaned = str(output_str).strip().lower().replace(".", "")
    return cleaned == "true"

def get_reasoning(result):
    """Safely extracts the reasoning field regardless of DSPy version."""
    return getattr(result, 'reasoning', getattr(result, 'rationale', "No reasoning generated."))

# 1. FINAL ANSWER JUDGE
class FinalAnswerJudge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractFinalAnswer)
        self.assess = dspy.ChainOfThought(AssessEquivalence)
    def forward(self, problem: str, proof: str, ground_truth: str):
        ext_result = self.extract(problem=problem, proof=proof)
        extracted = ext_result.extracted_answer
        assess_result = self.assess(ground_truth=ground_truth, extracted_answer=extracted)
        is_equiv = parse_boolean(assess_result.is_equivalent)
        return dspy.Prediction(
            extraction_reasoning=get_reasoning(ext_result),
            extracted_answer=extracted,
            assessment_reasoning=get_reasoning(assess_result),
            is_equivalent=is_equiv
        )

# 2. TOY CASE JUDGE
class ToyCaseJudge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(AssessToyCase)
    def forward(self, problem: str, proof: str):
        result = self.prog(problem=problem, proof=proof)
        return dspy.Prediction(
            reasoning=get_reasoning(result), 
            uses_toy_case=parse_boolean(result.uses_toy_case)
        )

# 3. LOGICAL GAP JUDGE
class LogicalGapJudge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(AssessLogicalGap)
    def forward(self, problem: str, proof: str):
        result = self.prog(problem=problem, proof=proof)
        return dspy.Prediction(
            reasoning=get_reasoning(result), 
            has_logical_gap=parse_boolean(result.has_logical_gap)
        )

# 4. NUMERICAL APPROXIMATION JUDGE
class ApproximationJudge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(AssessApproximation)
    def forward(self, problem: str, proof: str):
        result = self.prog(problem=problem, proof=proof)
        return dspy.Prediction(
            reasoning=get_reasoning(result), 
            uses_illegal_approximation=parse_boolean(result.uses_illegal_approximation)
        )

# 5. COMPUTATION JUDGE
class ComputationJudge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(AssessComputation)
    def forward(self, problem: str, proof: str):
        result = self.prog(problem=problem, proof=proof)
        return dspy.Prediction(
            reasoning=get_reasoning(result), 
            has_computation_error=parse_boolean(result.has_computation_error)
        )

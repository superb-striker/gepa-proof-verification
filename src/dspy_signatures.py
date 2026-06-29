import dspy

# 1. FINAL ANSWER JUDGE

class ExtractFinalAnswer(dspy.Signature):
    """
    You are an expert mathematical evaluator. Extract the final mathematical answer or multiple-choice letter from the provided proof.
    For Relation problems, extract the option letter (e.g., A, B, C, D, E). 
    For Bound problems, extract the exact value of C (e.g., 1, \\sqrt{2}).
    """
    problem = dspy.InputField(desc="The original inequality problem.")
    proof = dspy.InputField(desc="The generated mathematical proof.")
    extracted_answer = dspy.OutputField(desc="The extracted final answer or option letter.")

class AssessEquivalence(dspy.Signature):
    """
    Verify if the extracted answer is mathematically equivalent to the ground truth.
    Respond with True if they represent the exact same value or relation, and False otherwise.
    """
    ground_truth = dspy.InputField(desc="The correct answer.")
    extracted_answer = dspy.InputField(desc="The answer extracted from the model's proof.")
    is_equivalent = dspy.OutputField(desc="True if mathematically equivalent, False otherwise.", prefix="Equivalent:")

# 2. TOY CASE JUDGE

class AssessToyCase(dspy.Signature):
    """
    Evaluate the logical rigor of a mathematical proof. 
    Determine if the proof improperly relies on plugging in specific numerical values (toy cases) or extreme values to justify the general inequality direction, rather than providing a rigorous algebraic proof.
    Testing specific values is only acceptable if used merely to explore or find an equality case, NOT to conclude the final relation.
    """
    problem = dspy.InputField(desc="The original inequality problem.")
    proof = dspy.InputField(desc="The generated mathematical proof.")
    uses_toy_case = dspy.OutputField(desc="True if the proof relies on a toy case to justify the final conclusion, False if the reasoning is general and valid.", prefix="Uses Toy Case:")

# 3. LOGICAL GAP JUDGE

class AssessLogicalGap(dspy.Signature):
    """
    Evaluate the mathematical reasoning of a proof.
    Determine if the proof contains logical leaps, skips crucial algebraic derivations (like expansions or inductions), or makes non-trivial claims without showing the supporting work.
    Standard, well-known algebraic simplifications are allowed without full derivation.
    """
    problem = dspy.InputField(desc="The original inequality problem.")
    proof = dspy.InputField(desc="The generated mathematical proof.")
    has_logical_gap = dspy.OutputField(desc="True if there are unjustified claims or skipped derivations, False if the logic flows soundly.", prefix="Has Logical Gap:")

# 4. NUMERICAL APPROXIMATION JUDGE

class AssessApproximation(dspy.Signature):
    """
    Evaluate the exactness of a mathematical proof.
    Determine if the proof improperly replaces exact symbolic expressions (like fractions, \\pi, or radicals like \\sqrt{2}) with decimal approximations (e.g., 1.414) mid-proof to justify an algebraic step.
    Exact symbolic reasoning must be maintained throughout the proof.
    """
    problem = dspy.InputField(desc="The original inequality problem.")
    proof = dspy.InputField(desc="The generated mathematical proof.")
    uses_illegal_approximation = dspy.OutputField(desc="True if illegal decimal approximations are used in operations, False if exact math is maintained.", prefix="Uses Illegal Approximation:")

# 5. COMPUTATION JUDGE

class AssessComputation(dspy.Signature):
    """
    Evaluate the arithmetic and basic algebra of a mathematical proof.
    Determine if the proof contains a clear, objective arithmetic mistake or an incorrect basic algebraic expansion (e.g., stating 2+2=5, or incorrectly expanding a polynomial).
    """
    problem = dspy.InputField(desc="The original inequality problem.")
    proof = dspy.InputField(desc="The generated mathematical proof.")
    has_computation_error = dspy.OutputField(desc="True if there is an arithmetic or expansion error, False if computations are correct.", prefix="Has Computation Error:")

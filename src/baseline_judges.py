import os

from dotenv import load_dotenv
from google import genai
from groq import Groq
from openai import OpenAI

# Load environment variables
load_dotenv()

# Select provider from .env, defaulting to nvidia
PROVIDER = os.getenv("BASELINE_PROVIDER", "nvidia").lower().strip()

class BaselineJudges:
    """
    A suite of unoptimized, zero-shot judges to evaluate mathematical proofs.
    These represent the 'naive' approach before applying DSPy/GEPA optimization.
    """
    @staticmethod
    def _call_llm(prompt: str) -> bool:
        """Helper to call the selected LLM and parse a boolean response."""
        try:
            if PROVIDER == "ollama":
                base_url = os.getenv("OLLAMA_URL")
                client = OpenAI(base_url=base_url, api_key="ollama")
                model_name = os.getenv("BASELINE_MODEL", "qwen2.5-coder:14b")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0
                )
                content = response.choices[0].message.content

            elif PROVIDER == "nvidia" or PROVIDER == "nim":
                client = OpenAI(
                    api_key=os.getenv("NVIDIA_API_KEY"),
                    base_url="https://integrate.api.nvidia.com/v1"
                )
                model_name = os.getenv("BASELINE_MODEL", "meta/llama-3.1-70b-instruct")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0
                )
                content = response.choices[0].message.content

            elif PROVIDER == "groq":
                client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                model_name = os.getenv("BASELINE_MODEL", "llama3-70b-8192")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0
                )
                content = response.choices[0].message.content

            elif PROVIDER == "gemini":
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                # Switching to the new fast/lite model 
                model_name = os.getenv("BASELINE_MODEL", "gemini-2.5-flash")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                content = response.text

            else:
                raise ValueError(f"Wrong input in provider: {PROVIDER}")

            # Naive parsing: just check if 'true' is in the text
            text = (content or "").strip().lower()
            return "true" in text
            
        except Exception:
            print(f"Unexpected error while using provider {PROVIDER}")
            raise

    @classmethod
    def evaluate_final_answer(cls, problem: str, proof: str, ground_truth: str) -> bool:
        """Judge 1: Checks if the final answer matches the ground truth."""
        prompt = (
            f"Problem: {problem}\n"
            f"Proof: {proof}\n"
            f"Ground Truth: {ground_truth}\n\n"
            "Does the final answer in the proof mathematically match the ground truth exactly? "
            "Respond with ONLY 'True' or 'False'."
        )
        return cls._call_llm(prompt)

    @classmethod
    def evaluate_toy_case(cls, problem: str, proof: str) -> bool:
        """Judge 2: Checks if the proof relies on plugging in specific numbers."""
        prompt = (
            f"Problem: {problem}\n"
            f"Proof: {proof}\n\n"
            "Does this proof improperly rely on plugging in specific numerical values (a 'toy case') "
            "to justify the general inequality, rather than using a rigorous algebraic proof? "
            "Respond with ONLY 'True' or 'False'."
        )
        return cls._call_llm(prompt)

    @classmethod
    def evaluate_logical_gap(cls, problem: str, proof: str) -> bool:
        """Judge 3: Checks for skipped steps or unjustified claims."""
        prompt = (
            f"Problem: {problem}\n"
            f"Proof: {proof}\n\n"
            "Does this proof contain logical leaps, skip crucial algebraic derivations, "
            "or make non-trivial claims without showing the supporting work? "
            "Respond with ONLY 'True' or 'False'."
        )
        return cls._call_llm(prompt)

    @classmethod
    def evaluate_approximation(cls, problem: str, proof: str) -> bool:
        """Judge 4: Checks for illegal decimal approximations."""
        prompt = (
            f"Problem: {problem}\n"
            f"Proof: {proof}\n\n"
            "Does this proof improperly replace exact symbolic expressions (like fractions or radicals) "
            "with decimal approximations (e.g., 1.414) mid-proof to justify an algebraic step? "
            "Respond with ONLY 'True' or 'False'."
        )
        return cls._call_llm(prompt)

    @classmethod
    def evaluate_computation(cls, problem: str, proof: str) -> bool:
        """Judge 5: Checks for arithmetic or basic algebra errors."""
        prompt = (
            f"Problem: {problem}\n"
            f"Proof: {proof}\n\n"
            "Does this proof contain a clear arithmetic mistake or an incorrect basic algebraic expansion? "
            "Respond with ONLY 'True' or 'False'."
        )
        return cls._call_llm(prompt)

if __name__ == "__main__":
    # A quick dummy test to ensure the API and parsing work
    dummy_problem = "Prove that x^2 >= 0 for all real x."
    dummy_proof = "Let x = 2. 2^2 = 4. 4 >= 0. Since it works for 2, it works for all x. The answer is True."
    
    print(f"Testing Baseline Toy Case Judge using {PROVIDER.upper()}...")
    is_toy_case = BaselineJudges.evaluate_toy_case(dummy_problem, dummy_proof)
    print(f"Detected Toy Case? {is_toy_case} (Expected: True)")

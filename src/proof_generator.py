import json
import os
import time

from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

# Configure Gemini
client = genai.Client(api_key=api_key)

# PHASE 1: REFORMULATING THE INEQUALITIES (Base Prompts)

BASE_PROMPTS = {
    "cauchy_schwarz": {
        "type": "relation",
        "ground_truth": "(D) \\ge",
        "prompt": (
            "Let $u_1, u_2$ and $v_1, v_2$ be real numbers. Consider the following expression: "
            "$(u_1^2 + u_2^2)(v_1^2 + v_2^2) \\text{ [blank] } (u_1v_1 + u_2v_2)^2$. "
            "Determine the correct inequality relation to fill in the blank. "
            "Options: (A) $<$, (B) $\\le$, (C) $=$, (D) $\\ge$, (E) $>$. "
            "End your response EXACTLY with 'The answer is (Letter) Symbol'."
        )
    },
    "bernoulli": {
        "type": "bound",
        "ground_truth": "C = 1",
        "prompt": (
            "Let $x$ be a real number such that $x \\ge -1$, and let $r \\ge 1$ be a real number. "
            "Determine the minimal constant $C$ such that the following inequality holds for all valid $x$ and $r$: "
            "$(1+x)^r - rx \\ge C$. "
            "End your response EXACTLY with 'The answer is C = Value'."
        )
    },
    #  Held-out test inequalities (never seen during optimization)
    "triangle": {
        "type": "relation",
        "ground_truth": "(B) \\le",
        "prompt": (
            "Let $a, b$ be real numbers. Consider the following expression: "
            "$|a + b| \\text{ [blank] } |a| + |b|$. "
            "Determine the correct inequality relation to fill in the blank. "
            "Options: (A) $<$, (B) $\\le$, (C) $=$, (D) $\\ge$, (E) $>$. "
            "End your response EXACTLY with 'The answer is (Letter) Symbol'."
        )
    },
    "jensens": {
        "type": "relation",
        "ground_truth": "(D) \\ge",
        "prompt": (
            "Let $f$ be a convex function on $\\mathbb{R}$, let $x_1, x_2 \\in \\mathbb{R}$, "
            "and let $t \\in [0, 1]$. Consider the following expression: "
            "$t f(x_1) + (1-t) f(x_2) \\text{ [blank] } f(t x_1 + (1-t) x_2)$. "
            "Determine the correct inequality relation to fill in the blank. "
            "Options: (A) $<$, (B) $\\le$, (C) $=$, (D) $\\ge$, (E) $>$. "
            "End your response EXACTLY with 'The answer is (Letter) Symbol'."
        )
    },
}

# PHASE 2: GENERATING THE DATASET (Good & Flawed Instructions)

INSTRUCTIONS = [
    {
        "proof_type": "good",
        "flaw_type": "none",
        "instruction": "Provide clear, rigorous, and logically sound algebraic steps to prove this."
    },
    {
        "proof_type": "bad",
        "flaw_type": "toy_case",
        "instruction": "Solve this by ONLY plugging in specific, simple integers (like 1 and 2) for the variables. "
                       "Assume the relation holds for all numbers based purely on this single numerical test. "
                       "Do not provide a general algebraic proof."
    },
    {
        "proof_type": "bad",
        "flaw_type": "logical_gap",
        "instruction": "Start the proof correctly, but skip the actual core derivation (like expansion, induction, or derivatives) entirely. "
                       "Just write 'By obvious simplification, we arrive at the answer' and give the final answer."
    },
    {
        "proof_type": "bad",
        "flaw_type": "approximation",
        "instruction": "Provide a proof, but halfway through, pretend one of the variables is a specific number like 2, "
                       "and deliberately use a decimal approximation (e.g., substituting 1.414 for $\\sqrt{2}$) to justify the next algebraic step. "
                       "This should compromise the exact symbolic rigor of the proof."
    },
    {
        "proof_type": "bad",
        "flaw_type": "computation_error",
        "instruction": "Provide a mostly correct proof, but make a deliberate, obvious arithmetic or algebraic expansion mistake "
                       "near the end before concluding with the correct final answer. (e.g., state that 2 + 2 = 5 during a simplification)."
    }
]

# EXECUTION LOGIC

def generate_proof(inequality_id, base_data, config):
    """Calls Gemini to generate a specific type of proof based on instructions."""
    full_prompt = f"{base_data['prompt']}\n\n**INSTRUCTION FOR THIS GENERATION:**\n{config['instruction']}"
    print(f"Generating [{config['proof_type'].upper()} - {config['flaw_type']}] proof for {inequality_id}...")
    try:
        response = client.models.generate_content(model="gemini-3-flash-preview", contents=full_prompt)
        proof_text = response.text
    except Exception:
        print("Error generating proof")
        raise
    # Return structured dictionary for our JSON dataset
    return {
        "id": f"{inequality_id}_{config['flaw_type']}",
        "inequality": inequality_id,
        "task_type": base_data["type"],
        "ground_truth": base_data["ground_truth"],
        "proof_type": config["proof_type"],
        "flaw_type": config["flaw_type"],
        "prompt_used": full_prompt,
        "generated_proof": proof_text
    }

def main():
    dataset = []
    # Loop through both inequalities and all instruction sets
    for ineq_id, base_data in BASE_PROMPTS.items():
        for config in INSTRUCTIONS:
            data_entry = generate_proof(ineq_id, base_data, config)
            dataset.append(data_entry)
            # Sleep briefly to avoid hitting API rate limits
            time.sleep(3)
    # Save to raw_proofs.json
    output_dir = os.path.join("artifacts", "data_curated")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join("artifacts/data_curated", "raw_proofs.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    n_train = sum(1 for d in dataset if d["inequality"] in ("cauchy_schwarz", "bernoulli"))
    n_test = sum(1 for d in dataset if d["inequality"] in ("triangle", "jensens"))
    print(f"\n Dataset generation complete! Saved {len(dataset)} proofs to {output_path}")
    print(f"   ({n_train} train examples: cauchy_schwarz, bernoulli)")
    print(f"   ({n_test} held-out test examples: triangle, jensens - never used for optimization)")

if __name__ == "__main__":
    main()

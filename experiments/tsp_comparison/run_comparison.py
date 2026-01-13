#!/usr/bin/env python3
"""
TSP Improvement Comparison: Paper-Based vs ShinkaEvolve

This script compares two approaches for improving a TSP solver:
1. Paper-Based: Single LLM call with research paper ideas
2. ShinkaEvolve: Iterative LLM-based evolutionary search

Supports: OpenRouter, Anthropic, OpenAI APIs
"""

import os
import sys
import json
import time
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded API key from {env_path}")
except ImportError:
    pass  # dotenv not installed, rely on environment variables

# Try to import LLM clients
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class ExperimentResult:
    approach: str
    llm_calls: int
    tokens_used: int
    time_seconds: float
    score: float
    improvement_pct: float
    code_generated: str
    valid: bool
    error: Optional[str] = None


# ============================================================================
# LLM CLIENT SETUP
# ============================================================================

def get_llm_client() -> tuple:
    """
    Get LLM client based on available API keys.
    Priority: OpenRouter > Anthropic > OpenAI

    Returns:
        (client, client_type, default_model)
    """
    # Check OpenRouter first (most flexible)
    if os.environ.get('OPENROUTER_API_KEY'):
        if not HAS_OPENAI:
            print("ERROR: openai package required for OpenRouter")
            return None, None, None
        client = openai.OpenAI(
            api_key=os.environ['OPENROUTER_API_KEY'],
            base_url="https://openrouter.ai/api/v1"
        )
        return client, "openrouter", "anthropic/claude-sonnet-4"

    # Check Anthropic
    if os.environ.get('ANTHROPIC_API_KEY'):
        if not HAS_ANTHROPIC:
            print("ERROR: anthropic package required")
            return None, None, None
        client = anthropic.Anthropic()
        return client, "anthropic", "claude-sonnet-4-20250514"

    # Check OpenAI
    if os.environ.get('OPENAI_API_KEY'):
        if not HAS_OPENAI:
            print("ERROR: openai package required")
            return None, None, None
        client = openai.OpenAI()
        return client, "openai", "gpt-4o"

    return None, None, None


def call_llm(client, client_type: str, model: str, prompt: str, max_tokens: int = 4096) -> tuple:
    """
    Unified LLM call interface.

    Returns:
        (response_text, tokens_used, error)
    """
    try:
        if client_type == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text, response.usage.input_tokens + response.usage.output_tokens, None

        elif client_type in ("openrouter", "openai"):
            # OpenRouter and OpenAI use the same interface
            extra_headers = {}
            if client_type == "openrouter":
                extra_headers = {
                    "HTTP-Referer": "https://github.com/paperbench",
                    "X-Title": "PaperBench TSP Comparison"
                }

            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                extra_headers=extra_headers if extra_headers else None
            )
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            return content, tokens, None

        else:
            return None, 0, f"Unknown client type: {client_type}"

    except Exception as e:
        return None, 0, str(e)


# ============================================================================
# PAPER-BASED APPROACH
# ============================================================================

PAPER_TSP_IDEAS = """
## Paper: Lin-Kernighan Heuristic (1973)
Key ideas:
- Use k-opt moves with variable k (not just 2-opt)
- Sequential edge exchanges guided by gain criterion
- Backtracking when no improving move found

## Paper: Simulated Annealing for TSP (Kirkpatrick 1983)
Key ideas:
- Accept worse solutions with probability exp(-delta/T)
- Gradually decrease temperature
- Escape local optima through controlled randomization

## Paper: Or-opt moves
Key ideas:
- Relocate sequences of 1, 2, or 3 consecutive cities
- Often finds improvements 2-opt misses
- Lower complexity than 3-opt

## Paper: Don't-look bits (Bentley 1992)
Key ideas:
- Skip vertices unlikely to yield improvements
- Reset bit when neighbor changes
- Dramatically speeds up local search
"""

PAPER_PROMPT_TEMPLATE = """You are an expert algorithm engineer. Given a baseline TSP solver and research ideas,
generate an improved solver.

## Research Ideas to Incorporate
{paper_ideas}

## Baseline Implementation
```python
{baseline_code}
```

## Requirements
1. Keep the same interface: solve_tsp(coords, time_limit_ms) -> (tour, length)
2. Stay within the time limit
3. The code must be complete and runnable
4. Add improvements based on the research ideas above

## Output
Provide ONLY the improved code between ```python and ``` markers.
Focus on the most impactful improvement that can be implemented correctly.
"""


def paper_based_improvement(baseline_code: str, client, client_type: str, model: str) -> tuple:
    """
    Generate improved solver using paper-based approach.

    Returns:
        (improved_code, tokens_used, error)
    """
    prompt = PAPER_PROMPT_TEMPLATE.format(
        paper_ideas=PAPER_TSP_IDEAS,
        baseline_code=baseline_code
    )

    content, tokens, error = call_llm(client, client_type, model, prompt)

    if error:
        return None, tokens, error

    code = extract_python_code(content)
    if code:
        return code, tokens, None
    else:
        return None, tokens, "Could not extract Python code from response"


def extract_python_code(response: str) -> Optional[str]:
    """Extract Python code from markdown code blocks."""
    import re
    pattern = r'```python\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    return matches[0].strip() if matches else None


# ============================================================================
# SHINKAEVOLVE APPROACH (SIMPLIFIED)
# ============================================================================

SHINKA_MUTATION_PROMPT = """You are evolving a TSP solver. Given the current best code and its score,
generate a mutation that might improve it.

## Current Best Code (Score: {score:.2f})
```python
{code}
```

## Mutation Ideas
- Try different local search moves (3-opt, Or-opt)
- Add randomization to escape local optima
- Improve the construction heuristic
- Add caching or speedups

## Output
Provide ONLY the mutated code between ```python and ``` markers.
Make ONE targeted change that might improve the score.
"""


def shinka_evolution(
    baseline_code: str,
    client,
    client_type: str,
    model: str,
    evaluate_fn,
    budget: int = 10
) -> tuple:
    """
    Simplified ShinkaEvolve: iterative LLM mutations.

    Returns:
        (best_code, total_tokens, history, error)
    """
    best_code = baseline_code
    best_score = evaluate_fn(baseline_code)
    total_tokens = 0
    history = [(0, best_score, "initial")]

    print(f"  Initial score: {best_score:.4f}")

    for gen in range(1, budget + 1):
        prompt = SHINKA_MUTATION_PROMPT.format(
            code=best_code,
            score=best_score
        )

        content, tokens, error = call_llm(client, client_type, model, prompt)
        total_tokens += tokens

        if error:
            print(f"  Gen {gen}: LLM error - {error}")
            history.append((gen, 0, f"error: {error}"))
            continue

        new_code = extract_python_code(content)
        if new_code:
            new_score = evaluate_fn(new_code)

            if new_score > best_score:
                print(f"  Gen {gen}: {best_score:.4f} -> {new_score:.4f} (+{new_score - best_score:.4f})")
                best_code = new_code
                best_score = new_score
                history.append((gen, new_score, "improved"))
            else:
                print(f"  Gen {gen}: {new_score:.4f} (no improvement)")
                history.append((gen, new_score, "rejected"))
        else:
            print(f"  Gen {gen}: parse error")
            history.append((gen, 0, "parse_error"))

    return best_code, total_tokens, history, None


# ============================================================================
# EVALUATION
# ============================================================================

def quick_evaluate(code: str) -> float:
    """
    Quick evaluation of solver code.
    Returns score (higher is better).
    """
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Add run_solver wrapper if not present
        if 'run_solver' not in code:
            code += '''

def run_solver(instance):
    import numpy as np
    coords = np.array(instance['coords'])
    time_limit = instance.get('time_limit_ms', 5000)
    tour, length = solve_tsp(coords, time_limit)
    n = len(coords)
    valid = (len(tour) == n and set(tour) == set(range(n)))
    return {'tour': tour, 'length': length, 'valid': valid}
'''
        f.write(code)
        temp_path = f.name

    try:
        # Run evaluation
        result = subprocess.run(
            ['python3', 'evaluate_tsp.py', '--program_path', temp_path, '--results_dir', '/tmp/eval'],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.path.dirname(__file__)
        )

        # Parse score from output
        for line in result.stdout.split('\n'):
            if 'Overall Score:' in line:
                score = float(line.split(':')[1].strip())
                return score

        return 0.0

    except Exception as e:
        print(f"    Eval error: {e}")
        return 0.0

    finally:
        os.unlink(temp_path)


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def run_comparison(
    budget: int = 10,
    model: str = None,
) -> Dict[str, Any]:
    """
    Run the full comparison experiment.
    """
    # Load baseline
    baseline_path = os.path.join(os.path.dirname(__file__), 'tsp_initial.py')
    with open(baseline_path) as f:
        baseline_code = f.read()

    # Extract just the evolve block
    start_marker = "# EVOLVE-BLOCK-START"
    end_marker = "# EVOLVE-BLOCK-END"
    start_idx = baseline_code.find(start_marker)
    end_idx = baseline_code.find(end_marker) + len(end_marker)
    evolve_code = baseline_code[start_idx:end_idx]

    # Initialize LLM client
    client, client_type, default_model = get_llm_client()

    if client is None:
        print("ERROR: No LLM API key found!")
        print("Set one of: OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY")
        print("\nCreate a .env file with:")
        print("  OPENROUTER_API_KEY=your-key-here")
        return None

    # Use provided model or default
    model = model or default_model
    print(f"Using {client_type} API with model: {model}")

    results = {
        'baseline_score': quick_evaluate(baseline_code),
        'budget': budget,
        'model': model,
        'client_type': client_type,
        'approaches': {}
    }

    print(f"\nBaseline score: {results['baseline_score']:.4f}")

    # -------------------------------------------------------------------------
    # Approach 1: Paper-Based (1 LLM call)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("APPROACH 1: Paper-Based (1 LLM call)")
    print("="*60)

    start_time = time.time()
    improved_code, tokens, error = paper_based_improvement(
        evolve_code, client, client_type, model
    )
    paper_time = time.time() - start_time

    if improved_code:
        paper_score = quick_evaluate(improved_code)
        paper_improvement = (paper_score - results['baseline_score']) / results['baseline_score'] * 100

        results['approaches']['paper_based'] = asdict(ExperimentResult(
            approach='paper_based',
            llm_calls=1,
            tokens_used=tokens,
            time_seconds=paper_time,
            score=paper_score,
            improvement_pct=paper_improvement,
            code_generated=improved_code[:500] + "...",
            valid=True
        ))
        print(f"Score: {paper_score:.4f} (improvement: {paper_improvement:+.1f}%)")
    else:
        results['approaches']['paper_based'] = asdict(ExperimentResult(
            approach='paper_based',
            llm_calls=1,
            tokens_used=tokens,
            time_seconds=paper_time,
            score=0,
            improvement_pct=0,
            code_generated="",
            valid=False,
            error=error
        ))
        print(f"FAILED: {error}")

    # -------------------------------------------------------------------------
    # Approach 2: ShinkaEvolve (N LLM calls)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"APPROACH 2: ShinkaEvolve ({budget} LLM calls)")
    print("="*60)

    start_time = time.time()
    evolved_code, tokens, history, error = shinka_evolution(
        evolve_code, client, client_type, model, quick_evaluate, budget=budget
    )
    shinka_time = time.time() - start_time

    if evolved_code:
        shinka_score = quick_evaluate(evolved_code)
        shinka_improvement = (shinka_score - results['baseline_score']) / results['baseline_score'] * 100

        results['approaches']['shinka_evolve'] = asdict(ExperimentResult(
            approach='shinka_evolve',
            llm_calls=budget,
            tokens_used=tokens,
            time_seconds=shinka_time,
            score=shinka_score,
            improvement_pct=shinka_improvement,
            code_generated=evolved_code[:500] + "...",
            valid=True
        ))
        print(f"Final score: {shinka_score:.4f} (improvement: {shinka_improvement:+.1f}%)")
    else:
        results['approaches']['shinka_evolve'] = asdict(ExperimentResult(
            approach='shinka_evolve',
            llm_calls=budget,
            tokens_used=tokens,
            time_seconds=shinka_time,
            score=0,
            improvement_pct=0,
            code_generated="",
            valid=False,
            error=error
        ))
        print(f"FAILED: {error}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\n{'Approach':<20} {'LLM Calls':<12} {'Score':<12} {'Improvement':<12} {'Efficiency':<12}")
    print("-"*68)

    for name, data in results['approaches'].items():
        if data['valid']:
            efficiency = data['improvement_pct'] / data['llm_calls'] if data['llm_calls'] > 0 else 0
            print(f"{name:<20} {data['llm_calls']:<12} {data['score']:<12.4f} {data['improvement_pct']:<+11.1f}% {efficiency:<+11.2f}%/call")
        else:
            print(f"{name:<20} FAILED")

    return results


def main():
    parser = argparse.ArgumentParser(description="TSP Improvement Comparison")
    parser.add_argument('--budget', type=int, default=10,
                        help='Number of LLM calls for ShinkaEvolve')
    parser.add_argument('--model', type=str, default=None,
                        help='LLM model to use (default: auto-select based on API)')
    parser.add_argument('--output', type=str, default='comparison_results.json',
                        help='Output file for results')
    args = parser.parse_args()

    print("="*60)
    print("TSP IMPROVEMENT COMPARISON")
    print("Paper-Based vs ShinkaEvolve")
    print("="*60)

    results = run_comparison(budget=args.budget, model=args.model)

    if results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

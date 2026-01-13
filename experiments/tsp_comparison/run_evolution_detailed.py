#!/usr/bin/env python3
"""
Detailed ShinkaEvolve tracker - captures full evolution history with code diffs.
"""

import os
import sys
import json
import time
import difflib
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class Generation:
    gen: int
    score: float
    prev_score: float
    accepted: bool
    code: str
    mutation_description: str
    tokens_used: int
    time_seconds: float


@dataclass
class EvolutionHistory:
    baseline_code: str
    baseline_score: float
    generations: List[Generation] = field(default_factory=list)
    model: str = ""
    total_tokens: int = 0
    total_time: float = 0.0


def get_llm_client():
    if os.environ.get('OPENROUTER_API_KEY'):
        if not HAS_OPENAI:
            return None, None, None
        client = openai.OpenAI(
            api_key=os.environ['OPENROUTER_API_KEY'],
            base_url="https://openrouter.ai/api/v1"
        )
        return client, "openrouter", "anthropic/claude-sonnet-4"
    return None, None, None


def call_llm(client, model: str, prompt: str, max_tokens: int = 4096) -> tuple:
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            extra_headers={
                "HTTP-Referer": "https://github.com/paperbench",
                "X-Title": "PaperBench TSP Evolution"
            }
        )
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        return content, tokens, None
    except Exception as e:
        return None, 0, str(e)


def extract_python_code(response: str) -> Optional[str]:
    import re
    pattern = r'```python\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    return matches[0].strip() if matches else None


def quick_evaluate(code: str) -> float:
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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python3', 'evaluate_tsp.py', '--program_path', temp_path, '--results_dir', '/tmp/eval'],
            capture_output=True, text=True, timeout=60,
            cwd=os.path.dirname(__file__)
        )
        for line in result.stdout.split('\n'):
            if 'Overall Score:' in line:
                return float(line.split(':')[1].strip())
        return 0.0
    except Exception as e:
        return 0.0
    finally:
        os.unlink(temp_path)


MUTATION_PROMPT = """You are evolving a TSP solver. Given the current best code and its score,
generate a mutation that might improve it.

## Current Best Code (Score: {score:.4f})
```python
{code}
```

## Mutation Ideas
- Try different local search moves (3-opt, Or-opt)
- Add randomization to escape local optima
- Improve the construction heuristic
- Add caching or speedups
- Try different starting points

## Output Format
First, briefly describe (1 sentence) what mutation you're making.
Then provide the mutated code between ```python and ``` markers.
Make ONE targeted change that might improve the score.
"""


def extract_mutation_description(response: str) -> str:
    """Extract the mutation description from before the code block."""
    if '```python' in response:
        desc = response.split('```python')[0].strip()
        # Take last paragraph before code
        lines = [l.strip() for l in desc.split('\n') if l.strip()]
        if lines:
            return lines[-1][:200]
    return "Unknown mutation"


def run_detailed_evolution(budget: int = 20) -> EvolutionHistory:
    # Load baseline
    baseline_path = os.path.join(os.path.dirname(__file__), 'tsp_initial.py')
    with open(baseline_path) as f:
        baseline_code = f.read()

    # Extract evolve block
    start_marker = "# EVOLVE-BLOCK-START"
    end_marker = "# EVOLVE-BLOCK-END"
    start_idx = baseline_code.find(start_marker)
    end_idx = baseline_code.find(end_marker) + len(end_marker)
    evolve_code = baseline_code[start_idx:end_idx]

    # Init client
    client, client_type, model = get_llm_client()
    if client is None:
        print("ERROR: No API key found!")
        return None

    print(f"Using model: {model}")

    # Evaluate baseline
    baseline_score = quick_evaluate(baseline_code)
    print(f"Baseline score: {baseline_score:.4f}")

    history = EvolutionHistory(
        baseline_code=evolve_code,
        baseline_score=baseline_score,
        model=model
    )

    best_code = evolve_code
    best_score = baseline_score

    print(f"\n{'='*70}")
    print(f"Starting Evolution ({budget} generations)")
    print(f"{'='*70}\n")

    for gen in range(1, budget + 1):
        start_time = time.time()

        prompt = MUTATION_PROMPT.format(code=best_code, score=best_score)
        content, tokens, error = call_llm(client, model, prompt)

        gen_time = time.time() - start_time
        history.total_tokens += tokens
        history.total_time += gen_time

        if error:
            print(f"Gen {gen:2d}: ERROR - {error}")
            continue

        mutation_desc = extract_mutation_description(content)
        new_code = extract_python_code(content)

        if new_code:
            new_score = quick_evaluate(new_code)
            accepted = new_score > best_score

            gen_record = Generation(
                gen=gen,
                score=new_score,
                prev_score=best_score,
                accepted=accepted,
                code=new_code,
                mutation_description=mutation_desc,
                tokens_used=tokens,
                time_seconds=gen_time
            )
            history.generations.append(gen_record)

            if accepted:
                delta = new_score - best_score
                print(f"Gen {gen:2d}: {best_score:.4f} → {new_score:.4f} (+{delta:.4f}) ✓ ACCEPTED")
                print(f"        Mutation: {mutation_desc}")
                best_code = new_code
                best_score = new_score
            else:
                print(f"Gen {gen:2d}: {new_score:.4f} (best: {best_score:.4f}) ✗ rejected")
        else:
            print(f"Gen {gen:2d}: Parse error")

    return history


def visualize_evolution(history: EvolutionHistory):
    """Create ASCII visualization of evolution."""

    print(f"\n{'='*70}")
    print("EVOLUTION VISUALIZATION")
    print(f"{'='*70}\n")

    # Score timeline
    print("Score Timeline:")
    print("-" * 70)

    scores = [history.baseline_score]
    labels = ["Base"]

    for gen in history.generations:
        if gen.accepted:
            scores.append(gen.score)
            labels.append(f"G{gen.gen}")

    min_score = min(scores) - 0.01
    max_score = max(scores) + 0.01
    width = 50

    for i, (score, label) in enumerate(zip(scores, labels)):
        pos = int((score - min_score) / (max_score - min_score) * width)
        bar = "─" * pos + "●"
        print(f"{label:>5} │ {bar:<52} {score:.4f}")

    print()

    # Evolution tree (single lineage since greedy)
    print("Evolution Lineage (Greedy Single-Island):")
    print("-" * 70)
    print()
    print("    ┌─────────────────────────────────────────────────────────────┐")
    print("    │  This implementation uses GREEDY hill-climbing:             │")
    print("    │  • Single population (1 island)                             │")
    print("    │  • Always keeps best solution                               │")
    print("    │  • Mutations either accepted or discarded                   │")
    print("    │                                                             │")
    print("    │  Full ShinkaEvolve supports:                                │")
    print("    │  • Multiple islands with migration                          │")
    print("    │  • Crossover between solutions                              │")
    print("    │  • Diverse population maintenance                           │")
    print("    └─────────────────────────────────────────────────────────────┘")
    print()

    # Show lineage
    print("    Baseline")
    print("       │")
    print(f"       ▼  score: {history.baseline_score:.4f}")

    accepted_gens = [g for g in history.generations if g.accepted]
    for i, gen in enumerate(accepted_gens):
        is_last = (i == len(accepted_gens) - 1)
        connector = "└" if is_last else "├"
        print(f"       │")
        print(f"       {connector}── Gen {gen.gen}: {gen.mutation_description[:50]}")
        print(f"       {'   ' if is_last else '│  '}   score: {gen.prev_score:.4f} → {gen.score:.4f} (+{gen.score - gen.prev_score:.4f})")

    print()

    # Detailed mutations
    print("Detailed Mutation History:")
    print("-" * 70)

    for gen in accepted_gens:
        print(f"\n┌── Generation {gen.gen} {'─'*50}")
        print(f"│ Score: {gen.prev_score:.4f} → {gen.score:.4f} (+{gen.score - gen.prev_score:.4f})")
        print(f"│ Mutation: {gen.mutation_description}")
        print(f"│ Tokens: {gen.tokens_used}, Time: {gen.time_seconds:.1f}s")
        print("│")
        print("│ Code diff (key changes):")

        # Find previous accepted gen's code or baseline
        if gen == accepted_gens[0]:
            prev_code = history.baseline_code
        else:
            prev_idx = accepted_gens.index(gen) - 1
            prev_code = accepted_gens[prev_idx].code

        # Show diff
        prev_lines = prev_code.split('\n')
        new_lines = gen.code.split('\n')

        diff = list(difflib.unified_diff(prev_lines, new_lines, lineterm='', n=1))
        diff_lines = [l for l in diff if l.startswith('+') or l.startswith('-')]
        diff_lines = [l for l in diff_lines if not l.startswith('+++') and not l.startswith('---')]

        # Show up to 15 most relevant diff lines
        shown = 0
        for line in diff_lines[:20]:
            if line.strip() in ['+', '-']:
                continue
            prefix = "│   "
            if line.startswith('+'):
                print(f"{prefix}\033[32m{line}\033[0m")
            else:
                print(f"{prefix}\033[31m{line}\033[0m")
            shown += 1
            if shown >= 15:
                remaining = len(diff_lines) - 15
                if remaining > 0:
                    print(f"│   ... and {remaining} more changes")
                break

        print(f"└{'─'*69}")

    # Summary stats
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Total generations:     {len(history.generations)}")
    print(f"Accepted mutations:    {len(accepted_gens)} ({100*len(accepted_gens)/len(history.generations):.0f}%)")
    print(f"Rejected mutations:    {len(history.generations) - len(accepted_gens)}")
    print(f"Total improvement:     {history.baseline_score:.4f} → {accepted_gens[-1].score if accepted_gens else history.baseline_score:.4f}")
    print(f"Improvement %:         +{100*(accepted_gens[-1].score - history.baseline_score)/history.baseline_score:.2f}%" if accepted_gens else "0%")
    print(f"Total tokens used:     {history.total_tokens:,}")
    print(f"Total time:            {history.total_time:.1f}s")
    print(f"Avg time per gen:      {history.total_time/len(history.generations):.1f}s")


def save_history(history: EvolutionHistory, path: str):
    """Save full history to JSON."""
    data = {
        'baseline_code': history.baseline_code,
        'baseline_score': history.baseline_score,
        'model': history.model,
        'total_tokens': history.total_tokens,
        'total_time': history.total_time,
        'generations': [asdict(g) for g in history.generations]
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nFull history saved to: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=20)
    parser.add_argument('--output', type=str, default='evolution_history.json')
    args = parser.parse_args()

    history = run_detailed_evolution(budget=args.budget)

    if history:
        visualize_evolution(history)
        save_history(history, args.output)

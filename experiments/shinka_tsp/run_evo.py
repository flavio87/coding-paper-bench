#!/usr/bin/env python3
"""
Run ShinkaEvolve on TSP solver optimization.

Usage:
    python run_evo.py --generations 20 --islands 2
"""

import os
import argparse
from pathlib import Path
import numpy as np

# Load environment variables
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / "tsp_comparison" / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded API key from {env_path}")

# Mock the embedding client to avoid needing OpenAI API for embeddings
# This uses random embeddings which effectively disables similarity checking
class MockEmbeddingClient:
    def __init__(self, model_name="mock", verbose=False):
        self.model_name = model_name
        self.verbose = verbose
        self.total_cost = 0.0

    def get_embedding(self, code):
        """Return random embeddings - effectively disables similarity deduplication."""
        if isinstance(code, str):
            # Return random 1536-dim embedding (same as OpenAI)
            return list(np.random.randn(1536)), 0.0
        else:
            return [list(np.random.randn(1536)) for _ in code], 0.0

# Monkeypatch before importing shinka
import shinka.llm.embedding as embedding_module
embedding_module.EmbeddingClient = MockEmbeddingClient

from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig


TSP_TASK_SYSTEM_MSG = """You are an expert algorithm engineer specializing in combinatorial optimization and the Traveling Salesman Problem (TSP).

The current solver uses Nearest Neighbor construction + 2-opt local search. Your goal is to improve it significantly.

## State-of-the-Art Reference: Lin-Kernighan Algorithm

The best TSP heuristics (LKH) achieve <1% gap from optimal. Key techniques:

```
LIN-KERNIGHAN PSEUDOCODE:
1. Start with any tour
2. For each node i, attempt variable-depth improvement:
   - Break edge (t1, t2) where t1 = i
   - For depth d = 1 to max_depth:
     - Choose t_{2d+1} from candidate list (5-10 nearest neighbors of t_{2d})
     - Add edge (t_{2d}, t_{2d+1}), breaking edge (t_{2d+1}, t_{2d+2})
     - If closing the tour now gives positive gain, accept and restart
     - Otherwise continue deepening
   - Backtrack if no improvement found
3. Key insight: Don't stop at 2-opt or 3-opt - variable depth finds better moves

CANDIDATE LISTS (critical for speed):
- For each city, precompute 5-10 nearest neighbors
- Only consider these when searching for improving moves
- Reduces O(n²) search to O(n × k) where k ≈ 10
```

## Example: 3-opt Move Implementation

```python
def three_opt_move(tour, dist, i, j, k):
    '''Try reconnecting tour segments [0:i], [i:j], [j:k], [k:n] in best way'''
    n = len(tour)
    A, B, C, D = tour[i-1], tour[i], tour[j-1], tour[j]
    E, F = tour[k-1], tour[k % n]

    d0 = dist[A,B] + dist[C,D] + dist[E,F]  # current

    # Try all 4 reconnection patterns, return best improvement
    options = [
        dist[A,C] + dist[B,E] + dist[D,F],  # reverse [i:j]
        dist[A,D] + dist[E,B] + dist[C,F],  # reverse [j:k]
        dist[A,D] + dist[E,C] + dist[B,F],  # reverse both
        dist[A,E] + dist[D,B] + dist[C,F],  # reconnect differently
    ]
    best_gain = d0 - min(options)
    return best_gain, options.index(min(options))
```

## Key Directions to Explore

1. **Variable-depth search**: Don't stop at 2-opt. Implement 3-opt, 4-opt, or full LK moves
2. **Candidate lists**: Limit neighbor search to k-nearest (massive speedup)
3. **Don't-look bits**: Skip nodes that haven't changed since last improvement
4. **Better construction**: Greedy edge insertion, Christofides-inspired approaches
5. **Multi-start + best-of**: Try multiple starting tours, keep the best

## Performance Targets

Current baseline achieves 7-25% gap from optimal. Target: <5% gap.
- n=20: aim for <5% gap (currently 7.2%)
- n=50: aim for <5% gap (currently 9.4%)
- n=100: aim for <5% gap (currently 12.6%)

## Constraints

- Interface: solve_tsp(coords, time_limit_ms) -> (tour, length)
- Must complete within time_limit_ms
- Tour must visit all cities exactly once
- Minimize total tour length (lower is better)

## What NOT to Do (common failures)

- Don't use pure random search or simulated annealing alone (too slow to converge)
- Don't forget to validate tour after segment manipulations (off-by-one errors)
- Don't ignore the time limit - check it in inner loops
- Don't use O(n³) algorithms without candidate list pruning for n>50"""


def main():
    parser = argparse.ArgumentParser(description="Run ShinkaEvolve on TSP")
    parser.add_argument("--generations", type=int, default=20,
                        help="Number of evolution generations")
    parser.add_argument("--islands", type=int, default=2,
                        help="Number of evolution islands")
    parser.add_argument("--parallel", type=int, default=2,
                        help="Max parallel evaluation jobs")
    parser.add_argument("--results_dir", type=str, default="results_tsp",
                        help="Directory to save results")
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get('OPENROUTER_API_KEY'):
        print("ERROR: OPENROUTER_API_KEY not set!")
        print("Please set it in experiments/tsp_comparison/.env")
        return

    print("="*60)
    print("ShinkaEvolve TSP Optimization")
    print("="*60)
    print(f"Generations: {args.generations}")
    print(f"Islands: {args.islands}")
    print(f"Parallel jobs: {args.parallel}")
    print()

    # Job configuration
    job_config = LocalJobConfig(
        eval_program_path="evaluate.py",
        time="00:02:00",  # 2 minute timeout per evaluation
    )

    # Database configuration with multiple islands
    db_config = DatabaseConfig(
        db_path="evolution_db.sqlite",
        num_islands=args.islands,
        archive_size=30,
        # Inspiration parameters
        elite_selection_ratio=0.3,
        num_archive_inspirations=3,
        num_top_k_inspirations=2,
        # Island migration
        migration_interval=5,
        migration_rate=0.15,
        island_elitism=True,
        # Parent selection: weighted prioritization
        parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )

    # Evolution configuration
    evo_config = EvolutionConfig(
        task_sys_msg=TSP_TASK_SYSTEM_MSG,
        patch_types=["diff", "full"],
        patch_type_probs=[0.7, 0.3],
        num_generations=args.generations,
        max_parallel_jobs=args.parallel,
        max_patch_resamples=2,
        max_patch_attempts=3,
        job_type="local",
        language="python",
        # Use OpenRouter with Claude Opus 4.5
        llm_models=[
            "openrouter/anthropic/claude-opus-4.5",
        ],
        llm_kwargs=dict(
            temperatures=[0.3, 0.7, 1.0],
            max_tokens=8192,
        ),
        # Disable features that require OpenAI directly
        embedding_model=None,
        meta_rec_interval=None,  # Disable meta-recommendations
        init_program_path="initial.py",
        results_dir=args.results_dir,
    )

    # Create and run evolution
    runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )

    print("Starting evolution...")
    runner.run()
    print("\nEvolution complete!")
    print(f"Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()

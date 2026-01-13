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

The current solver uses Nearest Neighbor construction + 2-opt local search. Your goal is to improve it.

Key directions to explore:
1. **Better construction heuristics**: Christofides, insertion heuristics, savings algorithm
2. **Improved local search**: 3-opt, Or-opt (relocate segments of 1-3 cities), Lin-Kernighan moves
3. **Meta-heuristics**: Simulated annealing, genetic algorithms, tabu search
4. **Speed optimizations**: Don't-look bits, neighbor lists, incremental distance updates
5. **Multi-start strategies**: Try different starting cities, combine multiple solutions
6. **Hybrid approaches**: Combine construction with aggressive local search

Important constraints:
- Keep the same interface: solve_tsp(coords, time_limit_ms) -> (tour, length)
- Stay within the time limit
- The tour must visit all cities exactly once
- Minimize total tour length (lower is better)

Be creative and try different algorithmic improvements!"""


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
        # Use OpenRouter with Claude
        llm_models=[
            "openrouter/anthropic/claude-sonnet-4",
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

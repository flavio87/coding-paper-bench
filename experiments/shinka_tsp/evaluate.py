"""
Evaluator for TSP solver using ShinkaEvolve framework.

Tests the solver on multiple TSP instances and aggregates the scores.
"""

import os
import argparse
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from shinka.core import run_shinka_eval


# Test instances - mix of random and structured problems
TEST_INSTANCES = [
    # Small random instances
    {"name": "random_20", "n": 20, "seed": 42},
    {"name": "random_30", "n": 30, "seed": 123},
    {"name": "random_50", "n": 50, "seed": 456},
    # Medium instances
    {"name": "random_75", "n": 75, "seed": 789},
    {"name": "random_100", "n": 100, "seed": 101},
]


def generate_instance(n: int, seed: int) -> np.ndarray:
    """Generate a random TSP instance."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)) * 1000  # Scale to [0, 1000]


def compute_tour_length(coords: np.ndarray, tour: List[int]) -> float:
    """Compute the total length of a tour."""
    n = len(tour)
    length = 0.0
    for i in range(n):
        p1 = coords[tour[i]]
        p2 = coords[tour[(i + 1) % n]]
        length += np.sqrt(np.sum((p1 - p2) ** 2))
    return length


def validate_tsp_result(
    result: Dict[str, Any],
    instance_info: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Validate TSP result.

    Args:
        result: Dict with 'tour', 'length', 'valid', 'score'
        instance_info: Dict with instance metadata

    Returns:
        (is_valid, error_message)
    """
    if not isinstance(result, dict):
        return False, f"Result must be a dict, got {type(result)}"

    if 'tour' not in result:
        return False, "Result missing 'tour' key"

    if 'length' not in result:
        return False, "Result missing 'length' key"

    if not result.get('valid', False):
        return False, "Tour is invalid (wrong cities or count)"

    tour = result['tour']
    n = instance_info.get('n', len(tour))

    if len(tour) != n:
        return False, f"Tour has {len(tour)} cities, expected {n}"

    if set(tour) != set(range(n)):
        return False, "Tour doesn't visit all cities exactly once"

    return True, "Tour is valid"


def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    """
    Get kwargs for each evaluation run.
    Each run tests a different instance.
    """
    if run_index >= len(TEST_INSTANCES):
        run_index = run_index % len(TEST_INSTANCES)

    instance_info = TEST_INSTANCES[run_index]
    coords = generate_instance(instance_info['n'], instance_info['seed'])

    return {
        'coords': coords,
        'time_limit_ms': 5000,
        'optimal_length': None,  # We don't know optimal
    }


def aggregate_tsp_metrics(
    results: List[Dict[str, Any]],
    results_dir: str
) -> Dict[str, Any]:
    """
    Aggregate metrics from multiple TSP evaluations.

    Args:
        results: List of result dicts from run_experiment
        results_dir: Directory to save extra data

    Returns:
        Aggregated metrics dict
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results"}

    valid_results = [r for r in results if r.get('valid', False)]

    if not valid_results:
        return {
            "combined_score": 0.0,
            "error": "All solutions invalid",
            "num_valid": 0,
            "num_total": len(results),
        }

    # Compute average score across all valid instances
    scores = [r.get('score', 0.0) for r in valid_results]
    lengths = [r.get('length', 0.0) for r in valid_results]

    avg_score = np.mean(scores)
    avg_length = np.mean(lengths)

    # Combined score: average of individual scores
    # Higher is better (closer to optimal)
    combined_score = float(avg_score)

    public_metrics = {
        "avg_tour_length": float(avg_length),
        "num_instances_solved": len(valid_results),
        "individual_scores": [float(s) for s in scores],
    }

    private_metrics = {
        "individual_lengths": [float(l) for l in lengths],
    }

    metrics = {
        "combined_score": combined_score,
        "public": public_metrics,
        "private": private_metrics,
    }

    # Save extra data
    try:
        extra_file = os.path.join(results_dir, "extra.npz")
        np.savez(
            extra_file,
            scores=np.array(scores),
            lengths=np.array(lengths),
        )
    except Exception as e:
        metrics["extra_save_error"] = str(e)

    return metrics


def main(program_path: str, results_dir: str):
    """Run the TSP evaluation."""
    print(f"Evaluating TSP solver: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    num_runs = len(TEST_INSTANCES)

    def _aggregator(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return aggregate_tsp_metrics(results, results_dir)

    def _validator(result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        # Get instance info from the result or use a generic check
        return validate_tsp_result(result, {'n': len(result.get('tour', []))})

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=num_runs,
        get_experiment_kwargs=get_experiment_kwargs,
        validate_fn=_validator,
        aggregate_metrics_fn=_aggregator,
    )

    if correct:
        print("Evaluation completed successfully.")
    else:
        print(f"Evaluation failed: {error_msg}")

    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, list) and len(v) > 5:
                    print(f"    {k}: [{v[0]:.4f}, {v[1]:.4f}, ... ({len(v)} items)]")
                else:
                    print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSP Solver Evaluator")
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to TSP solver program",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)

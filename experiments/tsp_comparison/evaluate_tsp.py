"""
TSP Evaluator for ShinkaEvolve and Paper-Based comparison.
"""

import os
import json
import argparse
import numpy as np
import importlib.util
import traceback
from typing import Dict, Any, List, Tuple


# Test instances: coordinates and best-known solutions
TEST_INSTANCES = {
    # Small instances for quick testing
    'random_20': {
        'coords': None,  # Generated
        'bks': None,  # Unknown
        'seed': 42
    },
    'random_50': {
        'coords': None,
        'bks': None,
        'seed': 42
    },
    'random_100': {
        'coords': None,
        'bks': None,
        'seed': 42
    },
}


def generate_instance(n: int, seed: int) -> np.ndarray:
    """Generate random TSP instance."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1000, size=(n, 2))


def compute_tour_length(tour: List[int], coords: np.ndarray) -> float:
    """Compute tour length."""
    n = len(tour)
    length = 0.0
    for i in range(n):
        dx = coords[tour[i], 0] - coords[tour[(i+1) % n], 0]
        dy = coords[tour[i], 1] - coords[tour[(i+1) % n], 1]
        length += round(np.sqrt(dx*dx + dy*dy))
    return length


def validate_tour(tour: List[int], n: int) -> Tuple[bool, str]:
    """Validate that tour is a valid Hamiltonian cycle."""
    if not isinstance(tour, list):
        return False, "Tour must be a list"
    if len(tour) != n:
        return False, f"Tour length {len(tour)} != {n} cities"
    if set(tour) != set(range(n)):
        return False, "Tour must visit each city exactly once"
    return True, "Valid tour"


def load_solver(program_path: str):
    """Dynamically load solver module."""
    spec = importlib.util.spec_from_file_location("solver", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_solver(
    program_path: str,
    instances: Dict[str, Dict],
    time_limit_ms: int = 5000,
    seeds: List[int] = [0, 1, 2]
) -> Dict[str, Any]:
    """
    Evaluate a TSP solver on multiple instances.

    Returns:
        Dict with scores and detailed results
    """
    results = {
        'program_path': program_path,
        'instances': {},
        'overall_score': 0.0,
        'valid': True,
        'error': None
    }

    try:
        solver_module = load_solver(program_path)

        total_score = 0.0
        num_evals = 0

        for inst_name, inst_data in instances.items():
            inst_results = []

            # Generate or use provided coordinates
            if inst_data['coords'] is None:
                n = int(inst_name.split('_')[1])
                coords = generate_instance(n, inst_data['seed'])
            else:
                coords = np.array(inst_data['coords'])

            n = len(coords)

            for seed in seeds:
                try:
                    # Run solver
                    instance = {
                        'coords': coords.tolist(),
                        'time_limit_ms': time_limit_ms
                    }

                    result = solver_module.run_solver(instance)

                    tour = result['tour']
                    length = result['length']

                    # Validate
                    valid, msg = validate_tour(tour, n)

                    if valid:
                        # Verify length
                        computed_length = compute_tour_length(tour, coords)
                        length_valid = bool(abs(computed_length - length) < 1)

                        inst_results.append({
                            'seed': seed,
                            'tour': tour,
                            'length': float(length),
                            'computed_length': float(computed_length),
                            'valid': length_valid,
                            'error': None if length_valid else "Length mismatch"
                        })

                        if length_valid:
                            # Score: inverse of length (higher is better)
                            score = 10000.0 / length
                            total_score += score
                            num_evals += 1
                    else:
                        inst_results.append({
                            'seed': seed,
                            'valid': False,
                            'error': msg
                        })

                except Exception as e:
                    inst_results.append({
                        'seed': seed,
                        'valid': False,
                        'error': str(e)
                    })

            results['instances'][inst_name] = inst_results

        results['overall_score'] = total_score / max(num_evals, 1)

    except Exception as e:
        results['valid'] = False
        results['error'] = traceback.format_exc()

    return results


def main():
    parser = argparse.ArgumentParser(description="TSP Solver Evaluator")
    parser.add_argument('--program_path', type=str, default='tsp_initial.py')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--time_limit', type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Evaluating: {args.program_path}")

    results = evaluate_solver(
        args.program_path,
        TEST_INSTANCES,
        time_limit_ms=args.time_limit
    )

    # Save results
    results_file = os.path.join(args.results_dir, 'metrics.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nOverall Score: {results['overall_score']:.4f}")
    print(f"Results saved to: {results_file}")

    # Print per-instance summary
    print("\nPer-instance results:")
    for inst_name, inst_results in results['instances'].items():
        lengths = [r['length'] for r in inst_results if r.get('valid', False)]
        if lengths:
            print(f"  {inst_name}: avg={np.mean(lengths):.0f}, min={min(lengths):.0f}")
        else:
            print(f"  {inst_name}: FAILED")


if __name__ == "__main__":
    main()

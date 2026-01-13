#!/usr/bin/env python3
"""
TSP Baseline Experiment Runner

This script generates various TSP instances, runs the nearest neighbor + 2-opt
baseline, and reports interesting results and statistics.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from paperbench.tasks.tsp.baseline import solve, _compute_distance_matrix, _tour_length


def generate_random_instance(n: int, seed: int, distribution: str = 'uniform') -> dict:
    """Generate a random TSP instance."""
    rng = np.random.default_rng(seed)

    if distribution == 'uniform':
        coords = rng.uniform(0, 1000, size=(n, 2))
    elif distribution == 'clustered':
        # Generate 3-5 clusters
        n_clusters = rng.integers(3, 6)
        cluster_centers = rng.uniform(100, 900, size=(n_clusters, 2))
        coords = []
        for i in range(n):
            center = cluster_centers[i % n_clusters]
            point = center + rng.normal(0, 50, size=2)
            coords.append(point)
        coords = np.array(coords)
    elif distribution == 'circular':
        # Points on a circle with noise
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radius = 400
        coords = np.column_stack([
            500 + radius * np.cos(angles) + rng.normal(0, 20, n),
            500 + radius * np.sin(angles) + rng.normal(0, 20, n)
        ])
    elif distribution == 'grid':
        # Grid with jitter
        side = int(np.ceil(np.sqrt(n)))
        coords = []
        for i in range(n):
            x = (i % side) * (1000 / side) + rng.uniform(-20, 20)
            y = (i // side) * (1000 / side) + rng.uniform(-20, 20)
            coords.append([x, y])
        coords = np.array(coords)
    else:
        coords = rng.uniform(0, 1000, size=(n, 2))

    return {
        'coords': coords,
        'edge_weight_type': 'EUC_2D',
        'dimension': n,
        'distribution': distribution
    }


def generate_classic_instance(name: str) -> dict:
    """Generate a classic TSP instance."""

    if name == 'square':
        # 4 corners of a square - optimal is 4000
        coords = np.array([
            [0, 0], [1000, 0], [1000, 1000], [0, 1000]
        ], dtype=float)
    elif name == 'pentagon':
        # Regular pentagon
        n = 5
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radius = 500
        coords = np.column_stack([
            500 + radius * np.cos(angles),
            500 + radius * np.sin(angles)
        ])
    elif name == 'star':
        # 10-point star
        n = 10
        coords = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            r = 500 if i % 2 == 0 else 200
            coords.append([500 + r * np.cos(angle), 500 + r * np.sin(angle)])
        coords = np.array(coords)
    elif name == 'line':
        # Points on a line
        n = 20
        coords = np.column_stack([
            np.linspace(0, 1000, n),
            np.full(n, 500.0)
        ])
    else:
        # Default random
        coords = np.random.rand(10, 2) * 1000

    return {
        'coords': coords,
        'edge_weight_type': 'EUC_2D',
        'dimension': len(coords),
        'name': name
    }


def compute_lower_bound(instance: dict) -> float:
    """Compute a simple lower bound using minimum spanning tree."""
    coords = instance['coords']
    n = len(coords)
    dist = _compute_distance_matrix(coords, instance['edge_weight_type'])

    # Prim's algorithm for MST
    in_mst = np.zeros(n, dtype=bool)
    min_edge = np.full(n, np.inf)
    min_edge[0] = 0
    mst_weight = 0.0

    for _ in range(n):
        # Find minimum edge to non-MST vertex
        u = -1
        min_val = np.inf
        for v in range(n):
            if not in_mst[v] and min_edge[v] < min_val:
                min_val = min_edge[v]
                u = v

        in_mst[u] = True
        mst_weight += min_val

        # Update distances
        for v in range(n):
            if not in_mst[v] and dist[u, v] < min_edge[v]:
                min_edge[v] = dist[u, v]

    return mst_weight


def print_tour_ascii(coords: np.ndarray, tour: list, width: int = 60, height: int = 20):
    """Print an ASCII visualization of the tour."""
    # Normalize coordinates
    min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
    min_y, max_y = coords[:, 1].min(), coords[:, 1].max()

    # Add padding
    range_x = max_x - min_x if max_x > min_x else 1
    range_y = max_y - min_y if max_y > min_y else 1

    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Plot cities
    for i, (x, y) in enumerate(coords):
        col = int((x - min_x) / range_x * (width - 1))
        row = int((y - min_y) / range_y * (height - 1))
        row = height - 1 - row  # Flip y-axis
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))

        if i < 10:
            grid[row][col] = str(i)
        else:
            grid[row][col] = '*'

    # Print
    print("+" + "-" * width + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * width + "+")


def run_scaling_experiment():
    """Test how the algorithm scales with problem size."""
    print("\n" + "=" * 70)
    print("SCALING EXPERIMENT: Performance vs Problem Size")
    print("=" * 70)

    sizes = [10, 20, 50, 100, 200, 500]
    results = []

    print(f"\n{'Size':>6} | {'NN Length':>12} | {'2-opt Length':>12} | {'Improvement':>10} | {'Time (ms)':>10} | {'Iterations':>10}")
    print("-" * 80)

    for n in sizes:
        instance = generate_random_instance(n, seed=42, distribution='uniform')
        dist = _compute_distance_matrix(instance['coords'], instance['edge_weight_type'])

        # Get NN-only result
        from paperbench.tasks.tsp.baseline import _nearest_neighbor, _tour_length
        rng = np.random.default_rng(42)
        nn_tour = _nearest_neighbor(dist, n, rng)
        nn_length = _tour_length(nn_tour, dist)

        # Run full solver
        result = solve(instance, seed=42, time_limit_ms=5000)

        improvement = (nn_length - result['objective']) / nn_length * 100

        results.append({
            'size': n,
            'nn_length': nn_length,
            'opt_length': result['objective'],
            'improvement': improvement,
            'time_ms': result['meta']['elapsed_ms'],
            'iterations': result['meta']['2opt_iterations']
        })

        print(f"{n:>6} | {nn_length:>12.0f} | {result['objective']:>12.0f} | {improvement:>9.1f}% | {result['meta']['elapsed_ms']:>10.1f} | {result['meta']['2opt_iterations']:>10}")

    return results


def run_distribution_experiment():
    """Test performance on different spatial distributions."""
    print("\n" + "=" * 70)
    print("DISTRIBUTION EXPERIMENT: Performance on Different City Layouts")
    print("=" * 70)

    distributions = ['uniform', 'clustered', 'circular', 'grid']
    n = 50
    seeds = [0, 1, 2, 3, 4]

    print(f"\n{'Distribution':>12} | {'Avg Length':>12} | {'Std Dev':>10} | {'Avg Time (ms)':>12} | {'Avg Iterations':>14}")
    print("-" * 75)

    for dist in distributions:
        lengths = []
        times = []
        iterations = []

        for seed in seeds:
            instance = generate_random_instance(n, seed=seed, distribution=dist)
            result = solve(instance, seed=seed, time_limit_ms=3000)
            lengths.append(result['objective'])
            times.append(result['meta']['elapsed_ms'])
            iterations.append(result['meta']['2opt_iterations'])

        print(f"{dist:>12} | {np.mean(lengths):>12.1f} | {np.std(lengths):>10.1f} | {np.mean(times):>12.1f} | {np.mean(iterations):>14.1f}")


def run_seed_sensitivity_experiment():
    """Test how different random seeds affect solution quality."""
    print("\n" + "=" * 70)
    print("SEED SENSITIVITY: Solution Variance Across Random Seeds")
    print("=" * 70)

    n = 100
    instance = generate_random_instance(n, seed=0, distribution='uniform')

    seeds = range(20)
    results = []

    for seed in seeds:
        result = solve(instance, seed=seed, time_limit_ms=2000)
        results.append(result['objective'])

    print(f"\nInstance: {n} cities, uniform distribution")
    print(f"Number of seeds tested: {len(seeds)}")
    print(f"\nStatistics:")
    print(f"  Best solution:   {min(results):.0f}")
    print(f"  Worst solution:  {max(results):.0f}")
    print(f"  Mean:            {np.mean(results):.1f}")
    print(f"  Std deviation:   {np.std(results):.1f}")
    print(f"  Gap (best-worst): {(max(results) - min(results)) / min(results) * 100:.2f}%")

    # Show distribution
    print(f"\nSolution length distribution:")
    bins = 5
    hist, bin_edges = np.histogram(results, bins=bins)
    max_count = max(hist)
    for i in range(bins):
        bar_len = int(hist[i] / max_count * 30)
        print(f"  [{bin_edges[i]:>7.0f} - {bin_edges[i+1]:>7.0f}]: {'#' * bar_len} ({hist[i]})")

    return results


def run_classic_instances():
    """Run on classic geometric instances."""
    print("\n" + "=" * 70)
    print("CLASSIC INSTANCES: Known Geometric Configurations")
    print("=" * 70)

    instances = ['square', 'pentagon', 'star', 'line']

    for name in instances:
        instance = generate_classic_instance(name)
        n = instance['dimension']

        print(f"\n--- {name.upper()} ({n} cities) ---")

        # Compute lower bound
        lb = compute_lower_bound(instance)

        result = solve(instance, seed=0, time_limit_ms=1000)

        print(f"Tour length: {result['objective']:.0f}")
        print(f"MST lower bound: {lb:.0f}")
        print(f"Gap from LB: {(result['objective'] - lb) / lb * 100:.1f}%")
        print(f"Time: {result['meta']['elapsed_ms']:.1f} ms")
        print(f"Tour: {result['solution']}")

        # Visualize small instances
        if n <= 20:
            print("\nVisualization:")
            print_tour_ascii(instance['coords'], result['solution'])


def run_time_limit_experiment():
    """Test impact of time limit on solution quality."""
    print("\n" + "=" * 70)
    print("TIME LIMIT EXPERIMENT: Solution Quality vs Computation Time")
    print("=" * 70)

    n = 200
    instance = generate_random_instance(n, seed=42, distribution='uniform')

    time_limits = [10, 50, 100, 500, 1000, 2000, 5000, 10000]

    print(f"\nInstance: {n} cities, uniform distribution")
    print(f"\n{'Time Limit (ms)':>15} | {'Solution':>12} | {'Actual Time':>12} | {'Iterations':>10}")
    print("-" * 60)

    first_result = None
    for tl in time_limits:
        result = solve(instance, seed=42, time_limit_ms=tl)

        if first_result is None:
            first_result = result['objective']

        improvement = (first_result - result['objective']) / first_result * 100

        print(f"{tl:>15} | {result['objective']:>12.0f} | {result['meta']['elapsed_ms']:>11.1f}ms | {result['meta']['2opt_iterations']:>10}")

    print(f"\nImprovement from 10ms to 10000ms: {improvement:.1f}%")


def run_2opt_analysis():
    """Detailed analysis of 2-opt improvement process."""
    print("\n" + "=" * 70)
    print("2-OPT ANALYSIS: Improvement Progress")
    print("=" * 70)

    n = 50
    instance = generate_random_instance(n, seed=123, distribution='uniform')
    coords = instance['coords']

    # Run with detailed tracking
    from paperbench.tasks.tsp.baseline import _compute_distance_matrix, _nearest_neighbor

    rng = np.random.default_rng(123)
    dist = _compute_distance_matrix(coords, 'EUC_2D')

    tour = _nearest_neighbor(dist, n, rng)
    initial_length = _tour_length(tour, dist)

    print(f"\nInstance: {n} cities")
    print(f"Initial tour length (NN): {initial_length:.0f}")

    # Track 2-opt progress
    improvements = []
    current_length = initial_length

    improved = True
    iteration = 0
    while improved:
        improved = False
        iteration += 1
        iter_improvements = 0

        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == i + 1:
                    continue

                # Calculate delta
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])

                if delta < -1e-9:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    current_length += delta
                    improvements.append({
                        'iteration': iteration,
                        'length': current_length,
                        'delta': delta
                    })
                    iter_improvements += 1
                    improved = True

        if improved:
            print(f"  Iteration {iteration}: {iter_improvements} swaps, length = {current_length:.0f}")

    print(f"\nFinal tour length: {current_length:.0f}")
    print(f"Total improvement: {initial_length - current_length:.0f} ({(initial_length - current_length) / initial_length * 100:.1f}%)")
    print(f"Total 2-opt iterations: {iteration - 1}")
    print(f"Total swaps performed: {len(improvements)}")


def main():
    print("=" * 70)
    print("TSP BASELINE EXPERIMENTS")
    print("Nearest Neighbor Construction + 2-opt Local Search")
    print("=" * 70)

    # Run all experiments
    run_classic_instances()
    run_scaling_experiment()
    run_distribution_experiment()
    run_seed_sensitivity_experiment()
    run_time_limit_experiment()
    run_2opt_analysis()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

# EVOLVE-BLOCK-START
"""
TSP Solver: Nearest Neighbor + 2-opt Local Search

This solver constructs a tour using nearest neighbor heuristic,
then improves it using 2-opt edge swaps.
"""

import numpy as np
import time


def solve_tsp(coords: np.ndarray, time_limit_ms: int = 5000) -> tuple:
    """
    Solve TSP instance.

    Args:
        coords: np.ndarray of shape (n, 2) with city coordinates
        time_limit_ms: Wall-clock time limit in milliseconds

    Returns:
        Tuple of (tour, length) where tour is list of city indices
    """
    n = len(coords)

    # Compute distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            dist[i, j] = dist[j, i] = round(d)

    start_time = time.perf_counter()
    deadline = start_time + time_limit_ms / 1000.0

    # Phase 1: Nearest neighbor construction
    visited = np.zeros(n, dtype=bool)
    tour = np.zeros(n, dtype=int)

    current = 0  # Start from city 0
    tour[0] = current
    visited[current] = True

    for step in range(1, n):
        best_next = -1
        best_dist = np.inf
        for j in range(n):
            if not visited[j] and dist[current, j] < best_dist:
                best_dist = dist[current, j]
                best_next = j
        tour[step] = best_next
        visited[best_next] = True
        current = best_next

    # Phase 2: 2-opt improvement
    improved = True
    while improved and time.perf_counter() < deadline:
        improved = False
        for i in range(n - 1):
            if time.perf_counter() >= deadline:
                break
            for j in range(i + 2, n):
                if j == i + 1:
                    continue

                # Calculate delta
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])

                if delta < -1e-9:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True

    # Calculate tour length
    length = sum(dist[tour[i], tour[(i + 1) % n]] for i in range(n))

    return tour.tolist(), length


# EVOLVE-BLOCK-END


def run_solver(instance: dict) -> dict:
    """
    Entry point for evaluation.

    Args:
        instance: dict with 'coords' and 'time_limit_ms'

    Returns:
        dict with 'tour', 'length', 'valid'
    """
    coords = np.array(instance['coords'])
    time_limit = instance.get('time_limit_ms', 5000)

    tour, length = solve_tsp(coords, time_limit)

    # Validate tour
    n = len(coords)
    valid = (len(tour) == n and set(tour) == set(range(n)))

    return {
        'tour': tour,
        'length': length,
        'valid': valid
    }

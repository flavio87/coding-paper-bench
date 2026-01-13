# paperbench/tasks/tsp/baseline.py

import numpy as np
from typing import Any
import time

def solve(
    instance: dict,
    seed: int,
    time_limit_ms: int,
    **kwargs
) -> dict:
    """
    TSP baseline: Nearest neighbor construction + 2-opt local search.

    Args:
        instance: dict with keys:
            - 'coords': np.ndarray of shape (n, 2)
            - 'edge_weight_type': str ('EUC_2D', 'CEIL_2D', etc.)
            - 'dimension': int
        seed: Random seed (used for tie-breaking)
        time_limit_ms: Wall-clock time limit

    Returns:
        dict with 'solution' (tour), 'objective' (length), 'meta'
    """
    rng = np.random.default_rng(seed)
    coords = instance['coords']
    n = instance['dimension']

    # Precompute distance matrix
    dist = _compute_distance_matrix(coords, instance['edge_weight_type'])

    start_time = time.perf_counter()
    deadline = start_time + time_limit_ms / 1000.0

    # Phase 1: Nearest neighbor construction
    tour = _nearest_neighbor(dist, n, rng)
    best_length = _tour_length(tour, dist)

    # Phase 2: 2-opt improvement until timeout
    improved = True
    iterations = 0
    while improved and time.perf_counter() < deadline:
        improved = False
        iterations += 1
        for i in range(n - 1):
            if time.perf_counter() >= deadline:
                break
            for j in range(i + 2, n):
                if j == i + 1:
                    continue
                # 2-opt swap: reverse segment [i+1, j]
                delta = _two_opt_delta(tour, i, j, dist)
                if delta < -1e-9:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    best_length += delta
                    improved = True

    return {
        'solution': tour.tolist(),
        'objective': best_length,
        'meta': {
            'elapsed_ms': (time.perf_counter() - start_time) * 1000,
            'method': 'nearest_neighbor_2opt',
            '2opt_iterations': iterations
        }
    }


def _compute_distance_matrix(coords: np.ndarray, edge_type: str) -> np.ndarray:
    """Compute pairwise distances according to TSPLIB edge weight type."""
    n = len(coords)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]

            if edge_type == 'EUC_2D':
                d = np.sqrt(dx*dx + dy*dy)
                dist[i, j] = dist[j, i] = round(d)
            elif edge_type == 'CEIL_2D':
                d = np.sqrt(dx*dx + dy*dy)
                dist[i, j] = dist[j, i] = np.ceil(d)
            elif edge_type == 'ATT':
                r = np.sqrt((dx*dx + dy*dy) / 10.0)
                t = round(r)
                dist[i, j] = dist[j, i] = t + 1 if t < r else t
            else:
                # Default to Euclidean
                dist[i, j] = dist[j, i] = round(np.sqrt(dx*dx + dy*dy))

    return dist


def _nearest_neighbor(dist: np.ndarray, n: int, rng) -> np.ndarray:
    """Construct tour using nearest neighbor heuristic."""
    visited = np.zeros(n, dtype=bool)
    tour = np.zeros(n, dtype=int)

    # Start from random city
    current = rng.integers(0, n)
    tour[0] = current
    visited[current] = True

    for step in range(1, n):
        # Find nearest unvisited
        best_next = -1
        best_dist = np.inf
        for j in range(n):
            if not visited[j] and dist[current, j] < best_dist:
                best_dist = dist[current, j]
                best_next = j
        tour[step] = best_next
        visited[best_next] = True
        current = best_next

    return tour


def _tour_length(tour: np.ndarray, dist: np.ndarray) -> float:
    """Compute total tour length."""
    n = len(tour)
    length = 0.0
    for i in range(n):
        length += dist[tour[i], tour[(i + 1) % n]]
    return length


def _two_opt_delta(tour: np.ndarray, i: int, j: int, dist: np.ndarray) -> float:
    """Compute change in tour length from 2-opt swap."""
    n = len(tour)
    a, b = tour[i], tour[i + 1]
    c, d = tour[j], tour[(j + 1) % n]

    # Remove edges (a,b) and (c,d), add edges (a,c) and (b,d)
    return (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])

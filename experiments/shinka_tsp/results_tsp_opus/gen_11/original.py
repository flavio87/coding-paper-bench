# EVOLVE-BLOCK-START
"""
TSP Solver: Multi-start Nearest Neighbor + 2-opt + Or-opt Local Search

This solver constructs tours using nearest neighbor from multiple starts,
then improves using 2-opt and Or-opt moves.
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

    if n <= 1:
        return list(range(n)), 0.0

    # Vectorized distance matrix computation
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.round(np.sqrt(np.sum(diff ** 2, axis=2)))

    start_time = time.perf_counter()
    deadline = start_time + time_limit_ms / 1000.0

    def calc_tour_length(tour):
        return sum(dist[tour[i], tour[(i + 1) % n]] for i in range(n))

    def nearest_neighbor(start_city):
        visited = np.zeros(n, dtype=bool)
        tour = np.zeros(n, dtype=int)
        current = start_city
        tour[0] = current
        visited[current] = True

        for step in range(1, n):
            # Vectorized nearest neighbor search
            dists = dist[current].copy()
            dists[visited] = np.inf
            best_next = np.argmin(dists)
            tour[step] = best_next
            visited[best_next] = True
            current = best_next
        return tour

    def two_opt(tour):
        improved = True
        while improved:
            if time.perf_counter() >= deadline:
                break
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue  # Skip if would reverse entire tour

                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[(j + 1) % n]
                    delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])

                    if delta < -0.5:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
        return tour

    def or_opt(tour):
        """Or-opt: relocate segments of 1-3 cities"""
        improved = True
        while improved:
            if time.perf_counter() >= deadline:
                break
            improved = False
            for seg_len in [1, 2, 3]:  # Try segments of length 1, 2, 3
                if improved:
                    break
                for i in range(n):
                    if improved:
                        break
                    if time.perf_counter() >= deadline:
                        break
                    # Segment from i to i+seg_len-1
                    if i + seg_len > n:
                        continue

                    # Current cost of segment edges
                    prev_i = (i - 1) % n
                    next_seg = (i + seg_len) % n

                    # Cost to remove segment
                    old_cost = dist[tour[prev_i], tour[i]] + dist[tour[(i + seg_len - 1) % n], tour[next_seg]]
                    # Cost to connect prev to next (bypassing segment)
                    bypass_cost = dist[tour[prev_i], tour[next_seg]]

                    # Try inserting segment elsewhere
                    for j in range(n):
                        if j >= i - 1 and j <= i + seg_len:
                            continue  # Skip positions that overlap with current

                        next_j = (j + 1) % n
                        # Cost to insert segment between j and j+1
                        insert_old = dist[tour[j], tour[next_j]]
                        insert_new = dist[tour[j], tour[i]] + dist[tour[(i + seg_len - 1) % n], tour[next_j]]

                        delta = bypass_cost + insert_new - old_cost - insert_old

                        if delta < -0.5:
                            # Perform the move
                            segment = tour[i:i+seg_len].copy()
                            new_tour = np.concatenate([
                                tour[:i],
                                tour[i+seg_len:]
                            ])
                            # Find new position for j (adjusted for removal)
                            new_j = j if j < i else j - seg_len
                            new_tour = np.concatenate([
                                new_tour[:new_j+1],
                                segment,
                                new_tour[new_j+1:]
                            ])
                            tour[:] = new_tour
                            improved = True
                            break
        return tour

    # Multi-start nearest neighbor
    best_tour = None
    best_length = np.inf

    # Determine number of starts based on problem size
    num_starts = min(n, max(5, n // 10))
    start_cities = np.linspace(0, n-1, num_starts, dtype=int)

    for start_city in start_cities:
        if time.perf_counter() >= deadline:
            break
        tour = nearest_neighbor(start_city)
        length = calc_tour_length(tour)
        if length < best_length:
            best_length = length
            best_tour = tour.copy()

    # Apply local search to best tour
    tour = best_tour.copy()

    # Iterative improvement
    prev_length = np.inf
    while time.perf_counter() < deadline:
        tour = two_opt(tour)
        if time.perf_counter() >= deadline:
            break
        tour = or_opt(tour)

        current_length = calc_tour_length(tour)
        if current_length >= prev_length - 0.5:
            break
        prev_length = current_length

    # Calculate final tour length
    length = calc_tour_length(tour)

    return tour.tolist(), length


# EVOLVE-BLOCK-END


def run_experiment(coords: np.ndarray, time_limit_ms: int = 5000, optimal_length: float = None) -> dict:
    """
    Entry point for ShinkaEvolve evaluation.

    Args:
        coords: np.ndarray of shape (n, 2) with city coordinates
        time_limit_ms: Wall-clock time limit in milliseconds
        optimal_length: Known optimal tour length (if available)

    Returns:
        dict with tour, length, valid, and score
    """
    tour, length = solve_tsp(coords, time_limit_ms)

    # Validate tour
    n = len(coords)
    valid = (len(tour) == n and set(tour) == set(range(n)))

    # Score: ratio of optimal to found (higher is better, max 1.0)
    if optimal_length and optimal_length > 0:
        score = optimal_length / length if length > 0 else 0.0
    else:
        # If no optimal, use inverse of length (normalized)
        score = 1000.0 / length if length > 0 else 0.0

    return {
        'tour': tour,
        'length': length,
        'valid': valid,
        'score': score
    }
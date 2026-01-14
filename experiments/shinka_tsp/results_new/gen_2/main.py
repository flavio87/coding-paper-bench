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

    if n == 2:
        return [0, 1], round(np.sqrt(np.sum((coords[0] - coords[1]) ** 2))) * 2

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
            # Use vectorized min finding
            dists = dist[current].copy()
            dists[visited] = np.inf
            best_next = np.argmin(dists)
            tour[step] = best_next
            visited[best_next] = True
            current = best_next
        return tour

    def two_opt_pass(tour):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue  # Skip if it would be the same tour

                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])

                if delta < -0.5:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
        return improved

    def or_opt_pass(tour, seg_len):
        """Try relocating segments of length seg_len"""
        improved = False
        for i in range(n):
            if time.perf_counter() >= deadline:
                break
            # Segment from i to i+seg_len-1
            seg_end = (i + seg_len - 1) % n
            prev_i = (i - 1) % n
            next_seg = (seg_end + 1) % n

            # Cost of removing segment
            remove_cost = dist[tour[prev_i], tour[i]] + dist[tour[seg_end], tour[next_seg]]
            reconnect_cost = dist[tour[prev_i], tour[next_seg]]

            # Try inserting segment at different positions
            for j in range(n):
                if j == prev_i or j == i or j == seg_end:
                    continue
                next_j = (j + 1) % n
                if next_j == i:
                    continue

                # Cost of inserting segment after position j
                insert_cost = dist[tour[j], tour[i]] + dist[tour[seg_end], tour[next_j]] - dist[tour[j], tour[next_j]]

                delta = reconnect_cost + insert_cost - remove_cost

                if delta < -0.5:
                    # Perform the move
                    segment = [tour[(i + k) % n] for k in range(seg_len)]
                    new_tour = []
                    k = next_seg
                    while k != i:
                        new_tour.append(tour[k])
                        if tour[k] == tour[j]:
                            new_tour.extend(segment)
                        k = (k + 1) % n
                    tour[:] = new_tour
                    improved = True
                    return improved  # Restart after improvement
        return improved

    # Phase 1: Multi-start nearest neighbor
    num_starts = min(n, max(5, n // 10))
    start_cities = np.linspace(0, n - 1, num_starts, dtype=int)

    best_tour = None
    best_length = np.inf

    for start in start_cities:
        if time.perf_counter() >= deadline:
            break
        tour = nearest_neighbor(start)
        length = calc_tour_length(tour)
        if length < best_length:
            best_length = length
            best_tour = tour.copy()

    tour = best_tour

    # Phase 2: Iterative improvement with 2-opt and Or-opt
    while time.perf_counter() < deadline:
        improved = False

        # 2-opt pass
        if two_opt_pass(tour):
            improved = True

        if time.perf_counter() >= deadline:
            break

        # Or-opt passes (segments of length 1, 2, 3)
        for seg_len in [1, 2, 3]:
            if time.perf_counter() >= deadline:
                break
            if or_opt_pass(tour, seg_len):
                improved = True
                break

        if not improved:
            break

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
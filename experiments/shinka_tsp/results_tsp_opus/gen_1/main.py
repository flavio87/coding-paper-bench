# EVOLVE-BLOCK-START
"""
TSP Solver: Multi-start Nearest Neighbor + 2-opt + Or-opt Local Search

This solver constructs tours using nearest neighbor from multiple starts,
then improves using 2-opt edge swaps and Or-opt segment relocations.
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
        return [0, 1], round(np.sqrt(np.sum((coords[0] - coords[1]) ** 2)))

    # Vectorized distance matrix computation
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.round(np.sqrt(np.sum(diff ** 2, axis=2)))

    start_time = time.perf_counter()
    deadline = start_time + time_limit_ms / 1000.0

    def calc_tour_length(tour):
        return sum(dist[tour[i], tour[(i + 1) % n]] for i in range(n))

    def nearest_neighbor(start_city):
        """Build tour using nearest neighbor starting from given city."""
        visited = np.zeros(n, dtype=bool)
        tour = np.zeros(n, dtype=int)

        current = start_city
        tour[0] = current
        visited[current] = True

        for step in range(1, n):
            # Find nearest unvisited city
            min_dist = np.inf
            best_next = -1
            for j in range(n):
                if not visited[j] and dist[current, j] < min_dist:
                    min_dist = dist[current, j]
                    best_next = j
            tour[step] = best_next
            visited[best_next] = True
            current = best_next

        return tour

    def two_opt(tour, deadline):
        """Apply 2-opt improvement until no improvement or deadline."""
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            for i in range(n - 1):
                if time.perf_counter() >= deadline:
                    break
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue  # Skip if it would just reverse entire tour

                    # Calculate delta for reversing segment [i+1, j]
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[(j + 1) % n]
                    delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])

                    if delta < -0.5:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
        return tour

    def or_opt(tour, deadline):
        """Apply Or-opt: relocate segments of 1, 2, or 3 consecutive cities."""
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            for seg_len in [1, 2, 3]:  # Try segments of length 1, 2, 3
                if improved or time.perf_counter() >= deadline:
                    break
                for i in range(n):
                    if time.perf_counter() >= deadline:
                        break
                    if i + seg_len > n:
                        continue

                    # Segment to move: tour[i:i+seg_len]
                    prev_i = (i - 1) % n
                    next_seg = (i + seg_len) % n

                    # Cost of removing segment
                    remove_cost = (dist[tour[prev_i], tour[i]] +
                                   dist[tour[(i + seg_len - 1) % n], tour[next_seg]])
                    reconnect_cost = dist[tour[prev_i], tour[next_seg]]

                    for j in range(n):
                        # Skip positions that overlap with current segment position
                        if j >= i - 1 and j <= i + seg_len:
                            continue

                        next_j = (j + 1) % n

                        # Cost of inserting segment between j and j+1
                        insert_cost = (dist[tour[j], tour[i]] +
                                      dist[tour[(i + seg_len - 1) % n], tour[next_j]])
                        current_edge = dist[tour[j], tour[next_j]]

                        delta = (reconnect_cost + insert_cost) - (remove_cost + current_edge)

                        if delta < -0.5:
                            # Perform the move
                            segment = tour[i:i+seg_len].copy()
                            new_tour = np.concatenate([
                                tour[:i],
                                tour[i+seg_len:]
                            ])
                            # Find new position of j in reduced tour
                            if j < i:
                                insert_pos = j + 1
                            else:
                                insert_pos = j + 1 - seg_len
                            tour[:] = np.concatenate([
                                new_tour[:insert_pos],
                                segment,
                                new_tour[insert_pos:]
                            ])
                            improved = True
                            break
                    if improved:
                        break
        return tour

    # Phase 1: Multi-start nearest neighbor construction
    # Use a subset of starting cities based on problem size
    num_starts = min(n, max(5, n // 10))
    if n <= 20:
        start_cities = list(range(n))
    else:
        # Choose spread-out starting cities
        start_cities = list(range(0, n, n // num_starts))[:num_starts]

    best_tour = None
    best_length = np.inf

    construction_deadline = start_time + (time_limit_ms / 1000.0) * 0.1  # 10% for construction

    for start in start_cities:
        if time.perf_counter() >= construction_deadline:
            break
        tour = nearest_neighbor(start)
        length = calc_tour_length(tour)
        if length < best_length:
            best_length = length
            best_tour = tour.copy()

    if best_tour is None:
        best_tour = nearest_neighbor(0)

    # Phase 2: Local search improvement
    tour = best_tour.copy()

    # Alternate between 2-opt and Or-opt
    iteration = 0
    last_length = calc_tour_length(tour)

    while time.perf_counter() < deadline:
        iteration += 1

        # 2-opt pass
        mini_deadline = min(deadline, time.perf_counter() + 0.1)
        tour = two_opt(tour, mini_deadline)

        if time.perf_counter() >= deadline:
            break

        # Or-opt pass
        mini_deadline = min(deadline, time.perf_counter() + 0.1)
        tour = or_opt(tour, mini_deadline)

        current_length = calc_tour_length(tour)
        if current_length >= last_length - 0.5:
            break  # No significant improvement
        last_length = current_length

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
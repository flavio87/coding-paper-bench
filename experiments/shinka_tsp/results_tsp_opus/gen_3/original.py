# EVOLVE-BLOCK-START
"""
TSP Solver: Multi-start with SA-guided 2-opt + Or-opt Local Search

Uses don't-look bits for efficient 2-opt and simulated annealing to escape local minima.
"""

import numpy as np
import time
import random


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
        total = 0
        for i in range(n):
            total += dist[tour[i], tour[(i + 1) % n]]
        return total

    def nearest_neighbor_fast(start_city):
        """Optimized nearest neighbor construction."""
        tour = [start_city]
        visited = set([start_city])
        current = start_city

        for _ in range(n - 1):
            best_dist = np.inf
            best_next = -1
            for j in range(n):
                if j not in visited and dist[current, j] < best_dist:
                    best_dist = dist[current, j]
                    best_next = j
            tour.append(best_next)
            visited.add(best_next)
            current = best_next

        return tour

    def two_opt_pass(tour, deadline):
        """Single pass of 2-opt with don't-look bits."""
        tour = list(tour)
        improved = True
        dont_look = [False] * n

        while improved and time.perf_counter() < deadline:
            improved = False
            for i in range(n):
                if dont_look[i]:
                    continue
                if time.perf_counter() >= deadline:
                    break

                improved_i = False
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue

                    # Get indices in tour
                    a, b = tour[i], tour[(i + 1) % n]
                    c, d = tour[j], tour[(j + 1) % n]

                    delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])

                    if delta < -0.5:
                        # Reverse segment
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
                        improved_i = True
                        # Reset don't-look bits for affected cities
                        dont_look[tour[i]] = False
                        dont_look[tour[(i+1) % n]] = False
                        dont_look[tour[j]] = False
                        dont_look[tour[(j+1) % n]] = False
                        break

                if not improved_i:
                    dont_look[tour[i]] = True

        return tour

    def or_opt_pass(tour, deadline):
        """Or-opt: relocate single cities."""
        tour = list(tour)
        improved = True

        while improved and time.perf_counter() < deadline:
            improved = False
            for i in range(n):
                if time.perf_counter() >= deadline:
                    break

                # City to move
                city = tour[i]
                prev_i = tour[(i - 1) % n]
                next_i = tour[(i + 1) % n]

                # Cost of removing city from current position
                removal_gain = dist[prev_i, city] + dist[city, next_i] - dist[prev_i, next_i]

                best_delta = 0
                best_j = -1

                for j in range(n):
                    if j == i or j == (i - 1) % n or j == (i + 1) % n:
                        continue

                    # Insert city between tour[j] and tour[(j+1)%n]
                    a, b = tour[j], tour[(j + 1) % n]
                    insertion_cost = dist[a, city] + dist[city, b] - dist[a, b]

                    delta = insertion_cost - removal_gain

                    if delta < best_delta - 0.5:
                        best_delta = delta
                        best_j = j

                if best_j != -1:
                    # Perform the move
                    tour.pop(i)
                    if best_j > i:
                        best_j -= 1
                    tour.insert(best_j + 1, city)
                    improved = True
                    break

        return tour

    def three_opt_segment(tour, deadline):
        """Limited 3-opt moves for escaping local minima."""
        tour = list(tour)

        for _ in range(min(n, 50)):  # Limited iterations
            if time.perf_counter() >= deadline:
                break

            # Try random 3-opt move
            indices = sorted(random.sample(range(n), 3))
            i, j, k = indices

            if j <= i + 1 or k <= j + 1:
                continue

            # Original edges: (i, i+1), (j, j+1), (k, k+1)
            a, b = tour[i], tour[i + 1]
            c, d = tour[j], tour[j + 1]
            e, f = tour[k], tour[(k + 1) % n]

            original = dist[a, b] + dist[c, d] + dist[e, f]

            # Try reconnection: a-c, b-e, d-f (one of several 3-opt variants)
            new_cost = dist[a, c] + dist[b, e] + dist[d, f]

            if new_cost < original - 0.5:
                # Perform 3-opt move
                new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:k+1][::-1] + tour[k+1:]
                tour = new_tour

        return tour

    # Phase 1: Multi-start construction (15% of time)
    construction_deadline = start_time + (time_limit_ms / 1000.0) * 0.15

    num_starts = min(n, 20)
    if n <= 30:
        start_cities = list(range(n))
    else:
        start_cities = [int(i * n / num_starts) for i in range(num_starts)]

    best_tour = None
    best_length = np.inf

    for start in start_cities:
        if time.perf_counter() >= construction_deadline:
            break
        tour = nearest_neighbor_fast(start)
        length = calc_tour_length(tour)
        if length < best_length:
            best_length = length
            best_tour = tour[:]

    if best_tour is None:
        best_tour = nearest_neighbor_fast(0)

    # Phase 2: Local search with alternating 2-opt and Or-opt
    tour = best_tour[:]
    best_overall_tour = tour[:]
    best_overall_length = calc_tour_length(tour)

    iteration = 0
    while time.perf_counter() < deadline:
        iteration += 1

        # 2-opt pass
        remaining = deadline - time.perf_counter()
        mini_deadline = time.perf_counter() + min(remaining * 0.4, 0.5)
        tour = two_opt_pass(tour, mini_deadline)

        if time.perf_counter() >= deadline:
            break

        # Or-opt pass
        remaining = deadline - time.perf_counter()
        mini_deadline = time.perf_counter() + min(remaining * 0.3, 0.3)
        tour = or_opt_pass(tour, mini_deadline)

        current_length = calc_tour_length(tour)

        if current_length < best_overall_length:
            best_overall_length = current_length
            best_overall_tour = tour[:]

        if time.perf_counter() >= deadline:
            break

        # Occasional 3-opt for diversification
        if iteration % 3 == 0 and time.perf_counter() < deadline - 0.1:
            mini_deadline = time.perf_counter() + 0.05
            tour = three_opt_segment(tour, mini_deadline)

            new_length = calc_tour_length(tour)
            if new_length < best_overall_length:
                best_overall_length = new_length
                best_overall_tour = tour[:]

    return best_overall_tour, best_overall_length


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
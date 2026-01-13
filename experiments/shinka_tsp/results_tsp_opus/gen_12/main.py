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

    # Precompute k-nearest neighbors for faster 2-opt
    k_neighbors = min(25, n - 1)
    neighbor_list = np.argsort(dist, axis=1)[:, 1:k_neighbors+1]

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
        """2-opt with don't-look bits and neighbor list acceleration."""
        tour = np.array(tour)
        dont_look = np.zeros(n, dtype=bool)
        improved = True

        # Build position lookup for fast neighbor-based 2-opt
        pos = np.zeros(n, dtype=int)
        for idx in range(n):
            pos[tour[idx]] = idx

        while improved:
            if time.perf_counter() >= deadline:
                break
            improved = False
            for i in range(n):
                if dont_look[tour[i]]:
                    continue
                if time.perf_counter() >= deadline:
                    break

                improved_i = False
                a = tour[i]
                b = tour[(i + 1) % n]

                # First check k-nearest neighbors of city a
                for c in neighbor_list[a]:
                    j = pos[c]
                    if j == i or j == (i + 1) % n or (j == i - 1 and i > 0):
                        continue

                    # Ensure proper ordering for 2-opt
                    if j < i:
                        ii, jj = j, i
                    else:
                        ii, jj = i, j

                    if jj == n - 1 and ii == 0:
                        continue

                    aa, bb = tour[ii], tour[(ii + 1) % n]
                    cc, dd = tour[jj], tour[(jj + 1) % n]
                    delta = (dist[aa, cc] + dist[bb, dd]) - (dist[aa, bb] + dist[cc, dd])

                    if delta < -0.5:
                        tour[ii+1:jj+1] = tour[ii+1:jj+1][::-1]
                        # Update positions
                        for k in range(ii+1, jj+1):
                            pos[tour[k]] = k
                        improved = True
                        improved_i = True
                        dont_look[aa] = False
                        dont_look[bb] = False
                        dont_look[cc] = False
                        dont_look[dd] = False
                        break

                # If no improvement from neighbors, try some random positions
                if not improved_i:
                    step = max(1, n // 50)
                    for j in range(i + 2, n, step):
                        if j == n - 1 and i == 0:
                            continue

                        aa, bb = tour[i], tour[(i + 1) % n]
                        cc, dd = tour[j], tour[(j + 1) % n]
                        delta = (dist[aa, cc] + dist[bb, dd]) - (dist[aa, bb] + dist[cc, dd])

                        if delta < -0.5:
                            tour[i+1:j+1] = tour[i+1:j+1][::-1]
                            for k in range(i+1, j+1):
                                pos[tour[k]] = k
                            improved = True
                            improved_i = True
                            dont_look[aa] = False
                            dont_look[bb] = False
                            dont_look[cc] = False
                            dont_look[dd] = False
                            break

                if not improved_i:
                    dont_look[tour[i]] = True
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

    def farthest_insertion():
        """Farthest insertion heuristic - often produces good initial tours."""
        if n < 3:
            return np.arange(n)

        # Start with two farthest cities
        max_dist = 0
        c1, c2 = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] > max_dist:
                    max_dist = dist[i, j]
                    c1, c2 = i, j

        tour = [c1, c2]
        in_tour = np.zeros(n, dtype=bool)
        in_tour[c1] = True
        in_tour[c2] = True

        while len(tour) < n:
            if time.perf_counter() >= deadline:
                break

            # Find farthest city from tour
            max_min_dist = -1
            farthest = -1
            for c in range(n):
                if in_tour[c]:
                    continue
                min_dist = min(dist[c, tour[i]] for i in range(len(tour)))
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    farthest = c

            if farthest == -1:
                break

            # Find best position to insert
            best_increase = np.inf
            best_pos = 0
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                increase = dist[tour[i], farthest] + dist[farthest, tour[j]] - dist[tour[i], tour[j]]
                if increase < best_increase:
                    best_increase = increase
                    best_pos = i + 1

            tour.insert(best_pos, farthest)
            in_tour[farthest] = True

        return np.array(tour)

    # Multi-start nearest neighbor + farthest insertion
    best_tour = None
    best_length = np.inf

    # Try farthest insertion first
    if time.perf_counter() < deadline:
        tour = farthest_insertion()
        if len(tour) == n:
            length = calc_tour_length(tour)
            if length < best_length:
                best_length = length
                best_tour = tour.copy()

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

    def double_bridge(tour):
        """Double-bridge move for escaping local minima (4-opt)."""
        tour = np.array(tour)
        if n < 8:
            return tour

        # Select 4 random cut points ensuring proper spacing
        segment_size = n // 4
        p1 = np.random.randint(1, segment_size + 1)
        p2 = np.random.randint(p1 + 2, p1 + segment_size + 2)
        p3 = np.random.randint(p2 + 2, p2 + segment_size + 2)

        # Ensure p3 doesn't exceed bounds
        p3 = min(p3, n - 2)
        if p2 >= p3:
            p2 = p3 - 2
        if p1 >= p2:
            p1 = p2 - 2
        if p1 < 1:
            p1 = 1

        # Double-bridge reconnection: A-B-C-D -> A-C-B-D
        new_tour = np.concatenate([
            tour[:p1],
            tour[p2:p3],
            tour[p1:p2],
            tour[p3:]
        ])
        return new_tour

    # Apply local search to best tour
    tour = best_tour.copy()
    best_overall_tour = tour.copy()
    best_overall_length = calc_tour_length(tour)

    # Iterated Local Search with double-bridge perturbation
    no_improve_count = 0
    max_no_improve = 8  # More attempts before giving up

    while time.perf_counter() < deadline:
        # Local search phase
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

        # Update best solution
        current_length = calc_tour_length(tour)
        if current_length < best_overall_length:
            best_overall_length = current_length
            best_overall_tour = tour.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Check termination
        if time.perf_counter() >= deadline or no_improve_count >= max_no_improve:
            break

        # Perturbation: double-bridge move
        tour = double_bridge(best_overall_tour)

    # Calculate final tour length
    length = best_overall_length
    tour = best_overall_tour

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
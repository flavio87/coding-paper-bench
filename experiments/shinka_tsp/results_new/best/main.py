# EVOLVE-BLOCK-START
"""
TSP Solver: Greedy Insertion + Don't-Look 2-opt + Or-opt + Simulated Annealing

Combines efficient construction, fast local search with don't-look bits,
segment moves, and simulated annealing for diversification.
"""

import numpy as np
import time


def solve_tsp(coords: np.ndarray, time_limit_ms: int = 5000) -> tuple:
    n = len(coords)

    if n <= 1:
        return list(range(n)), 0.0

    if n == 2:
        return [0, 1], round(np.sqrt(np.sum((coords[0] - coords[1]) ** 2))) * 2

    # Compute distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.round(np.sqrt(np.sum(diff ** 2, axis=2)))

    start_time = time.perf_counter()
    deadline = start_time + time_limit_ms / 1000.0

    def calc_tour_length(tour):
        return sum(dist[tour[i], tour[(i + 1) % n]] for i in range(n))

    def greedy_insertion(start_city):
        """Build tour by inserting cities where they cause minimum increase."""
        in_tour = np.zeros(n, dtype=bool)
        
        # Start with triangle of 3 nearest cities to start
        dists_from_start = dist[start_city].copy()
        dists_from_start[start_city] = np.inf
        c1 = np.argmin(dists_from_start)
        
        dists_from_start[c1] = np.inf
        c2 = np.argmin(dists_from_start)
        
        tour = [start_city, c1, c2]
        in_tour[start_city] = in_tour[c1] = in_tour[c2] = True
        
        while len(tour) < n:
            best_city = -1
            best_pos = -1
            best_increase = np.inf
            
            # Find city not in tour with minimum insertion cost
            for city in range(n):
                if in_tour[city]:
                    continue
                
                # Find best position to insert this city
                for pos in range(len(tour)):
                    prev = tour[pos]
                    next_city = tour[(pos + 1) % len(tour)]
                    increase = dist[prev, city] + dist[city, next_city] - dist[prev, next_city]
                    
                    if increase < best_increase:
                        best_increase = increase
                        best_city = city
                        best_pos = pos + 1
            
            tour.insert(best_pos, best_city)
            in_tour[best_city] = True
        
        return np.array(tour, dtype=int)

    def nearest_neighbor(start_city):
        """Fallback: nearest neighbor construction."""
        visited = np.zeros(n, dtype=bool)
        tour = np.zeros(n, dtype=int)
        current = start_city
        tour[0] = current
        visited[current] = True

        for step in range(1, n):
            dists = dist[current].copy()
            dists[visited] = np.inf
            best_next = np.argmin(dists)
            tour[step] = best_next
            visited[best_next] = True
            current = best_next
        return tour

    def two_opt_dl(tour):
        """2-opt with don't-look bits."""
        tour = tour.copy()
        dont_look = np.zeros(n, dtype=bool)
        improved = True
        
        while improved and time.perf_counter() < deadline:
            improved = False
            for i in range(n):
                if dont_look[tour[i]]:
                    continue
                if time.perf_counter() >= deadline:
                    break
                    
                local_improved = False
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue
                    
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[(j + 1) % n]
                    delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])

                    if delta < -0.5:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        dont_look[a] = dont_look[b] = dont_look[c] = dont_look[d] = False
                        improved = True
                        local_improved = True
                        break
                
                if not local_improved:
                    dont_look[tour[i]] = True
        
        return tour

    def or_opt(tour):
        """Or-opt: relocate segments of 1, 2, or 3 cities."""
        tour = list(tour)
        improved = True
        
        while improved and time.perf_counter() < deadline:
            improved = False
            
            for seg_len in [1, 2, 3]:
                if improved or time.perf_counter() >= deadline:
                    break
                    
                for i in range(n):
                    if time.perf_counter() >= deadline:
                        break
                    
                    # Segment from i to i+seg_len-1
                    if i + seg_len > n:
                        continue
                    
                    prev_i = (i - 1) % n
                    next_seg = (i + seg_len) % n
                    
                    seg_start = tour[i]
                    seg_end = tour[(i + seg_len - 1) % n]
                    city_prev = tour[prev_i]
                    city_next = tour[next_seg]
                    
                    # Cost of removing segment
                    removal_gain = (dist[city_prev, seg_start] + dist[seg_end, city_next] 
                                   - dist[city_prev, city_next])
                    
                    best_delta = 0
                    best_j = -1
                    
                    for j in range(n):
                        # Skip positions that overlap with segment
                        if j >= i - 1 and j <= i + seg_len:
                            continue
                        
                        j_next = (j + 1) % n
                        city_j = tour[j]
                        city_j_next = tour[j_next]
                        
                        insertion_cost = (dist[city_j, seg_start] + dist[seg_end, city_j_next]
                                         - dist[city_j, city_j_next])
                        
                        delta = insertion_cost - removal_gain
                        
                        if delta < best_delta - 0.5:
                            best_delta = delta
                            best_j = j
                    
                    if best_j >= 0:
                        # Extract segment
                        segment = tour[i:i+seg_len]
                        del tour[i:i+seg_len]
                        
                        # Find new insertion position
                        insert_pos = best_j + 1 if best_j < i else best_j - seg_len + 1
                        insert_pos = max(0, min(insert_pos, len(tour)))
                        
                        for k, city in enumerate(segment):
                            tour.insert(insert_pos + k, city)
                        
                        improved = True
                        break
        
        return np.array(tour, dtype=int)

    def double_bridge(tour):
        """Double-bridge perturbation for escaping local optima."""
        if n < 8:
            return tour.copy()
        
        # Select 4 random cut points
        cuts = sorted(np.random.choice(n, 4, replace=False))
        p1, p2, p3, p4 = cuts
        
        new_tour = np.concatenate([
            tour[:p1+1],
            tour[p3+1:p4+1],
            tour[p2+1:p3+1],
            tour[p1+1:p2+1],
            tour[p4+1:]
        ])
        
        return new_tour if len(new_tour) == n else tour.copy()

    # Phase 1: Construction - try both methods
    construction_deadline = start_time + (time_limit_ms / 1000.0) * 0.15
    
    best_tour = None
    best_length = np.inf
    
    # Try greedy insertion from a few starts
    num_starts = min(n, 5)
    start_cities = [0] + list(np.random.choice(range(1, n), min(num_starts - 1, n - 1), replace=False))
    
    for start_city in start_cities:
        if time.perf_counter() >= construction_deadline:
            break
        
        # Try greedy insertion
        try:
            tour = greedy_insertion(start_city)
            length = calc_tour_length(tour)
            if length < best_length:
                best_length = length
                best_tour = tour.copy()
        except:
            pass
        
        # Also try nearest neighbor
        tour = nearest_neighbor(start_city)
        length = calc_tour_length(tour)
        if length < best_length:
            best_length = length
            best_tour = tour.copy()

    tour = best_tour
    
    # Phase 2: Local search
    local_search_deadline = start_time + (time_limit_ms / 1000.0) * 0.6
    
    while time.perf_counter() < local_search_deadline:
        old_length = calc_tour_length(tour)
        
        tour = two_opt_dl(tour)
        tour = or_opt(tour)
        
        new_length = calc_tour_length(tour)
        if new_length >= old_length - 0.5:
            break
    
    best_tour = tour.copy()
    best_length = calc_tour_length(best_tour)
    
    # Phase 3: Simulated annealing with double-bridge
    sa_deadline = deadline - 0.05  # Leave some margin
    
    temp = best_length * 0.02
    cooling = 0.995
    
    current_tour = tour.copy()
    current_length = best_length
    
    iterations = 0
    while time.perf_counter() < sa_deadline and temp > 1:
        iterations += 1
        
        # Perturb with double-bridge
        new_tour = double_bridge(current_tour)
        
        # Quick 2-opt improvement
        new_tour = two_opt_dl(new_tour)
        new_length = calc_tour_length(new_tour)
        
        # Accept or reject
        delta = new_length - current_length
        if delta < 0 or np.random.random() < np.exp(-delta / temp):
            current_tour = new_tour
            current_length = new_length
            
            if current_length < best_length:
                best_tour = current_tour.copy()
                best_length = current_length
        
        temp *= cooling
    
    # Final polish
    if time.perf_counter() < deadline:
        best_tour = two_opt_dl(best_tour)
        best_tour = or_opt(best_tour)
    
    length = calc_tour_length(best_tour)
    
    return best_tour.tolist(), length


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
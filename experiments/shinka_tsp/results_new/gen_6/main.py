# EVOLVE-BLOCK-START
"""
TSP Solver: Fast Multi-start with Simulated Annealing Hybrid

Uses farthest insertion construction, efficient 2-opt with candidate lists,
and simulated annealing for escaping local optima.
"""

import numpy as np
import time
import random


def solve_tsp(coords: np.ndarray, time_limit_ms: int = 5000) -> tuple:
    """
    Solve TSP instance.
    """
    n = len(coords)
    if n <= 1:
        return list(range(n)), 0.0
    if n == 2:
        return [0, 1], 2 * np.sqrt(np.sum((coords[0] - coords[1]) ** 2))

    # Vectorized distance matrix computation
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.round(np.sqrt(np.sum(diff ** 2, axis=2)))
    
    start_time = time.perf_counter()
    deadline = start_time + time_limit_ms / 1000.0
    
    # Precompute k-nearest neighbors for each city
    k_nearest = min(15, n - 1)
    nearest = np.argsort(dist, axis=1)[:, 1:k_nearest+1]

    def tour_length(tour):
        total = 0.0
        for i in range(n):
            total += dist[tour[i], tour[(i + 1) % n]]
        return total

    def nearest_neighbor(start):
        visited = np.zeros(n, dtype=bool)
        tour = np.zeros(n, dtype=int)
        current = start
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

    def farthest_insertion(start=0):
        """Farthest insertion heuristic - often produces better initial tours"""
        in_tour = np.zeros(n, dtype=bool)
        tour = [start]
        in_tour[start] = True
        
        # Find farthest city from start
        farthest = np.argmax(dist[start])
        tour.append(farthest)
        in_tour[farthest] = True
        
        while len(tour) < n:
            # Find farthest city from tour
            best_city = -1
            best_dist = -1
            for c in range(n):
                if in_tour[c]:
                    continue
                min_dist = min(dist[c, t] for t in tour)
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_city = c
            
            # Find best position to insert
            best_pos = 0
            best_increase = np.inf
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                increase = dist[tour[i], best_city] + dist[best_city, tour[j]] - dist[tour[i], tour[j]]
                if increase < best_increase:
                    best_increase = increase
                    best_pos = i + 1
            
            tour.insert(best_pos, best_city)
            in_tour[best_city] = True
        
        return np.array(tour)

    def two_opt_fast(tour):
        """2-opt with don't-look bits and early termination"""
        tour = tour.copy()
        position = np.zeros(n, dtype=int)
        for i in range(n):
            position[tour[i]] = i
        
        dont_look = np.zeros(n, dtype=bool)
        improved = True
        
        while improved and time.perf_counter() < deadline:
            improved = False
            for idx in range(n):
                if dont_look[tour[idx]]:
                    continue
                if time.perf_counter() >= deadline:
                    break
                    
                i = idx
                a = tour[i]
                b = tour[(i + 1) % n]
                
                found = False
                # Check moves involving neighbors
                for c in nearest[a]:
                    j = position[c]
                    if j == i or j == (i + 1) % n or j == (i - 1) % n:
                        continue
                    
                    d = tour[(j + 1) % n]
                    delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])
                    
                    if delta < -0.5:
                        # Perform 2-opt swap
                        if i < j:
                            tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        else:
                            tour[j+1:i+1] = tour[j+1:i+1][::-1]
                        
                        # Update positions
                        for k in range(n):
                            position[tour[k]] = k
                        
                        dont_look[a] = False
                        dont_look[b] = False
                        dont_look[c] = False
                        dont_look[d] = False
                        improved = True
                        found = True
                        break
                
                if not found:
                    dont_look[tour[idx]] = True
        
        return tour

    def two_opt_full(tour):
        """Full 2-opt when time permits"""
        tour = tour.copy()
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            for i in range(n - 1):
                if time.perf_counter() >= deadline:
                    break
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[(j + 1) % n]
                    delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])
                    if delta < -0.5:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
        return tour

    def or_opt_fast(tour):
        """Or-opt: relocate segments of 1-3 cities using candidate lists"""
        tour = list(tour)
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            for seg_len in [1, 2, 3]:
                if improved or time.perf_counter() >= deadline:
                    break
                for i in range(n):
                    if improved or time.perf_counter() >= deadline:
                        break
                    
                    prev_i = (i - 1) % n
                    next_seg = (i + seg_len) % n
                    seg_start = tour[i]
                    seg_end = tour[(i + seg_len - 1) % n]
                    
                    removal_gain = dist[tour[prev_i], seg_start] + dist[seg_end, tour[next_seg]] - dist[tour[prev_i], tour[next_seg]]
                    
                    # Try inserting near neighbors of segment endpoints
                    for target in nearest[seg_start]:
                        if target in tour[i:i+seg_len]:
                            continue
                        j = tour.index(target)
                        
                        if j >= i - 1 and j <= (i + seg_len) % n:
                            continue
                        
                        next_j = (j + 1) % n
                        insert_cost = dist[tour[j], seg_start] + dist[seg_end, tour[next_j]] - dist[tour[j], tour[next_j]]
                        
                        if insert_cost < removal_gain - 0.5:
                            segment = [tour[(i + k) % n] for k in range(seg_len)]
                            new_tour = [tour[k] for k in range(n) if k < i or k >= i + seg_len]
                            
                            if j < i:
                                insert_pos = j + 1
                            else:
                                insert_pos = j - seg_len + 1
                            
                            insert_pos = max(0, min(insert_pos, len(new_tour)))
                            tour = new_tour[:insert_pos] + segment + new_tour[insert_pos:]
                            improved = True
                            break
        return np.array(tour)

    def simulated_annealing(tour, max_iterations=1000):
        """Simulated annealing to escape local optima"""
        tour = tour.copy()
        current_len = tour_length(tour)
        best_tour = tour.copy()
        best_len = current_len
        
        temp = current_len * 0.01
        cooling = 0.995
        
        for iteration in range(max_iterations):
            if time.perf_counter() >= deadline:
                break
            
            # Random 2-opt move
            i = random.randint(0, n - 2)
            j = random.randint(i + 2, n - 1 if i > 0 else n - 2)
            
            a, b = tour[i], tour[i + 1]
            c, d = tour[j], tour[(j + 1) % n]
            delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])
            
            if delta < 0 or random.random() < np.exp(-delta / temp):
                tour[i+1:j+1] = tour[i+1:j+1][::-1]
                current_len += delta
                
                if current_len < best_len:
                    best_len = current_len
                    best_tour = tour.copy()
            
            temp *= cooling
        
        return best_tour

    # Multi-start strategy
    best_tour = None
    best_length = np.inf
    
    # Phase 1: Quick multi-start with fast local search
    phase1_deadline = start_time + time_limit_ms / 1000.0 * 0.5
    
    # Try farthest insertion + nearest neighbor starts
    constructions = []
    for start in range(0, n, max(1, n // 8)):
        constructions.append(('nn', start))
    for start in range(0, n, max(1, n // 4)):
        constructions.append(('fi', start))
    
    for method, start in constructions:
        if time.perf_counter() >= phase1_deadline:
            break
        
        if method == 'nn':
            tour = nearest_neighbor(start)
        else:
            tour = farthest_insertion(start)
        
        tour = two_opt_fast(tour)
        length = tour_length(tour)
        
        if length < best_length:
            best_length = length
            best_tour = tour.copy()
    
    # Phase 2: Intensive improvement on best tour
    if best_tour is not None and time.perf_counter() < deadline:
        best_tour = two_opt_full(best_tour)
        best_tour = or_opt_fast(best_tour)
        best_tour = two_opt_fast(best_tour)
        best_length = tour_length(best_tour)
    
    # Phase 3: Simulated annealing
    if best_tour is not None and time.perf_counter() < deadline:
        remaining_time = deadline - time.perf_counter()
        sa_iterations = int(remaining_time * 5000)  # Roughly 5000 iterations per second
        sa_tour = simulated_annealing(best_tour, max_iterations=sa_iterations)
        sa_tour = two_opt_fast(sa_tour)
        sa_length = tour_length(sa_tour)
        
        if sa_length < best_length:
            best_length = sa_length
            best_tour = sa_tour
    
    # Final polish
    while time.perf_counter() < deadline:
        old_length = best_length
        best_tour = two_opt_full(best_tour)
        best_length = tour_length(best_tour)
        if best_length >= old_length - 0.5:
            break

    return best_tour.tolist(), best_length


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

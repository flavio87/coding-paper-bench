# EVOLVE-BLOCK-START
"""
TSP Solver: Simulated Annealing with Cheapest Insertion Construction

Uses cheapest insertion for initial tour, then simulated annealing
with mixed 2-opt and relocate moves for optimization.
"""

import numpy as np
import time
import math


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

    # Compute distance matrix using vectorized operations
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.round(np.sqrt(np.sum(diff ** 2, axis=2)))

    start_time = time.perf_counter()
    deadline = start_time + time_limit_ms / 1000.0

    def calc_tour_length(tour):
        length = 0
        for i in range(len(tour)):
            length += dist[tour[i], tour[(i + 1) % len(tour)]]
        return length

    def cheapest_insertion():
        """Build tour using cheapest insertion heuristic."""
        if n <= 3:
            return np.arange(n, dtype=int)
        
        # Start with triangle of 3 furthest apart cities
        # Find two furthest cities
        max_dist = 0
        c1, c2 = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] > max_dist:
                    max_dist = dist[i, j]
                    c1, c2 = i, j
        
        # Find third city furthest from both
        max_min_dist = 0
        c3 = 0
        for i in range(n):
            if i != c1 and i != c2:
                min_d = min(dist[i, c1], dist[i, c2])
                if min_d > max_min_dist:
                    max_min_dist = min_d
                    c3 = i
        
        tour = [c1, c2, c3]
        in_tour = np.zeros(n, dtype=bool)
        in_tour[c1] = in_tour[c2] = in_tour[c3] = True
        
        # Insert remaining cities
        while len(tour) < n:
            best_city = -1
            best_pos = -1
            best_cost = np.inf
            
            for city in range(n):
                if in_tour[city]:
                    continue
                
                # Find best position to insert this city
                for pos in range(len(tour)):
                    prev = tour[pos]
                    next_city = tour[(pos + 1) % len(tour)]
                    cost = dist[prev, city] + dist[city, next_city] - dist[prev, next_city]
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_city = city
                        best_pos = pos + 1
            
            tour.insert(best_pos, best_city)
            in_tour[best_city] = True
        
        return np.array(tour, dtype=int)

    def nearest_neighbor(start_city):
        """Construct tour using nearest neighbor from given start city."""
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

    def two_opt_delta(tour, i, j):
        """Calculate delta for 2-opt move."""
        a, b = tour[i], tour[(i + 1) % n]
        c, d = tour[j], tour[(j + 1) % n]
        return (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])

    def apply_two_opt(tour, i, j):
        """Apply 2-opt move."""
        new_tour = tour.copy()
        new_tour[i+1:j+1] = new_tour[i+1:j+1][::-1]
        return new_tour

    def relocate_delta(tour, i, j):
        """Calculate delta for relocating city at position i to after position j."""
        if j == (i - 1) % n or j == i:
            return 0
        
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        j_next = (j + 1) % n
        
        city = tour[i]
        
        # Cost of removing
        removal = dist[tour[prev_i], city] + dist[city, tour[next_i]] - dist[tour[prev_i], tour[next_i]]
        
        # Cost of inserting
        insertion = dist[tour[j], city] + dist[city, tour[j_next]] - dist[tour[j], tour[j_next]]
        
        return insertion - removal

    def apply_relocate(tour, i, j):
        """Relocate city at position i to after position j."""
        tour_list = tour.tolist()
        city = tour_list.pop(i)
        insert_pos = j if j < i else j
        tour_list.insert(insert_pos + 1, city)
        return np.array(tour_list, dtype=int)

    def double_bridge(tour):
        """Apply double-bridge perturbation."""
        if n < 8:
            return tour
        
        # Select 4 random positions
        positions = sorted(np.random.choice(n, 4, replace=False))
        p1, p2, p3, p4 = positions
        
        new_tour = np.concatenate([
            tour[:p1+1],
            tour[p3+1:p4+1],
            tour[p2+1:p3+1],
            tour[p1+1:p2+1],
            tour[p4+1:]
        ])
        
        return new_tour if len(new_tour) == n else tour

    def full_two_opt(tour):
        """Run 2-opt until no improvement."""
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            for i in range(n - 1):
                if time.perf_counter() >= deadline:
                    break
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue
                    delta = two_opt_delta(tour, i, j)
                    if delta < -0.5:
                        tour = apply_two_opt(tour, i, j)
                        improved = True
        return tour

    # Phase 1: Construction - try both methods
    construction_deadline = start_time + (time_limit_ms / 1000.0) * 0.15
    
    best_tour = None
    best_length = np.inf
    
    # Try cheapest insertion
    tour1 = cheapest_insertion()
    len1 = calc_tour_length(tour1)
    if len1 < best_length:
        best_length = len1
        best_tour = tour1.copy()
    
    # Try multiple nearest neighbor starts
    num_starts = min(n, max(3, n // 20))
    start_cities = [0] + list(np.random.choice(range(1, n), min(num_starts - 1, n - 1), replace=False)) if n > 1 else [0]
    
    for start_city in start_cities:
        if time.perf_counter() >= construction_deadline:
            break
        tour = nearest_neighbor(start_city)
        length = calc_tour_length(tour)
        if length < best_length:
            best_length = length
            best_tour = tour.copy()

    tour = best_tour
    current_length = best_length

    # Phase 2: Initial 2-opt improvement
    tour = full_two_opt(tour)
    current_length = calc_tour_length(tour)
    best_tour = tour.copy()
    best_length = current_length

    # Phase 3: Simulated Annealing
    # Temperature schedule
    initial_temp = current_length * 0.05
    final_temp = 0.1
    
    temp = initial_temp
    cooling_rate = 0.9995
    
    iterations_without_improvement = 0
    max_no_improve = n * 5
    
    while time.perf_counter() < deadline:
        # Generate random move
        move_type = np.random.random()
        
        if move_type < 0.7:  # 2-opt move
            i = np.random.randint(0, n - 1)
            j = np.random.randint(i + 2, n + 1)
            if j >= n:
                j = j % n
                if j <= i + 1:
                    continue
            if j == n - 1 and i == 0:
                continue
            
            delta = two_opt_delta(tour, i, j)
            
            if delta < 0 or np.random.random() < math.exp(-delta / temp):
                tour = apply_two_opt(tour, i, j)
                current_length += delta
                
                if current_length < best_length - 0.5:
                    best_length = current_length
                    best_tour = tour.copy()
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
        else:  # Relocate move
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if j == i or j == (i - 1) % n:
                continue
            
            delta = relocate_delta(tour, i, j)
            
            if delta < 0 or np.random.random() < math.exp(-delta / temp):
                tour = apply_relocate(tour, i, j)
                current_length += delta
                
                if current_length < best_length - 0.5:
                    best_length = current_length
                    best_tour = tour.copy()
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
        
        # Cool down
        temp *= cooling_rate
        if temp < final_temp:
            temp = final_temp
        
        # Restart if stuck
        if iterations_without_improvement > max_no_improve:
            # Perturb and restart
            tour = double_bridge(best_tour)
            tour = full_two_opt(tour)
            current_length = calc_tour_length(tour)
            
            if current_length < best_length:
                best_length = current_length
                best_tour = tour.copy()
            
            temp = initial_temp * 0.5
            iterations_without_improvement = 0

    # Final 2-opt on best tour
    tour = full_two_opt(best_tour)
    length = calc_tour_length(tour)
    
    if length < best_length:
        best_tour = tour
        best_length = length

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
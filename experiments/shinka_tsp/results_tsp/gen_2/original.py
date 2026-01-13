# EVOLVE-BLOCK-START
"""
TSP Solver: Multi-start Christofides + 3-opt Local Search

This solver uses Christofides algorithm for construction and 3-opt for improvement,
with multi-start strategy for better solutions.
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

    best_tour = None
    best_length = np.inf

    # Multi-start approach - try different starting points
    max_starts = min(n, 10)  # Limit starts for larger instances
    time_per_start = (time_limit_ms / 1000.0) / max_starts

    for start_city in range(max_starts):
        if time.perf_counter() >= deadline:
            break

        start_deadline = time.perf_counter() + time_per_start

        # Phase 1: Christofides-inspired construction
        tour = christofides_construction(dist, n, start_city)

        # Phase 2: 3-opt improvement
        tour = three_opt_improvement(tour, dist, n, start_deadline)

        # Calculate tour length
        length = sum(dist[tour[i], tour[(i + 1) % n]] for i in range(n))

        if length < best_length:
            best_length = length
            best_tour = tour[:]

    return best_tour, best_length


def christofides_construction(dist, n, start_city=0):
    """Christofides-inspired construction heuristic."""
    # Build MST using Prim's algorithm
    mst_edges = []
    visited = np.zeros(n, dtype=bool)
    visited[start_city] = True

    for _ in range(n - 1):
        min_edge = (np.inf, -1, -1)
        for i in range(n):
            if visited[i]:
                for j in range(n):
                    if not visited[j] and dist[i, j] < min_edge[0]:
                        min_edge = (dist[i, j], i, j)

        _, u, v = min_edge
        mst_edges.append((u, v))
        visited[v] = True

    # Build adjacency list from MST
    adj = [[] for _ in range(n)]
    for u, v in mst_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Find odd degree vertices
    odd_vertices = [i for i in range(n) if len(adj[i]) % 2 == 1]

    # Simple minimum weight perfect matching approximation
    # Greedily pair closest odd vertices
    used = set()
    for i in range(0, len(odd_vertices), 2):
        if i + 1 < len(odd_vertices):
            u, v = odd_vertices[i], odd_vertices[i + 1]
            if u not in used and v not in used:
                adj[u].append(v)
                adj[v].append(u)
                used.add(u)
                used.add(v)

    # Find Eulerian circuit and convert to Hamiltonian
    tour = eulerian_to_hamiltonian(adj, n, start_city)
    return tour


def eulerian_to_hamiltonian(adj, n, start):
    """Convert Eulerian multigraph to Hamiltonian tour via DFS."""
    visited = np.zeros(n, dtype=bool)
    tour = []

    def dfs(v):
        if visited[v]:
            return
        visited[v] = True
        tour.append(v)
        for u in adj[v]:
            if not visited[u]:
                dfs(u)

    dfs(start)

    # Add any unvisited vertices (shouldn't happen in connected graph)
    for i in range(n):
        if not visited[i]:
            tour.append(i)

    return tour


def three_opt_improvement(tour, dist, n, deadline):
    """3-opt local search improvement."""
    tour = tour[:]
    improved = True

    while improved and time.perf_counter() < deadline:
        improved = False

        for i in range(n - 2):
            if time.perf_counter() >= deadline:
                break

            for j in range(i + 2, n - 1):
                if time.perf_counter() >= deadline:
                    break

                for k in range(j + 2, n):
                    if k == i:
                        continue

                    # Current edges: (i,i+1), (j,j+1), (k,k+1)
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[j + 1]
                    e, f = tour[k], tour[(k + 1) % n]

                    current_cost = dist[a, b] + dist[c, d] + dist[e, f]

                    # Try different 3-opt moves
                    moves = [
                        # Move 1: reverse segment (i+1, j)
                        (dist[a, c] + dist[b, d] + dist[e, f], 1),
                        # Move 2: reverse segment (j+1, k)
                        (dist[a, b] + dist[c, e] + dist[d, f], 2),
                        # Move 3: reverse both segments
                        (dist[a, c] + dist[b, e] + dist[d, f], 3),
                        # Move 4: relocate segment (i+1, j) after k
                        (dist[a, d] + dist[c, f] + dist[e, b], 4),
                        # Move 5: relocate segment (j+1, k) after i
                        (dist[a, d] + dist[c, b] + dist[e, f], 5)
                    ]

                    best_cost, best_move = min(moves)

                    if best_cost < current_cost - 1e-9:
                        # Apply the best move
                        if best_move == 1:
                            tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        elif best_move == 2:
                            tour[j+1:k+1] = tour[j+1:k+1][::-1]
                        elif best_move == 3:
                            tour[i+1:j+1] = tour[i+1:j+1][::-1]
                            tour[j+1:k+1] = tour[j+1:k+1][::-1]
                        elif best_move == 4:
                            # Relocate (i+1, j) after k
                            segment = tour[i+1:j+1]
                            tour = tour[:i+1] + tour[j+1:k+1] + segment + tour[k+1:]
                        elif best_move == 5:
                            # Relocate (j+1, k) after i
                            segment = tour[j+1:k+1]
                            tour = tour[:i+1] + segment + tour[i+1:j+1] + tour[k+1:]

                        improved = True
                        break

                if improved:
                    break
            if improved:
                break

    return tour


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
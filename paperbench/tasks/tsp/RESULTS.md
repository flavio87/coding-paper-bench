# TSP Baseline Results

## Summary of Experiments

This document presents the results of running the **Nearest Neighbor + 2-opt** baseline on various TSP instances.

---

## Key Findings

### 1. Algorithm Performance

| Metric | Value |
|--------|-------|
| Typical improvement from 2-opt | 10-20% over Nearest Neighbor |
| Time complexity | Scales roughly quadratically with problem size |
| Convergence | Usually converges within 3-6 iterations |

### 2. Scaling Behavior (Uniform Distribution)

| Size | NN Length | 2-opt Length | Improvement | Time (ms) |
|------|-----------|--------------|-------------|-----------|
| 10   | 2,646     | 2,646        | 0.0%        | 0.1       |
| 20   | 4,155     | 3,428        | 17.5%       | 0.9       |
| 50   | 7,145     | 5,700        | 20.2%       | 5.4       |
| 100  | 9,977     | 8,225        | 17.6%       | 34.0      |
| 200  | 12,628    | 11,151       | 11.7%       | 87.2      |
| 500  | 20,505    | 17,804       | 13.2%       | 840.6     |

**Observation**: 2-opt provides consistent 10-20% improvement over nearest neighbor, with diminishing returns at larger sizes due to local search getting trapped in local minima.

### 3. Impact of Spatial Distribution

| Distribution | Avg Length | Std Dev | Notes |
|--------------|------------|---------|-------|
| Uniform      | 6,264      | 193     | Hardest case, most variance |
| Clustered    | 3,050      | 486     | Easier, high variance between instances |
| Circular     | 2,929      | 73      | Easiest, very consistent |
| Grid         | 6,475      | 223     | Similar to uniform |

**Observation**: Circular distributions are easiest (cities naturally form a tour). Clustered instances show high variance due to cluster arrangement differences.

### 4. Sensitivity to Random Seed

For a 100-city uniform instance across 20 seeds:
- **Best solution**: 8,165
- **Worst solution**: 9,165
- **Gap**: 12.25%

**Observation**: Starting city selection significantly impacts final solution quality. Multi-start approaches could yield ~12% improvement.

### 5. Time Limit Impact

| Time Limit | Solution | Converged? |
|------------|----------|------------|
| 10ms       | 12,400   | No         |
| 50ms       | 11,193   | Partial    |
| 100ms+     | 11,151   | Yes        |

**Observation**: For 200 cities, the algorithm converges in ~90ms. Additional time provides no benefit once local optimum is reached.

### 6. 2-opt Convergence Analysis (50 cities)

| Iteration | Swaps | Tour Length | Cumulative Improvement |
|-----------|-------|-------------|------------------------|
| Initial   | -     | 7,448       | -                      |
| 1         | 8     | 6,750       | 9.4%                   |
| 2         | 9     | 6,253       | 16.0%                  |
| 3         | 3     | 6,213       | 16.6%                  |

**Observation**: Most improvement happens in early iterations. Later iterations find fewer improving moves.

---

## Interesting Observations

### Edge Cases

1. **Square (4 cities)**: Optimal tour found (length 4000 = perimeter)
2. **Pentagon (5 cities)**: Optimal tour found (length 2940)
3. **Line (20 cities)**: Near-optimal with 99.2% gap from MST lower bound, which is expected since MST lower bound is loose for degenerate cases

### Why Circular is Easiest

The circular distribution has cities approximately arranged in a ring. The optimal tour simply visits them in circular order. Nearest neighbor naturally tends toward this solution, and 2-opt easily fixes any mistakes.

### Why Clustered Has High Variance

The quality depends heavily on how clusters are connected. If nearest neighbor makes poor inter-cluster choices, 2-opt (which only reverses segments) cannot easily fix these global mistakes.

---

## Code

The baseline implementation is in `baseline.py`:
- **Nearest Neighbor**: O(n^2) construction heuristic
- **2-opt**: O(n^2) per iteration local search

## Running the Experiments

```bash
python3 paperbench/tasks/tsp/run_experiments.py
```

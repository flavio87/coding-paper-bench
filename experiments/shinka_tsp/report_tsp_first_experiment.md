# TSP Solver Evolution Experiment Report

**Date:** 2026-01-14
**Framework:** ShinkaEvolve (LLM-guided evolutionary code optimization)
**LLM:** Claude Opus 4.5 via OpenRouter
**Runtime:** 9 minutes 19 seconds
**Total API Cost:** $0.52

---

## 1. Objective

Test whether LLM-guided evolutionary optimization can improve a baseline TSP (Traveling Salesman Problem) solver. This experiment serves as a proof-of-concept for using frontier LLMs as intelligent mutation operators in code evolution.

## 2. Hypothesis

An LLM with domain knowledge about combinatorial optimization can suggest meaningful algorithmic improvements to a simple TSP heuristic, leading to measurably better solutions across a range of problem sizes.

## 3. Experimental Setup

### 3.1 Evolution Configuration

| Parameter | Value |
|-----------|-------|
| Generations | 10 |
| Islands | 2 (parallel evolutionary populations) |
| Parallel Jobs | 2 |
| Patch Types | 70% diff, 30% full rewrite |
| LLM Temperatures | [0.3, 0.7, 1.0] (sampled per generation) |
| Max Tokens | 8192 |

### 3.2 Test Instances

Random Euclidean TSP instances with coordinates in [0, 1000]²:

| Instance | Cities (n) | Seed | Time Limit |
|----------|------------|------|------------|
| random_20 | 20 | 42 | 5 sec |
| random_30 | 30 | 123 | 5 sec |
| random_50 | 50 | 456 | 5 sec |
| random_75 | 75 | 789 | 5 sec |
| random_100 | 100 | 101 | 5 sec |

### 3.3 Scoring

Each solver is evaluated on all 5 instances. The score is computed as:
```
score = mean(1000 / tour_length)  # Higher is better
```

---

## 4. Baseline Program

**Algorithm:** Nearest Neighbor construction + 2-opt local search

```python
def solve_tsp(coords, time_limit_ms=5000):
    # Phase 1: Nearest neighbor construction
    # Start from city 0, greedily visit nearest unvisited city

    # Phase 2: 2-opt improvement
    # While improving and within time limit:
    #   For each pair of edges (i,i+1) and (j,j+1):
    #     If reversing segment [i+1:j+1] reduces tour length:
    #       Apply the reversal
```

**Characteristics:**
- Simple O(n²) construction
- Basic 2-opt local search
- No multi-start, no advanced moves
- ~80 lines of code

---

## 5. Evolved Program (Best Solution)

**Algorithm:** Greedy Insertion + Don't-Look 2-opt + Or-opt + Simulated Annealing

The evolution discovered several key improvements:

### 5.1 Better Construction (Greedy Insertion)
```python
def greedy_insertion(start_city):
    # Start with triangle of 3 nearest cities
    # Iteratively insert remaining cities at position
    # that causes minimum tour length increase
```

### 5.2 Faster Local Search (Don't-Look Bits)
```python
def two_opt_dl(tour):
    # Skip nodes that haven't changed since last improvement
    # Reduces redundant edge evaluations
```

### 5.3 Or-opt Moves
```python
def or_opt(tour):
    # Relocate segments of 1, 2, or 3 consecutive cities
    # More powerful than pure 2-opt for certain configurations
```

### 5.4 Simulated Annealing with Double-Bridge
```python
def double_bridge(tour):
    # 4-opt perturbation to escape local optima

# SA loop accepts worse solutions probabilistically
# Allows exploration beyond local optima
```

### 5.5 Multi-Phase Strategy
1. **Construction (15% of time):** Try greedy insertion + nearest neighbor from multiple starts
2. **Local Search (45% of time):** Alternate 2-opt and Or-opt until no improvement
3. **Diversification (40% of time):** Simulated annealing with double-bridge perturbations
4. **Final Polish:** Quick 2-opt + Or-opt on best found

**Total:** ~300 lines of evolved code

---

## 6. Results

### 6.1 Evolution Progress

| Gen | Status | Score | Patch Name |
|-----|--------|-------|------------|
| 0 | Baseline | 0.181 | initial_program |
| 1 | ✓ Improved | 0.187 | multistart_oropt_improved |
| 2 | ✗ Invalid | 0.000 | multistart_oropt_optimized |
| 3 | ✓ **Best** | 0.190 | multistart_hybrid_lkh_inspired |
| 4 | ✓ | 0.189 | lin_kernighan_multistart |
| 5 | ✓ | 0.187 | improved_local_search_with_3opt |
| 6 | ✓ | 0.189 | fast_multistart_sa_hybrid |
| 7 | ✓ | 0.189 | improved_local_search_oropt_se |
| 8 | ✓ **Best** | 0.190 | greedy_insertion_sa_oropt |
| 9 | ✓ | 0.102 | simulated_annealing_cheapest_i |

**Success Rate:** 10/11 programs correct (91%)

### 6.2 Per-Instance Performance

| Instance | Baseline | Evolved | Improvement | Expected Optimal* |
|----------|----------|---------|-------------|-------------------|
| n=20 | 3,428 | 3,416 | +0.4% | ~3,186 |
| n=30 | 5,149 | 4,771 | **+7.3%** | ~3,902 |
| n=50 | 5,772 | 5,512 | +4.5% | ~5,037 |
| n=75 | 7,553 | 7,013 | **+7.1%** | ~6,170 |
| n=100 | 8,893 | 8,020 | **+9.8%** | ~7,124 |

*Expected optimal from Beardwood-Halton-Hammersley formula: E[tour] ≈ 0.7124 × √(n × L²)

### 6.3 Gap from Theoretical Optimal

| Instance | Baseline Gap | Evolved Gap | Gap Reduction |
|----------|--------------|-------------|---------------|
| n=20 | +7.6% | +7.2% | 0.4pp |
| n=30 | +32.0% | +22.3% | 9.7pp |
| n=50 | +14.6% | +9.4% | 5.2pp |
| n=75 | +22.4% | +13.7% | 8.7pp |
| n=100 | +24.8% | +12.6% | 12.2pp |

---

## 7. Comparison to State-of-the-Art

### 7.1 Best Known TSP Solvers

| Solver | Type | Typical Gap from Optimal |
|--------|------|--------------------------|
| **Concorde** | Exact (branch-and-cut) | 0% (optimal) |
| **LKH-3** | Heuristic (Lin-Kernighan-Helsgaun) | 0.01-0.1% |
| **EAX** | Genetic (Edge Assembly Crossover) | 0.01-0.1% |
| **Our Evolved** | Heuristic (2-opt + Or-opt + SA) | 7-22% |
| **Our Baseline** | Heuristic (NN + 2-opt) | 8-32% |

### 7.2 Gap Analysis

The evolved solver is **far from state-of-the-art**:

- **LKH/EAX** achieve gaps of <1% on instances of this size
- **Our evolved solver** achieves gaps of 7-22%
- **Gap to SOTA:** ~10-20x worse than LKH/EAX

### 7.3 What SOTA Does Differently

State-of-the-art TSP solvers use techniques not present in our evolved solution:

1. **Variable-depth k-opt (k up to 10+):** LKH uses sophisticated sequential/non-sequential moves
2. **Candidate edge lists:** Precompute promising edges using α-nearness from MST
3. **Sensitivity analysis:** Identify which edges are likely in optimal tour
4. **Population-based recombination:** EAX combines tours via edge assembly
5. **Backbone identification:** Fix edges that appear in many good solutions

---

## 8. Analysis

### 8.1 What the LLM Did Well

1. **Identified relevant techniques:** The LLM suggested Or-opt, simulated annealing, and multi-start - all legitimate TSP techniques
2. **Proper implementation:** Most generations produced valid, working code
3. **Incremental improvement:** Each successful generation built on previous ideas
4. **Time management:** Evolved solver uses phased approach respecting time limits

### 8.2 What the LLM Missed

1. **Lin-Kernighan moves:** Despite mentioning LK in patch names, no true variable-depth search was implemented
2. **Candidate lists:** No use of α-nearness or 1-tree bounds for pruning
3. **Data structures:** No neighbor lists, segment trees, or other speedup structures
4. **Theoretical grounding:** No use of LP relaxation bounds or Christofides algorithm

### 8.3 Why This Gap Exists

1. **Limited generations:** 10 generations may not be enough to discover complex algorithms
2. **No external knowledge:** LLM can only use its training knowledge, not reference implementations
3. **Black-box fitness:** Score doesn't tell LLM *why* a solution is suboptimal
4. **Prompt guidance:** The system prompt suggested basic techniques, not advanced ones

---

## 9. Conclusions

### 9.1 Experiment Success

The experiment **successfully demonstrated** that LLM-guided evolution can improve algorithmic code:

- **5% aggregate improvement** in score (0.181 → 0.190)
- **4-10% improvement** in tour lengths across instances
- **91% validity rate** (10/11 programs correct)
- Evolved solver uses more sophisticated multi-phase strategy

### 9.2 Limitations

The evolved solver remains **far from state-of-the-art**:

- 10-20x worse gap than LKH/EAX
- Missing key algorithmic innovations (variable-depth search, candidate lists)
- Would not be competitive in any TSP benchmark

### 9.3 Why TSP for This Experiment?

TSP is an ideal test problem for LLM-guided code evolution because:

1. **Well-defined objective:** Tour length is unambiguous
2. **Rich algorithmic landscape:** Many known techniques at different sophistication levels
3. **Clear correctness criteria:** Valid tour = visits all cities exactly once
4. **Scalable difficulty:** Can test on various problem sizes
5. **Known optimal baselines:** Can measure gap from true optimal
6. **LLM familiarity:** TSP is extensively covered in training data

### 9.4 Future Directions

To close the gap to SOTA, future experiments could:

1. **Longer evolution:** Run for 100+ generations
2. **Better prompts:** Include pseudocode for LKH/EAX in system prompt
3. **Intermediate feedback:** Tell LLM which instances are hardest
4. **Seed with better baseline:** Start from a basic LK implementation
5. **Use TSPLIB instances:** Test on standard benchmarks with known optima

---

## 10. Artifacts

- **Baseline code:** `initial.py`
- **Best evolved code:** `results_new/best/main.py`
- **Evolution database:** `results_new/evolution_db.sqlite`
- **Experiment config:** `results_new/experiment_config.yaml`
- **Full log:** `results_new/evolution_run.log`

---

## References

1. Helsgaun, K. (2000). "An effective implementation of the Lin-Kernighan traveling salesman heuristic." European Journal of Operational Research.
2. Nagata, Y., & Kobayashi, S. (2013). "A powerful genetic algorithm using edge assembly crossover for the traveling salesman problem." INFORMS Journal on Computing.
3. [Rigorous Performance Analysis of State-of-the-Art TSP Heuristic Solvers](https://www.cs.stir.ac.uk/~jad/RigorousPerf.pdf)
4. [TSP Algorithm Selection](https://tspalgsel.github.io/)
5. [ShinkaEvolve Paper](https://arxiv.org/abs/2509.19349)

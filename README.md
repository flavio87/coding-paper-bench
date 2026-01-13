# PaperBench: TSP Improvement Comparison

## What This Is

An experiment comparing two approaches for improving optimization algorithms using LLMs:

1. **Paper-Based Approach**: Single LLM call with research paper ideas (efficient, leverages human knowledge)
2. **ShinkaEvolve Approach**: Iterative LLM-based evolutionary search (multiple calls, discovers improvements through trial/error)

## Research Question

> Can a single-shot paper-based approach match or beat iterative LLM evolution for algorithm improvement?

## Repository Structure

```
coding-paper-bench/
├── README.md                          # This file
├── .gitignore                         # Ignores .env files with API keys
├── paperbench/
│   └── tasks/
│       └── tsp/
│           ├── baseline.py            # TSP solver: Nearest Neighbor + 2-opt
│           ├── run_experiments.py     # Baseline performance tests
│           └── RESULTS.md             # Baseline experiment results
└── experiments/
    └── tsp_comparison/
        ├── README.md                  # Experiment design documentation
        ├── .env.example               # Template for API keys
        ├── tsp_initial.py             # TSP baseline with EVOLVE-BLOCK markers
        ├── evaluate_tsp.py            # Scoring function for TSP solutions
        └── run_comparison.py          # Main comparison runner
```

## What's Been Done

### 1. TSP Baseline Implementation (`paperbench/tasks/tsp/`)
- Nearest Neighbor construction heuristic
- 2-opt local search improvement
- Tested on various instance sizes and distributions
- **Key finding**: 2-opt improves NN solutions by 10-20%

### 2. Comparison Experiment Framework (`experiments/tsp_comparison/`)
- Supports OpenRouter, Anthropic, and OpenAI APIs
- Paper-based: Prompts LLM with research ideas (Lin-Kernighan, Simulated Annealing, Or-opt)
- ShinkaEvolve: Iteratively mutates code based on fitness score
- Evaluation on synthetic TSP instances (20, 50, 100 cities)

### 3. ShinkaEvolve Integration
- Clone with: `git clone https://github.com/SakanaAI/ShinkaEvolve`
- Install in venv to avoid Debian setuptools issues:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -e /path/to/ShinkaEvolve
  ```

## How to Run the Experiment

### Prerequisites
```bash
pip install numpy openai anthropic python-dotenv
```

### Setup API Key
```bash
cd experiments/tsp_comparison
cp .env.example .env
# Edit .env and add your API key:
# OPENROUTER_API_KEY=sk-or-v1-...
# OR
# ANTHROPIC_API_KEY=sk-ant-...
```

### Run Comparison
```bash
cd experiments/tsp_comparison
python3 run_comparison.py --budget 10
```

### Expected Output
```
============================================================
TSP IMPROVEMENT COMPARISON
Paper-Based vs ShinkaEvolve
============================================================
Using openrouter API with model: anthropic/claude-sonnet-4

Baseline score: 1.9570

============================================================
APPROACH 1: Paper-Based (1 LLM call)
============================================================
Score: X.XXXX (improvement: +X.X%)

============================================================
APPROACH 2: ShinkaEvolve (10 LLM calls)
============================================================
  Initial score: 1.9570
  Gen 1: 1.9570 -> 2.0123 (+0.0553)
  ...
Final score: X.XXXX (improvement: +X.X%)

============================================================
SUMMARY
============================================================
Approach             LLM Calls    Score        Improvement  Efficiency
paper_based          1            X.XXXX       +X.X%        +X.X%/call
shinka_evolve        10           X.XXXX       +X.X%        +X.X%/call
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| **Score** | 10000 / tour_length (higher is better) |
| **Improvement** | % improvement over baseline |
| **Efficiency** | Improvement per LLM call |

## Network Note

If running in a restricted sandbox, ensure your API endpoint is accessible:
- **OpenRouter**: Requires `openrouter.ai` to be allowed
- **Anthropic**: Requires `api.anthropic.com` to be allowed

## Command Reference

```bash
# Run baseline TSP experiments only (no LLM needed)
python3 paperbench/tasks/tsp/run_experiments.py

# Run comparison with custom budget
python3 experiments/tsp_comparison/run_comparison.py --budget 5

# Run with specific model
python3 experiments/tsp_comparison/run_comparison.py --model anthropic/claude-3-5-sonnet

# Evaluate a solver directly
python3 experiments/tsp_comparison/evaluate_tsp.py --program_path tsp_initial.py
```

## Next Steps

1. **Run the comparison experiment** with proper API access
2. **Analyze results**: Does paper-based approach match ShinkaEvolve efficiency?
3. **Try different paper ideas** or models
4. **Extend to other problems**: CVRP, Max-Cut, BBOB (specs in original PaperBench doc)

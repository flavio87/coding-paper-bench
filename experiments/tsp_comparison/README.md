# Experiment: Paper-Based vs ShinkaEvolve for TSP

## Research Question

**Can a single-shot paper-based approach match or beat iterative LLM evolution (ShinkaEvolve) for algorithm improvement?**

## Approaches Compared

### 1. Paper-Based Approach (Ours)
- **Method**: Single LLM call with paper abstract + concrete ideas
- **LLM Budget**: 1 generation call
- **Input**: Paper abstract describing optimization technique
- **Output**: Improved solver code

### 2. ShinkaEvolve (Baseline)
- **Method**: Evolutionary search with LLM mutations
- **LLM Budget**: N generation calls (configurable, e.g., 50-200)
- **Input**: Task description + initial code
- **Output**: Best evolved solver from population

## Experimental Design

```
                    ┌─────────────────────┐
                    │   TSP Baseline      │
                    │ (NN + 2-opt)        │
                    └─────────┬───────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐
    │  Paper-Based    │             │   ShinkaEvolve  │
    │  (1 LLM call)   │             │  (N LLM calls)  │
    └────────┬────────┘             └────────┬────────┘
             │                               │
             ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐
    │ Improved Solver │             │ Evolved Solver  │
    └────────┬────────┘             └────────┬────────┘
             │                               │
             └───────────────┬───────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │     Evaluation      │
                    │  (Same test suite)  │
                    └─────────────────────┘
```

## Metrics

| Metric | Description |
|--------|-------------|
| **Solution Quality** | Gap from best-known solution (%) |
| **LLM Efficiency** | Quality improvement per LLM call |
| **Time Efficiency** | Wall-clock time to generate solver |
| **Success Rate** | % of runs producing valid, improved code |

## Test Instances

Using TSPLIB instances:
- **Public**: eil51, berlin52, kroA100, kroB100 (tune/validate)
- **Hidden**: pr124, bier127, ch130, pr136 (final evaluation)

## Fair Comparison

To make the comparison fair:
1. **Same evaluation**: Both approaches tested on identical instances
2. **Same base model**: Both use same underlying LLM (e.g., Claude Sonnet)
3. **Budget parity**: Compare at equal LLM token budgets
4. **Multiple seeds**: Run each approach 5x with different seeds

## Expected Insights

1. **Sample efficiency**: Paper-based may win on efficiency (quality/call)
2. **Peak performance**: ShinkaEvolve may find better solutions given budget
3. **Knowledge transfer**: Paper-based leverages human research insights
4. **Exploration**: ShinkaEvolve explores novel combinations

## Requirements

- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` environment variable
- Python 3.10+
- numpy, scipy

## Running the Experiment

```bash
# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Run comparison
python run_comparison.py --budget 50 --seeds 5
```

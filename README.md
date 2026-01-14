# PaperBench: TSP Improvement Comparison

## Overview

This project compares two approaches for improving optimization algorithms using LLMs:

| Approach | Description | LLM Calls | Strategy |
|----------|-------------|-----------|----------|
| **Paper-Based** | Single prompt with research paper ideas | 1 | Leverage human knowledge |
| **Naive ShinkaEvolve** | Greedy hill-climbing with LLM mutations | N | Simple iteration |
| **Real ShinkaEvolve** | Full evolutionary search (islands, archives, weighted selection) | N | Diverse exploration |

## Results Summary (Opus 4.5, 20 iterations)

| Approach | LLM Calls | Improvement | Efficiency |
|----------|-----------|-------------|------------|
| Paper-Based | 1 | +2.5% | +2.54%/call |
| Naive ShinkaEvolve | 20 | +3.0% | +0.15%/call |
| **Real ShinkaEvolve** | 20 | **+5.5%** | +0.28%/call |

**Key Finding**: Real ShinkaEvolve's island-based exploration outperforms naive greedy by ~2x, finding 8 diverse top-performing solutions.

---

## Quick Start

```bash
# 1. Clone the repository (includes modified ShinkaEvolve with OpenRouter support)
git clone https://github.com/flavio87/coding-paper-bench.git
cd coding-paper-bench

# 2. Create venv and install ShinkaEvolve
cd ShinkaEvolve
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cd ..

# 3. Set your API key
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"

# 4. Run an experiment
cd experiments/shinka_tsp
python run_evo.py
```

That's it! The repository includes a modified version of [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) with OpenRouter support already integrated.

---

## Visualization

ShinkaEvolve includes a web UI for visualizing evolution results, showing:
- **Evolution Tree**: Parent-child relationships between solutions
- **Island Visualization**: How solutions migrate between islands
- **Performance Metrics**: Fitness over generations
- **Code Diffs**: What changed between parent and child

### Launch the Visualization

```bash
# Make sure you're in the ShinkaEvolve venv
source ShinkaEvolve/.venv/bin/activate

# Launch the web UI pointing to your results
cd experiments/shinka_tsp
python -m shinka.webui.visualization results_tsp_opus/ --port 8888 --open

# Or use the CLI command directly
shinka_visualize results_tsp_opus/ --port 8888 --open
```

### Access the Web UI

- **Local**: Open http://localhost:8888 in your browser
- **Remote/SSH**: Create an SSH tunnel first:
  ```bash
  ssh -L 8888:localhost:8888 user@remote-host
  ```
  Then open http://localhost:8888 locally

### Web UI Features

1. **Database Browser**: Select which experiment database to view
2. **Evolution Tree**: Interactive visualization of solution genealogies
3. **Code Diff Viewer**: Compare parent vs child code
4. **Meta Files**: View generation-by-generation logs
5. **Real-time Updates**: Auto-refreshes during running experiments

---

## Running Experiments

### Naive Comparison (Paper-Based vs Simple ShinkaEvolve)

```bash
cd experiments/tsp_comparison
python run_comparison.py --budget 20 --model anthropic/claude-opus-4.5
```

### Real ShinkaEvolve

Edit `experiments/shinka_tsp/run_evo.py` to configure:

```python
# Model selection
model = "openrouter/anthropic/claude-opus-4.5"  # or claude-sonnet-4, gemini-3-pro

# Evolution parameters
num_islands = 2           # Number of parallel populations
pop_size = 10            # Solutions per island
num_generations = 20     # Evolution iterations
```

Then run:

```bash
cd experiments/shinka_tsp
python run_evo.py
```

---

## OpenRouter Integration

The included ShinkaEvolve has been modified to support OpenRouter. Key changes (already applied):

### 1. `shinka/llm/models/pricing.py`
Added OpenRouter model pricing:
```python
OPENROUTER_MODELS = {
    "openrouter/anthropic/claude-sonnet-4": {...},
    "openrouter/anthropic/claude-opus-4.5": {...},
    "openrouter/google/gemini-3-pro-preview": {...},
}
```

### 2. `shinka/llm/models/openrouter.py` (new file)
Created dedicated query function using Chat Completions API:
```python
def query_openrouter(client, model_name, system_prompt, user_prompt, **kwargs):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        **kwargs
    )
    return response.choices[0].message.content
```

### 3. `shinka/llm/client.py`
Added OpenRouter client creation:
```python
if model_name.startswith("openrouter/"):
    client = openai.OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
```

### 4. `shinka/llm/query.py`
Updated dispatch to route OpenRouter models correctly.

---

## Repository Structure

```
coding-paper-bench/
├── README.md                              # This file
├── .gitignore
├── ShinkaEvolve/                          # Modified ShinkaEvolve with OpenRouter support
│   └── shinka/                            # Core library (see OpenRouter Integration below)
├── experiments/
│   ├── tsp_comparison/                    # Naive comparison experiment
│   │   ├── run_comparison.py              # Paper-based vs simple evolution
│   │   ├── tsp_initial.py                 # TSP baseline with EVOLVE-BLOCK
│   │   └── evaluate_tsp.py                # Scoring function
│   └── shinka_tsp/                        # Real ShinkaEvolve experiment
│       ├── run_evo.py                     # Main experiment runner
│       ├── initial.py                     # TSP baseline code
│       ├── evaluate.py                    # Evaluation function
│       └── results_tsp_opus/              # Opus 4.5 experiment results
│           ├── gen_*/                     # Each generation's code and results
│           └── evolution_run.log          # Experiment log
└── paperbench/
    └── tasks/tsp/                         # TSP baseline implementation
```

---

## Experiment Results

### Results Directory Structure

Each experiment creates a `results_*` directory with:

```
results_tsp_opus/
├── experiment_config.yaml     # Experiment configuration
├── evolution_db.sqlite        # SQLite database for visualization
├── evolution_run.log          # Detailed run log
├── gen_0/                     # Initial generation
│   ├── main.py               # The evolved code
│   └── results/
│       ├── metrics.json      # Performance metrics
│       └── correct.json      # Correctness check
├── gen_1/                     # First evolution
│   ├── main.py               # Evolved code
│   ├── original.py           # Parent code
│   ├── edit.diff             # What changed
│   └── results/
└── best/                      # Best solution found
    ├── main.py
    └── results/
```

### Viewing Past Experiments

```bash
# List available experiments
ls experiments/shinka_tsp/results_*/

# Visualize a specific experiment
shinka_visualize experiments/shinka_tsp/results_tsp_opus/ --port 8888 --open
```

---

## Troubleshooting

### "Module shinka not found"
Make sure you activated the ShinkaEvolve venv:
```bash
source ShinkaEvolve/.venv/bin/activate
```

### "Invalid model ID" for OpenRouter
Use the correct model format: `openrouter/anthropic/claude-opus-4.5` (not `anthropic/claude-opus-4-5-20251101`)

### "Database is locked" in visualization
The experiment might still be running. Wait for it to finish or use read-only mode.

### Port already in use
```bash
shinka_visualize --port 9000  # Use a different port
```

---

## Further Experiments

### Try Different Models
```python
# In run_evo.py, change the model:
model = "openrouter/google/gemini-3-pro-preview"
model = "openrouter/anthropic/claude-sonnet-4"
```

### Adjust Evolution Parameters
```python
num_islands = 4      # More diversity
pop_size = 20        # Larger populations
num_generations = 50 # Longer evolution
```

### Try Different Problems
The framework can be extended to other optimization problems. See the `paperbench/tasks/` directory for examples.

---

## License

MIT License - See LICENSE file for details.

## Credits

- [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) by Sakana AI
- TSP optimization techniques from various research papers

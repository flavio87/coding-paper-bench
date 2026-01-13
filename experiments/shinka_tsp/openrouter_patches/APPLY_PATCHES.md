# OpenRouter Patches for ShinkaEvolve

This directory contains the modifications needed to make ShinkaEvolve work with OpenRouter API.

## Quick Apply

```bash
# From the repo root directory
cp experiments/shinka_tsp/openrouter_patches/openrouter.py ShinkaEvolve/shinka/llm/models/
```

Then manually apply the changes below to the other files.

## Required Modifications

### 1. `ShinkaEvolve/shinka/llm/models/pricing.py`

Add the following dictionary after the existing model definitions:

```python
OPENROUTER_MODELS = {
    "openrouter/anthropic/claude-sonnet-4": {
        "input_price": 3.0 / M,
        "output_price": 15.0 / M,
    },
    "openrouter/anthropic/claude-opus-4.5": {
        "input_price": 3.0 / M,
        "output_price": 15.0 / M,
    },
    "openrouter/anthropic/claude-3.5-sonnet": {
        "input_price": 3.0 / M,
        "output_price": 15.0 / M,
    },
    "openrouter/openai/gpt-4o": {
        "input_price": 2.5 / M,
        "output_price": 10.0 / M,
    },
    "openrouter/google/gemini-3-pro-preview": {
        "input_price": 2.0 / M,
        "output_price": 12.0 / M,
    },
}
```

### 2. `ShinkaEvolve/shinka/llm/models/__init__.py`

Add the export for the new query function:

```python
from .openrouter import query_openrouter
```

### 3. `ShinkaEvolve/shinka/llm/client.py`

In the `get_client_and_model` function, add OpenRouter handling:

```python
# After the existing model checks, add:
elif model_name in OPENROUTER_MODELS.keys() or model_name.startswith("openrouter/"):
    actual_model = model_name.replace("openrouter/", "")
    client = openai.OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/SakanaAI/ShinkaEvolve",
            "X-Title": "ShinkaEvolve",
        }
    )
    model_name = actual_model
```

Also add the import at the top:
```python
from .models.pricing import OPENROUTER_MODELS
```

### 4. `ShinkaEvolve/shinka/llm/query.py`

Update the query dispatch to handle OpenRouter:

```python
# Add import at the top
from .models.pricing import OPENROUTER_MODELS
from .models import query_openrouter

# In the query function, before the model dispatch logic:
is_openrouter = (model_name in OPENROUTER_MODELS.keys() or model_name.startswith("openrouter/"))

# Then in the dispatch:
if is_openrouter:
    query_fn = query_openrouter
```

## Testing

After applying patches:

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key"
cd experiments/shinka_tsp
python run_evo.py
```

## Supported Models

| Model ID | Description |
|----------|-------------|
| `openrouter/anthropic/claude-opus-4.5` | Claude Opus 4.5 |
| `openrouter/anthropic/claude-sonnet-4` | Claude Sonnet 4 |
| `openrouter/anthropic/claude-3.5-sonnet` | Claude 3.5 Sonnet |
| `openrouter/google/gemini-3-pro-preview` | Gemini 3 Pro |
| `openrouter/openai/gpt-4o` | GPT-4o |

You can add more models by updating the `OPENROUTER_MODELS` dictionary in `pricing.py`.

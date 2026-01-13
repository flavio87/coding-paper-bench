"""
OpenRouter query function using standard OpenAI Chat Completions API.

This file should be placed in: ShinkaEvolve/shinka/llm/models/openrouter.py
"""
import backoff
import openai
from .pricing import OPENROUTER_MODELS
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)

# Default pricing for unknown OpenRouter models
DEFAULT_OPENROUTER_PRICING = {
    "input_price": 3.0 / 1_000_000,
    "output_price": 15.0 / 1_000_000,
}


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"OpenRouter - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=10,
    max_value=30,
    on_backoff=backoff_handler,
)
def query_openrouter(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query OpenRouter model using standard Chat Completions API."""
    # Build messages list
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(msg_history)
    messages.append({"role": "user", "content": msg})

    # Filter out unsupported kwargs for chat completions
    supported_kwargs = {}
    if "max_tokens" in kwargs:
        supported_kwargs["max_tokens"] = kwargs["max_tokens"]
    if "temperature" in kwargs:
        supported_kwargs["temperature"] = kwargs["temperature"]

    if output_model is None:
        # Standard completion
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **supported_kwargs,
        )
        content = response.choices[0].message.content
        new_msg_history = msg_history + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": content},
        ]
    else:
        # Structured output - use JSON mode
        supported_kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **supported_kwargs,
        )
        content = response.choices[0].message.content
        new_msg_history = msg_history + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": content},
        ]

    # Calculate costs
    # Try to get pricing from OPENROUTER_MODELS, otherwise use default
    original_model_key = f"openrouter/{model}"
    if original_model_key in OPENROUTER_MODELS:
        pricing = OPENROUTER_MODELS[original_model_key]
    else:
        pricing = DEFAULT_OPENROUTER_PRICING

    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
    input_cost = pricing["input_price"] * input_tokens
    output_cost = pricing["output_price"] * output_tokens

    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought="",
        model_posteriors=model_posteriors,
    )
    return result

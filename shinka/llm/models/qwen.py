import backoff
import openai
import re
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)

def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Qwen - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=20,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_qwen(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query Qwen model."""
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    args_dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            *new_msg_history,
        ],
        **kwargs,
    }

    if output_model:
        # Use native vLLM structured outputs via OpenAI API compatible extra_body
        json_schema = output_model.model_json_schema()
        # Ensure title is set correctly as some parsers are picky
        if "title" not in json_schema:
            json_schema["title"] = output_model.__name__

        args_dict["extra_body"] = {
            "guided_json": json_schema
        }
        # Force temperature to something reasonable for constrained generation (optional)
        # but respecting kwargs if set.

    extra_body = args_dict.get("extra_body") or {}
    extra_body.setdefault("chat_template_kwargs", {})
    extra_body["chat_template_kwargs"]["enable_thinking"] = True
    args_dict["extra_body"] = extra_body
    
    response = client.chat.completions.create(**args_dict)
    message = response.choices[0].message
    thought, content = message.reasoning, message.content
    new_msg_history.append({"role": "assistant", "content": content})

    if thought is None:
        thought, content = content.split("</think>")
        content = content.replace("</think>", "").strip()
        thought = thought.replace("</think>", "").strip()
    
    # Qwen Local Usage
    # Pricing is 0 for local
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.total_tokens - input_tokens if response.usage else 0
    
    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=0.0,
        input_cost=0.0,
        output_cost=0.0,
        thought=thought,
        model_posteriors=model_posteriors,
    )
    return result

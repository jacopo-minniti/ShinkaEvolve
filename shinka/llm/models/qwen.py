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
    if output_model is None:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            **kwargs,
        )
        try:
            text = response.choices[0].message.content
        except Exception:
            # Fallback or specific handling if needed
            text = ""
        new_msg_history.append({"role": "assistant", "content": text})
    else:
        # Assuming Qwen support for structured output via instructor/patching if client is capable,
        # otherwise raise as Gemini did. For now, sticking to basics or what the user asked (similar to Gemini).
        # Gemini implementation raised ValueError. I will do same unless specified otherwise,
        # but user said "super similar to gemini.py".
        raise ValueError("Qwen does not support structured output yet.")

    # Modified parsing for <think> tag instead of <thought>
    thought_match = re.search(
        r"<think>(.*?)</think>", response.choices[0].message.content, re.DOTALL
    )

    thought = thought_match.group(1) if thought_match else ""

    content_match = re.search(
        r"<think>(.*?)</think>", response.choices[0].message.content, re.DOTALL
    )
    if content_match:
        # Extract everything before and after the <think> tag as content
        content = (
            response.choices[0].message.content[: content_match.start()]
            + response.choices[0].message.content[content_match.end() :]
        ).strip()
    else:
        content = response.choices[0].message.content

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

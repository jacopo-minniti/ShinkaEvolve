from typing import List, Optional, Dict
from pydantic import BaseModel
import openai
import logging
from ..result import QueryResult, TokenUsage

logger = logging.getLogger(__name__)

def query_local(
    client: openai.OpenAI,
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[BaseModel] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    **kwargs,
) -> QueryResult:
    """
    Query a local OpenAI-compatible model (e.g. Unsloth, vLLM).
    Extracts total tokens and thinking tokens (if available) from usage/metadata.
    """

    # Prepare messages
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(msg_history)
    messages.append({"role": "user", "content": msg})

    try:
        if output_model:
            # Using instructor (patched client)
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_model=output_model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            # Instructor returns the Pydantic model directly
            # We don't easily get usage info in this mode unless we inspect raw response
            # defaulting usage to 0 for now in structured mode or need to access ._raw_response
            total_tokens = 0
            thinking_tokens = 0
            # response_text = str(response)  # or serialized JSON
            
            return QueryResult(
                response=response,
                token_usage=TokenUsage(
                    total_tokens=total_tokens,
                    thinking_tokens=thinking_tokens
                )
            )

        else:
            # Standard OpenAI call
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            
            content = response.choices[0].message.content
            
            # Extract usage
            usage = response.usage
            total_tokens = usage.total_tokens if usage else 0
            
            # Attempt to extract thinking tokens if present in usage or specific field
            # Some local servers might put it in completion_tokens_details
            thinking_tokens = 0
            if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                # access attribute safely
                if hasattr(usage.completion_tokens_details, "reasoning_tokens"):
                     thinking_tokens = usage.completion_tokens_details.reasoning_tokens

            return QueryResult(
                response=content,
                token_usage=TokenUsage(
                    total_tokens=total_tokens,
                    thinking_tokens=thinking_tokens
                )
            )

    except Exception as e:
        logger.error(f"Error querying local model {model_name}: {e}")
        raise e

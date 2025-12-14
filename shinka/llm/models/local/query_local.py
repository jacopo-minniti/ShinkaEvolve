from typing import List, Optional, Dict
from pydantic import BaseModel
import openai
import logging
from ..result import QueryResult

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
    
    # Store new history for result
    new_msg_history = msg_history + [{"role": "user", "content": msg}]

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
            # For instructor, we don't get easy access to token usage
            input_tokens = 0
            output_tokens = 0
            
            # Since response is the parsed object, we create a content string representation
            content = str(response) 
            new_msg_history.append({"role": "assistant", "content": content})
            
            return QueryResult(
                content=response, # Instructor mode returns the object
                msg=msg,
                system_msg=system_msg,
                new_msg_history=new_msg_history,
                model_name=model_name,
                kwargs=kwargs,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=0.0,
                input_cost=0.0,
                output_cost=0.0,
                thought="",
                model_posteriors=model_posteriors
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
            new_msg_history.append({"role": "assistant", "content": content})
            
            # Extract usage
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            
            # Attempt to extract thinking tokens if present
            thinking_tokens = 0
            if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                if hasattr(usage.completion_tokens_details, "reasoning_tokens"):
                     thinking_tokens = usage.completion_tokens_details.reasoning_tokens

            return QueryResult(
                content=content,
                msg=msg,
                system_msg=system_msg,
                new_msg_history=new_msg_history,
                model_name=model_name,
                kwargs=kwargs,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=0.0,
                input_cost=0.0,
                output_cost=0.0,
                thought=str(thinking_tokens) if thinking_tokens > 0 else "",
                model_posteriors=model_posteriors
            )

    except Exception as e:
        logger.error(f"Error querying local model {model_name}: {e}")
        raise e

    except Exception as e:
        logger.error(f"Error querying local model {model_name}: {e}")
        raise e

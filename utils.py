import ollama
from ollama._types import Message, ChatResponse
from db_utils import cache_response, get_cached_response
from typing import List, Optional, Generator, Union, Any, Dict

def get_ollama_response(
    prompt: str,
    model: str = "llama2",
    stream: bool = False,
    temperature: float = 0.7,
    context: Optional[List[int]] = None
) -> Union[str, Generator[str, None, None]]:
    """
    Get response from Ollama model with streaming support
    """
    try:
        # Create message object
        message: Message = {"role": "user", "content": prompt}

        # Set model parameters
        options: Dict[str, Any] = {
            "temperature": temperature,
            "context": context
        }

        if stream:
            # Stream response
            response = ollama.chat(
                model=model,
                messages=[message],
                stream=True,
                options=options
            )
            for chunk in response:
                yield chunk['message']['content']
        else:
            # Check cache first for non-streaming responses
            cached_response = get_cached_response(prompt, model)
            if cached_response:
                return cached_response

            # Get chat response from Ollama
            response: ChatResponse = ollama.chat(
                model=model,
                messages=[message],
                options=options
            )
            response_text = response['message']['content']

            # Cache the response
            cache_response(prompt, model, response_text)
            return response_text

    except ollama.ResponseError as e:
        return f"Model Error: {str(e)}"
    except ollama.RequestError as e:
        return f"Request Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

def format_message(message: str, role: str) -> str:
    """
    Format message with markdown and styling
    """
    if role == "user":
        return f"<div class='message-container'><div class='user-message'>{message}</div></div>"
    else:
        return f"<div class='message-container'><div class='assistant-message'>{message}</div></div>"

def get_available_models() -> List[str]:
    """
    Get list of available Ollama models with error handling
    """
    try:
        response = ollama.list()
        return [model['name'] for model in response['models']]
    except Exception as e:
        return ["llama2", "mistral", "codellama"]  # Default fallback models

def get_model_details(model: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model
    """
    try:
        return ollama.show(model=model)
    except Exception:
        return {
            "model": model,
            "details": "Model information unavailable"
        }
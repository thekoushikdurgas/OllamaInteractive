import ollama
from ollama._types import Message, ChatResponse
from db_utils import cache_response, get_cached_response
from typing import List, Optional

def get_ollama_response(prompt: str, model: str = "llama2") -> str:
    """
    Get response from Ollama model with caching
    """
    try:
        # Check cache first
        cached_response = get_cached_response(prompt, model)
        if cached_response:
            return cached_response

        # Create message object
        message: Message = {"role": "user", "content": prompt}

        # Get chat response from Ollama
        response: ChatResponse = ollama.chat(model=model, messages=[message])
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
    Get list of available Ollama models
    """
    try:
        response = ollama.list()
        return [model['name'] for model in response['models']]
    except Exception:
        return ["llama2", "mistral", "codellama"]  # Default fallback models
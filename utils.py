import ollama
from db_utils import cache_response, get_cached_response

def get_ollama_response(prompt: str, model: str = "llama2") -> str:
    """
    Get response from Ollama model with caching
    """
    try:
        # Check cache first
        cached_response = get_cached_response(prompt, model)
        if cached_response:
            return cached_response

        # If not in cache, get from Ollama
        response = ollama.generate(model=model, prompt=prompt)
        response_text = response['response']

        # Cache the response
        cache_response(prompt, model, response_text)

        return response_text
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
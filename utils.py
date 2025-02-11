import ollama

def get_ollama_response(prompt: str, model: str = "llama2") -> str:
    """
    Get response from Ollama model
    """
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response['response']
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

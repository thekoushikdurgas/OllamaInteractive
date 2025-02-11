import ollama
from ollama._types import Message, ChatResponse, Image, Tool, ListResponse
from db_utils import cache_response, get_cached_response
from typing import List, Optional, Generator, Union, Any, Dict
import base64
from pathlib import Path
import logging

logger = logging.getLogger(__name__) #Added logger

async def get_ollama_response_async(
    prompt: str,
    model: str = "llama2",
    stream: bool = False,
    temperature: float = 0.7,
    context: Optional[List[int]] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    image_path: Optional[str] = None,
    use_generate: bool = False,
    use_tools: bool = False
) -> Union[str, Generator[str, None, None]]:
    client = ollama.AsyncClient()
    try:
        current_messages = messages or []
        current_messages.append({"role": "user", "content": prompt})
        if image_path:
            try:
                image = Image(value=Path(image_path))
                message["images"] = [image]
            except Exception as e:
                return f"Image Error: {str(e)}"

        options = {"temperature": temperature, "context": context}

        if stream:
            response = await client.chat(
                model=model,
                messages=[message],
                stream=True,
                options=options
            )
            async for chunk in response:
                yield chunk['message']['content']
        else:
            cached_response = get_cached_response(prompt, model)
            if cached_response:
                return cached_response

            if use_generate:
                response = await client.generate(
                    model=model,
                    prompt=prompt,
                    options=options
                )
                response_text = response['response']
            else:
                from tools import AVAILABLE_TOOLS, TOOL_DEFINITIONS

                chat_options = {**options}
                if use_tools:
                    chat_options['tools'] = TOOL_DEFINITIONS

                response = await client.chat(
                    model=model,
                    messages=[message],
                    options=chat_options
                )

                if response.message.tool_calls and use_tools:
                    tool_outputs = []
                    for tool in response.message.tool_calls:
                        if function_to_call := AVAILABLE_TOOLS.get(tool.function.name):
                            try:
                                output = function_to_call(**tool.function.arguments)
                                tool_outputs.append({
                                    'role': 'tool',
                                    'content': str(output),
                                    'name': tool.function.name
                                })
                            except Exception as e:
                                logger.error(f"Tool call failed: {str(e)}")

                    if tool_outputs:
                        messages = [message] + [response.message] + tool_outputs
                        final_response = await client.chat(model=model, messages=messages)
                        response_text = final_response.message.content
                    else:
                        response_text = response.message.content
                else:
                    response_text = response.message.content
            cache_response(prompt, model, response_text)
            return response_text

    except Exception as e:
        return f"Error: {str(e)}"

def get_ollama_response(
    prompt: str,
    model: str = "llama2",
    stream: bool = False,
    temperature: float = 0.7,
    context: Optional[List[int]] = None,
    image_path: Optional[str] = None
) -> Union[str, Generator[str, None, None]]:
    """
    Get response from Ollama model with streaming support and image handling.

    Args:
        prompt: Text prompt to send to the model
        model: Name of the Ollama model to use
        stream: Whether to stream the response
        temperature: Controls randomness in the response
        context: Optional context from previous interactions
        image_path: Optional path to image file for multimodal models

    Returns:
        Either a string response or a generator of response chunks if streaming
    """
    try:
        # Create message object
        message: Message = {"role": "user", "content": prompt}

        # Handle image if provided
        if image_path:
            try:
                image = Image(value=Path(image_path))
                message["images"] = [image]
            except Exception as e:
                return f"Image Error: {str(e)}"

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
    """Format message with markdown and styling"""
    if role == "user":
        return f"<div class='message-container'><div class='user-message'>{message}</div></div>"
    else:
        return f"<div class='message-container'><div class='assistant-message'>{message}</div></div>"

def get_available_models() -> List[Dict[str, Any]]:
    """
    Get list of available Ollama models with detailed information.

    Returns:
        List of dictionaries containing detailed model information
    """
    try:
        response: ListResponse = ollama.list()
        models = []
        # Sort models by name for consistent display
        for model in sorted(response.models, key=lambda x: x.model):
            model_info = {
                'name': model.model,
                'size_mb': f'{(model.size.real / 1024 / 1024):.2f}',
                'details': {}
            }
            if model.details:
                model_info['details'] = {
                    'format': model.details.format,
                    'family': model.details.family,
                    'parameter_size': model.details.parameter_size,
                    'quantization_level': model.details.quantization_level
                }
            models.append(model_info)
        logger.info(f"Successfully fetched {len(models)} models")
        return models
    except Exception as e:
        logger.error(f"Failed to fetch models: {str(e)}")
        # Return default models with minimal info if fetch fails
        return [
            {'name': name, 'size_mb': 'N/A', 'details': {}} 
            for name in ["llama2", "mistral", "codellama"]
        ]

def get_model_details(model: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.

    Args:
        model: Name of the model to get information about

    Returns:
        Dictionary containing formatted model details
    """
    try:
        details = ollama.show(model=model)
        # Format the details for better display
        formatted_details = {
            "Model Name": model,
            "Size": details.size.real / 1024 / 1024 if hasattr(details, 'size') else 'Unknown',
            "Family": details.get('details', {}).get('family', 'Unknown'),
            "Format": details.get('details', {}).get('format', 'Unknown'),
            "Parameter Size": details.get('details', {}).get('parameter_size', 'Unknown'),
            "Quantization": details.get('details', {}).get('quantization_level', 'None'),
            "License": details.get('license', 'Unknown'),
            "Modified": details.get('modified_at', 'Unknown')
        }
        return formatted_details
    except Exception as e:
        logger.error(f"Failed to get model details: {str(e)}")
        return {
            "Model Name": model,
            "Status": "Details unavailable",
            "Error": str(e)
        }

def encode_image(image_path: str) -> Optional[str]:
    """
    Encode image to base64 for UI display.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image or None if encoding fails
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception:
        return None

def create_tool_from_function(func: Any) -> Tool:
    """
    Convert a Python function to an Ollama chat tool.

    Args:
        func: Python function to convert to a tool

    Returns:
        Ollama Tool object that can be used in chat conversations
    """
    try:
        from inspect import getdoc, signature

        # Get function signature and docstring
        sig = signature(func)
        doc = getdoc(func) or ""

        # Create tool parameters
        parameters = {
            name: {
                "type": str(param.annotation.__name__ if param.annotation != param.empty else "string"),
                "description": ""
            }
            for name, param in sig.parameters.items()
        }

        # Create tool object
        tool = Tool(
            type="function",
            function=Tool.Function(
                name=func.__name__,
                description=doc.split("\n")[0],
                parameters=Tool.Function.Parameters(
                    type="object",
                    properties=parameters,
                    required=[name for name, param in sig.parameters.items()
                            if param.default == param.empty]
                )
            )
        )

        return tool
    except Exception as e:
        raise ValueError(f"Failed to create tool from function: {str(e)}")
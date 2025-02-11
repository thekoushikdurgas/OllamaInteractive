
from typing import Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)

def add_two_numbers(a: int, b: int) -> int:
    return a + b

def subtract_two_numbers(a: int, b: int) -> int:
    return a - b

AVAILABLE_TOOLS: Dict[str, Callable] = {
    'add_two_numbers': add_two_numbers,
    'subtract_two_numbers': subtract_two_numbers,
}

TOOL_DEFINITIONS = [
    {
        'type': 'function',
        'function': {
            'name': 'add_two_numbers',
            'description': 'Add two numbers',
            'parameters': {
                'type': 'object',
                'required': ['a', 'b'],
                'properties': {
                    'a': {'type': 'integer', 'description': 'First number'},
                    'b': {'type': 'integer', 'description': 'Second number'},
                },
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'subtract_two_numbers',
            'description': 'Subtract two numbers',
            'parameters': {
                'type': 'object',
                'required': ['a', 'b'],
                'properties': {
                    'a': {'type': 'integer', 'description': 'First number'},
                    'b': {'type': 'integer', 'description': 'Second number'},
                },
            },
        },
    }
]

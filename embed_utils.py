
from ollama import embed
from typing import List, Dict, Any
import numpy as np
from db_utils import get_db_connection
import logging

logger = logging.getLogger(__name__)

def get_embedding(text: str, model: str = "llama2") -> List[float]:
    """Get embeddings for text using Ollama model"""
    try:
        response = embed(model=model, input=text)
        return response['embeddings']
    except Exception as e:
        logger.error(f"Failed to get embedding: {str(e)}")
        return []

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if not a or not b:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def save_message_with_embedding(message: str, role: str, model: str):
    """Save message with its embedding to MongoDB"""
    try:
        db = get_db_connection()
        if db is not None:
            embedding = get_embedding(message, model)
            message_doc = {
                'content': message,
                'role': role,
                'model': model,
                'embedding': embedding,
                'timestamp': datetime.utcnow()
            }
            db['messages'].insert_one(message_doc)
            return True
    except Exception as e:
        logger.error(f"Failed to save message with embedding: {str(e)}")
    return False

def find_similar_messages(query: str, model: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Find similar messages using embeddings"""
    try:
        db = get_db_connection()
        if db is not None:
            query_embedding = get_embedding(query, model)
            messages = list(db['messages'].find({'model': model}))
            
            # Calculate similarities
            similarities = [
                (msg, cosine_similarity(query_embedding, msg.get('embedding', [])))
                for msg in messages
            ]
            
            # Sort by similarity
            similar_messages = sorted(similarities, key=lambda x: x[1], reverse=True)[:limit]
            return [{'content': msg['content'], 'similarity': sim} 
                   for msg, sim in similar_messages]
    except Exception as e:
        logger.error(f"Failed to find similar messages: {str(e)}")
    return []

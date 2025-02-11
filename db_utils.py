from pymongo import MongoClient
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection with fallback to local instance
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')

def get_db_connection():
    """Get MongoDB connection with error handling"""
    try:
        client = MongoClient(MONGODB_URI)
        # Test the connection
        client.admin.command('ping')
        db = client['chat_app']
        logger.info("Successfully connected to MongoDB")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

def save_message(message_content: str, role: str, model: str):
    """Save a message to MongoDB"""
    try:
        db = get_db_connection()
        if db is not None:  # Fixed boolean check
            messages = db['messages']
            message_doc = {
                'content': message_content,
                'role': role,
                'model': model,
                'timestamp': datetime.utcnow()
            }
            messages.insert_one(message_doc)
            logger.info(f"Message saved successfully: {role}")
            return True
    except Exception as e:
        logger.error(f"Failed to save message: {str(e)}")
    return False

def get_chat_history():
    """Retrieve chat history from MongoDB"""
    try:
        db = get_db_connection()
        if db is not None:  # Fixed boolean check
            messages = db['messages']
            return list(messages.find().sort('timestamp', 1))
        return []
    except Exception as e:
        logger.error(f"Failed to retrieve chat history: {str(e)}")
        return []

def cache_response(prompt: str, model: str, response: str):
    """Cache an Ollama response"""
    try:
        db = get_db_connection()
        if db is not None:  # Fixed boolean check
            cache = db['response_cache']
            cache_doc = {
                'prompt': prompt,
                'model': model,
                'response': response,
                'timestamp': datetime.utcnow()
            }
            cache.insert_one(cache_doc)
            logger.info("Response cached successfully")
    except Exception as e:
        logger.error(f"Failed to cache response: {str(e)}")

def get_cached_response(prompt: str, model: str):
    """Get cached response if available"""
    try:
        db = get_db_connection()
        if db is not None:  # Fixed boolean check
            cache = db['response_cache']
            cached = cache.find_one({
                'prompt': prompt,
                'model': model
            })
            if cached:
                logger.info("Cache hit: Found cached response")
                return cached['response']
    except Exception as e:
        logger.error(f"Failed to retrieve cached response: {str(e)}")
    return None

def clear_chat_history():
    """Clear all chat messages"""
    try:
        db = get_db_connection()
        if db is not None:  # Fixed boolean check
            messages = db['messages']
            messages.delete_many({})
            logger.info("Chat history cleared successfully")
            return True
    except Exception as e:
        logger.error(f"Failed to clear chat history: {str(e)}")
    return False